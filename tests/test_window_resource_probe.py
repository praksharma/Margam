"""Native Expert handoff accounting, exact instrumentation parity and disposable UI."""
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import torch
from margam.execution.probe import begin_measurement,finish_measurement,validate_probe,probe_limit,probe_study
from margam.execution.probe_report import validate_window_report
from margam.execution.codec import encode_config,cpu_tree
from margam.execution.transport import JobTransport,JobFailed
from margam.templates.poisson import build_pinn
try:
    from .test_expert_extensions import config as expert_config
except ImportError:
    from test_expert_extensions import config as expert_config


def config(device='cpu',case='expert_burgers',mode='ntk',steps=2,windows=3):
    c=expert_config(case,device,windows=windows);c.expert_settings.update(window_steps=steps,balance_mode=mode,balance_every=2);c.iterations=steps*windows;return c


def payload(c,updates):return dict(operation='probe',config=encode_config(c),requests=[],checkpoint=None,metadata={},dtype='float32',probe_updates=updates)


def equal_state(test,a,b,path='state'):
    if isinstance(a,torch.Tensor):test.assertTrue(torch.equal(a,b),path)
    elif isinstance(a,dict):
        test.assertEqual(set(a),set(b),path)
        for key in a:equal_state(test,a[key],b[key],path+'.'+str(key))
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b),path)
        for i,(x,y) in enumerate(zip(a,b)):equal_state(test,x,y,path+'.'+str(i))
    else:test.assertEqual(a,b,path)


def teacher_states(model):
    return [dict(identity=t.identity,components=t.components,network=cpu_tree(t.network.state_dict()),
        cached={key:cpu_tree(network.state_dict()) for key,network in t.cache.items()},
        frozen=[p.requires_grad for p in t.network.parameters()],training=t.network.training)
        for t in model.teachers]


class WindowProbeMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def parity(self,device):
        for case in ('expert_burgers','expert_cylinder'):
            for mode in ('gradient','ntk'):
                with self.subTest(device=device,case=case,mode=mode):
                    c=config(device,case,mode);before=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
                    try:
                        full=build_pinn(c);full.train(6,0);expected=cpu_tree(full.training_state())
                        low,high=c.bounds;points=[[(a+b)/2 for a,b in zip(low,high)] for _ in range(3)]
                        for i,row in enumerate(points):row[-1]=low[-1]+(high[-1]-low[-1])*(i+.5)/3
                        predictions=full.predict(points).detach().cpu();expected_teachers=teacher_states(full);cpu_after=torch.get_rng_state().clone();gpu_after=torch.cuda.get_rng_state().clone() if gpu is not None else None
                        del full;gc.collect();torch.set_rng_state(before)
                        if gpu is not None:torch.cuda.set_rng_state(gpu)
                        measurement=begin_measurement(c);model=build_pinn(c);measurement['windows'].attach(model);model.train(6,0)
                        report=finish_measurement(measurement,model,6,len(model.collocation_point_sample));validate_window_report(report)
                        equal_state(self,expected,cpu_tree(model.training_state()));self.assertTrue(torch.equal(predictions,model.predict(points).detach().cpu()));equal_state(self,expected_teachers,teacher_states(model),'teachers')
                        self.assertTrue(torch.equal(cpu_after,torch.get_rng_state()))
                        if gpu_after is not None:self.assertTrue(torch.equal(gpu_after,torch.cuda.get_rng_state()))
                        self.assertIsNone(model._transition_observer);self.assertEqual(report['observed_transfers'],2);self.assertEqual(report['teacher_count'],2)
                        self.assertTrue(all(not p.requires_grad for teacher in model.teachers for p in teacher.network.parameters()))
                        self.assertEqual([p['kind'] for p in report['phases']],['construction','window_training','teacher_transfer','window_training','teacher_transfer','window_training'])
                        self.assertEqual([p['end_update'] for p in report['phases']],[0,2,2,4,4,6])
                        del model;gc.collect()
                    finally:
                        torch.set_rng_state(before)
                        if gpu is not None:torch.cuda.set_rng_state(gpu)
    def test_cpu_instrumentation_preserves_all_checkpoint_fields_and_piecewise_predictions(self):self.parity('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_instrumentation_preserves_weights_adam_sampling_balancing_teachers_and_rng(self):self.parity('cuda')

    def test_end_before_at_after_boundaries_and_final_window_extension(self):
        for n,transfers,pending in [(1,0,False),(2,0,True),(3,1,False),(4,1,True),(5,2,False),(6,2,False),(7,2,False)]:
            with self.subTest(updates=n):
                c=config();c.iterations=7;m=begin_measurement(c);model=build_pinn(c);initial=len(model.collocation_point_sample);m['windows'].attach(model);model.train(n,0)
                r=finish_measurement(m,model,n,initial);self.assertEqual(r['observed_transfers'],transfers);self.assertEqual(r['next_transfer_pending'],pending);self.assertEqual(r['completed_updates'],n);validate_window_report(r)

    def test_budget_caps_and_transition_failure_never_form_complete_measurement(self):
        c=config(steps=1000);self.assertEqual(probe_limit(c),3000);validate_probe(c,1001)
        for n in (0,True,3001):
            with self.assertRaises(ValueError):validate_probe(c,n)
        c.expert_settings['windows']=1;c.iterations=2000;self.assertEqual(probe_limit(c),1000)
        with self.assertRaises(ValueError):validate_probe(c,1001)
        c=config();measurement=begin_measurement(c);model=build_pinn(c);measurement['windows'].attach(model)
        with patch.object(model,'_advance_window',side_effect=RuntimeError('injected handoff failure')):
            with self.assertRaisesRegex(RuntimeError,'handoff failure'):model.train(3,0)
        self.assertIn('window 2',measurement['windows'].failure_stage('training'))
        with self.assertRaises(ValueError):finish_measurement(measurement,model,3,len(model.collocation_point_sample))
        self.assertIsNone(model._transition_observer)

    def test_report_rejects_partial_reordered_and_misleading_peaks_and_is_data_only(self):
        c=config();m=begin_measurement(c);model=build_pinn(c);m['windows'].attach(model);model.train(5,0);r=finish_measurement(m,model,5,len(model.collocation_point_sample));before=deepcopy(r)
        self.assertIs(validate_window_report(r),r);self.assertEqual(before,r)
        mutations=[lambda v:v.update(version=True),lambda v:v.update(observed_transfers=1),lambda v:v.update(next_transfer_pending=True),lambda v:v['phases'].pop(),lambda v:v['phases'].reverse(),lambda v:v['phases'][2].update(start_update=1),lambda v:v['phases'][2].update(teacher_count=0),lambda v:v['phases'][2].update(time_interval=[0.,.1]),lambda v:v['phases'][1].update(duration_seconds=float('nan')),lambda v:v['phases'][1].update(cuda_peak_allocated_bytes=1),lambda v:v.update(cuda_peak_allocated_bytes=0),lambda v:v.update(initial_pde_rows=0),lambda v:v.update(device='invalid')]
        for mutate in mutations:
            bad=deepcopy(r);mutate(bad)
            with self.assertRaises(ValueError):validate_window_report(bad)
        synthetic=deepcopy(r);synthetic['device']='cuda:0'
        for i,phase in enumerate(synthetic['phases']):
            for counter,extra in [('allocated',0),('reserved',100)]:
                phase[f'cuda_start_{counter}_bytes']=100*i+extra;phase[f'cuda_peak_{counter}_bytes']=100*i+extra+50;phase[f'cuda_end_{counter}_bytes']=100*i+extra+20
        for counter in ('allocated','reserved'):synthetic[f'cuda_peak_{counter}_bytes']=max(p[f'cuda_peak_{counter}_bytes'] for p in synthetic['phases'])
        validate_window_report(synthetic)
        zero=deepcopy(synthetic)
        for phase in zero['phases']:
            for key in phase:
                if key.startswith('cuda_'):phase[key]=0
        zero['cuda_peak_allocated_bytes']=False;zero['cuda_peak_reserved_bytes']=0
        with self.assertRaises(ValueError):validate_window_report(zero)
        synthetic['cuda_peak_allocated_bytes']=sum(p['cuda_peak_allocated_bytes'] for p in synthetic['phases'])
        with self.assertRaises(ValueError):validate_window_report(synthetic)
        giant=deepcopy(r);giant['phases'][0]['duration_seconds']=10**400
        with self.assertRaises(ValueError):validate_window_report(giant)
        child=subprocess.run([sys.executable,'-c',"import sys; from margam.execution.probe_report import validate_window_report; assert 'torch' not in sys.modules and 'PySide6' not in sys.modules"],capture_output=True,text=True,timeout=15);self.assertEqual(child.returncode,0,child.stderr)


class WindowProbeWorkerTests(unittest.TestCase):
    def run_worker(self,device):
        c=config(device);events=[];packet=JobTransport(wall_time_seconds=60).run(payload(c,5),events.append);r=packet['result'];validate_window_report(r)
        self.assertEqual(r['version'],2);self.assertEqual(r['observed_windows'],3);self.assertIsNone(packet['checkpoint']);self.assertEqual(packet['metadata'],{});self.assertTrue(packet['headless'])
        self.assertEqual(r['cached_teacher_copies'],0 if device=='cpu' else 2)
        if device=='cuda':
            self.assertGreater(r['cuda_peak_allocated_bytes'],0)
            for counter in ('allocated','reserved'):self.assertEqual(r[f'cuda_peak_{counter}_bytes'],max(p[f'cuda_peak_{counter}_bytes'] for p in r['phases']))
        self.assertTrue(r['internal_window_snapshots_included']);self.assertFalse(r['checkpoint_retained'])
    def test_cpu_isolated_worker_reports_full_handoff_phases_without_retained_state(self):self.run_worker('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_isolated_worker_measures_resident_teacher_copies_and_maximum_peaks(self):self.run_worker('cuda')
    def test_maximum_window_report_is_bounded_and_native_state_is_discarded(self):
        c=config(steps=1,windows=32);r=JobTransport(wall_time_seconds=60).run(payload(c,32))['result'];validate_window_report(r)
        self.assertEqual(len(r['phases']),64);self.assertEqual(r['observed_transfers'],31);self.assertLess(len(json.dumps(r,indent=2).encode()),262144)
        from margam.ui.study_execution import StudyProcess
        worker=StudyProcess([],None,tail_limit=262144);self.assertEqual(worker.tail_limit,262144);worker.deleteLater()
        for limit in (0,True,262145):
            with self.assertRaises(ValueError):StudyProcess([],None,tail_limit=limit)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_maximum_windows_track_all_resident_teacher_copies(self):
        c=config('cuda',steps=1,windows=32);r=JobTransport(wall_time_seconds=60).run(payload(c,32))['result'];validate_window_report(r)
        self.assertEqual(len(r['phases']),64);self.assertEqual(r['teacher_count'],31);self.assertEqual(r['cached_teacher_copies'],31)
        self.assertEqual(r['cuda_peak_allocated_bytes'],max(p['cuda_peak_allocated_bytes'] for p in r['phases']));self.assertLess(len(json.dumps(r,indent=2).encode()),262144)

    def test_cancelled_multiwindow_job_returns_no_measurement(self):
        c=config();c.iterations=100000;transport=JobTransport(wall_time_seconds=60)
        def event(value):
            if value['kind']=='progress':transport.cancel()
        with self.assertRaisesRegex(JobFailed,'cancelled'):transport.run(payload(c,100000),event)
        self.assertIsNone(transport.process)


try:
    from . import test_resource_probe_ui as fixture
except ImportError:
    import test_resource_probe_ui as fixture
from margam.ui.model_document import capture_model
from margam.ui.app import Window
from margam.studies import catalogue
from margam.study_document import capture,write


@unittest.skipUnless(os.name=='posix','POSIX desktop supervision required')
class WindowProbeUITests(unittest.TestCase):
    setUpClass=classmethod(fixture.ResourceProbeUITests.setUpClass.__func__)
    setUp=fixture.ResourceProbeUITests.setUp
    tearDown=fixture.ResourceProbeUITests.tearDown
    wait=fixture.ResourceProbeUITests.wait
    wait_until=fixture.ResourceProbeUITests.wait_until
    def make_study(self,windows=3,steps=2):
        source=Window(state_directory=False)
        try:
            source.load_example('expert_burgers')
            for key,value in [('points',8),('boundary_points',2),('layers',(4,4))]:source.change(key,value)
            options=dict(source.config.network_options);options['features']=2;source.change('network_options',options)
            for key,value in [('windows',windows),('window_steps',steps),('chunks',2),('balance_mode','ntk'),('balance_every',2),('ntk_points',1)]:source.change_expert_option(key,value)
            model=capture_model(source);axis=next(a for a in catalogue(model) if a['parameter']=='network_seed')
            study,files=capture(model,[dict(node_id=axis['node_id'],parameter='network_seed',mode='fixed',text='0')],source.project_workspace,execution=dict(device='cpu',wall_time_seconds=0),scene=source.scene)
            path=self.root/'windows.margam-study';write(path,study,files);return path
        finally:source.undo_stack.setClean();source.close()
    def exercise(self,device):
        self.assertTrue(self.w.start_training());self.wait();self.w.set_application_device(device)
        path=self.make_study();original=path.read_bytes();model=capture_model(self.w);undo=self.w.undo_stack.index();old_result=self.w.result;old_last=self.w.last_run;cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
        d=self.w.show_resource_probe();d.source.setText(str(path));d.updates.setValue(5);self.assertTrue(d.review());self.wait_until(lambda:d.reader is None);self.assertIsNotNone(d.prepared,d.status.text());self.assertEqual(d.prepared['description']['reachable_transfers'],2)
        self.assertTrue(d.start());self.wait_until(lambda:not d.active);self.assertIsNotNone(d.result,d.text.toPlainText());validate_window_report(d.result)
        self.assertEqual(d.phases.rowCount(),6);self.assertFalse(d.phases.isHidden());self.assertFalse(d.phase_scope.isHidden());self.assertEqual(d.phases.item(2,0).text(),'teacher transfer');self.assertEqual(d.phases.item(2,1).text(),'1 → 2');self.assertTrue(d.copy_report());self.assertEqual(json.loads(self.app.clipboard().text()),d.result)
        self.assertEqual(original,path.read_bytes());self.assertEqual(model,capture_model(self.w));self.assertEqual(undo,self.w.undo_stack.index());self.assertIs(old_result,self.w.result);self.assertIs(old_last,self.w.last_run);self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
        if gpu is not None:self.assertTrue(torch.equal(gpu,torch.cuda.get_rng_state()))
        d.updates.setValue(1);self.assertIsNone(d.result);self.assertTrue(d.phases.isHidden());self.assertTrue(d.phase_scope.isHidden());self.assertEqual(d.phases.rowCount(),0)
    def test_maximum_phase_report_reaches_gui_and_copy_without_header_truncation(self):
        path=self.make_study(32,1);d=self.w.show_resource_probe();d.source.setText(str(path));d.updates.setValue(32)
        self.assertTrue(d.review());self.wait_until(lambda:d.reader is None);self.assertIsNotNone(d.prepared,d.status.text());self.assertTrue(d.start());self.wait_until(lambda:not d.active)
        self.assertIsNotNone(d.result,d.text.toPlainText());self.assertEqual(d.phases.rowCount(),64);self.assertIn('32 complete updates/attempts',d.text.toPlainText());self.assertTrue(d.copy_report());self.assertEqual(len(json.loads(self.app.clipboard().text())['phases']),64)

    def test_cpu_gui_review_phase_table_copy_and_existing_weights_are_preserved(self):self.exercise('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_gui_phase_table_does_not_adopt_profiled_weights_or_change_rng(self):self.exercise('cuda')


if __name__=='__main__':unittest.main()
