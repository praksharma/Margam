"""Disposable native update probes measure work without retaining model state."""
from copy import deepcopy
import json
import os
import unittest
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import torch
import test_study_cli as fixture
from margam.execution.probe import probe_study,validate_probe
from margam.execution.transport import JobTransport,JobFailed
from margam.execution.codec import encode_config
from margam.templates.poisson import PoissonConfig
from margam.ui.model_document import capture_model


class ResourceProbeTests(unittest.TestCase):
    setUpClass=classmethod(fixture.StudyCLITests.setUpClass.__func__)
    setUp=fixture.StudyCLITests.setUp
    tearDown=fixture.StudyCLITests.tearDown
    wait=fixture.StudyCLITests.wait
    study=fixture.StudyCLITests.study
    cli=fixture.StudyCLITests.cli

    def payload(self,c,updates=1):return dict(operation='probe',config=encode_config(c),requests=[],checkpoint=None,metadata={},dtype='float32',probe_updates=updates)
    def assert_probe(self,result,device,steps=1):
        self.assertEqual(result['format'],'margam-resource-probe');self.assertEqual(result['completed_updates'],steps);self.assertFalse(result['checkpoint_retained']);self.assertFalse(result['capacity_prediction']);self.assertIn(device,result['device']);self.assertGreater(result['build_and_train_seconds'],0)
        if os.name=='posix':self.assertGreater(result['process_peak_rss_bytes'],0)
        if device=='cuda':self.assertGreater(result['cuda_peak_allocated_bytes'],0);self.assertGreaterEqual(result['cuda_peak_reserved_bytes'],result['cuda_peak_allocated_bytes'])
        else:self.assertIsNone(result['cuda_peak_allocated_bytes'])

    def exercise(self,device):
        path=self.study('[2,3]');original=path.read_bytes();model=capture_model(self.w);undo=self.w.undo_stack.index();cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
        result=probe_study(path,trial=2,updates=1,device=device)
        self.assert_probe(result,device);self.assertEqual(result['configured_updates'],3);self.assertTrue(result['trial_id'].startswith('DP0002-'));self.assertEqual(result['requested_output_count'],0)
        self.assertEqual(original,path.read_bytes());self.assertEqual(model,capture_model(self.w));self.assertEqual(undo,self.w.undo_stack.index());self.assertTrue(torch.equal(cpu,torch.get_rng_state()));self.assertIsNone(self.w.worker)
        if gpu is not None:self.assertTrue(torch.equal(gpu,torch.cuda.get_rng_state()))
    def test_cpu_study_probe_preserves_original_budget_graph_rng_and_files(self):self.exercise('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_study_probe_allocator_peaks_and_parent_state(self):self.exercise('cuda')

    def test_complete_method_specific_updates_use_native_worker_without_checkpoint(self):
        for method in ('baseline','self_adaptive','lra','expert','scale','gpinn','rba','dsgnar'):
            with self.subTest(method=method):
                c=PoissonConfig(shape='Interval',device='cpu',points=6,boundary_points=2,layers=(4,),iterations=2,method=method)
                if method in ('expert','scale'):
                    from margam.templates.examples import example_config
                    c=example_config('expert_burgers' if method=='expert' else 'scale_allen_cahn');c.device='cpu';c.points=8;c.boundary_points=2;c.boundary_sampling={};c.layers=(4,4);c.network_options['features']=2;c.iterations=2
                    if method=='expert':
                        from margam.templates.poisson import BoundarySampling
                        c.expert_settings['chunks']=2;c.boundary_sampling={side:BoundarySampling(2,'random',10+i) for i,side in enumerate(c.sides)}
                packet=JobTransport(wall_time_seconds=45).run(self.payload(c));self.assert_probe(packet['result'],'cpu');self.assertIsNone(packet['checkpoint']);self.assertEqual(packet['metadata'],{})

    def test_probe_includes_scheduled_adaptive_scoring_and_growth(self):
        from margam.components.adaptive_sampling import sampling_defaults
        for mode in ('rar','importance'):
            settings=sampling_defaults(mode);settings.update(interval=1,candidates=12,chunk=4)
            if mode=='rar':settings.update(batch=2,maximum=20)
            c=PoissonConfig(shape='Interval',device='cpu',points=6,layers=(4,),iterations=4,adaptive_sampling=settings)
            result=JobTransport().run(self.payload(c,2))['result']
            self.assert_probe(result,'cpu',2);self.assertGreater(result['observed_sampling_events'],0)
            if mode=='rar':self.assertGreater(result['final_pde_rows'],result['initial_pde_rows'])

    def test_bounds_windowed_rejection_checkpoint_rejection_and_cancel(self):
        c=PoissonConfig(shape='Interval',device='cpu',points=6,layers=(4,),iterations=2)
        for updates in (0,True,1001,3):
            with self.assertRaises(ValueError):validate_probe(c,updates)
        c.method='expert';c.expert_settings['windows']=2
        with self.assertRaisesRegex(ValueError,'single Expert'):validate_probe(c,1)
        c.method='baseline';payload=self.payload(c);payload['checkpoint']={}
        with self.assertRaisesRegex(JobFailed,'cannot restore'):JobTransport().run(payload)
        c.iterations=1000;transport=JobTransport();cancelled=[]
        def event(value):
            if value['kind']=='progress':cancelled.append(value['step']);transport.cancel()
        with self.assertRaisesRegex(JobFailed,'cancelled'):transport.run(self.payload(c,1000),event)
        self.assertTrue(cancelled);self.assertIsNone(transport.process)
        with self.assertRaisesRegex(JobFailed,'time limit'):probe_study(self.study('[2]'),wall_time_seconds=.01)

    def test_cli_json_is_headless_and_output_evaluation_is_explicitly_excluded(self):
        self.w.load_burgers_reference_example()
        for key,value in [('points',8),('boundary_points',4),('boundary_sampling',{}),('layers',(4,))]:self.w.change(key,value)
        path=self.study('[2]');before={p.relative_to(self.root):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        child=self.cli('study','probe',path,'--updates','1','--device','cpu');self.assertEqual(child.returncode,0,child.stderr);result=json.loads(child.stdout);self.assert_probe(result,'cpu');self.assertGreater(result['requested_output_count'],0);self.assertIn('output evaluation and exports',result['excluded'])
        self.assertEqual(before,{p.relative_to(self.root):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        child=self.cli('study','probe',path,'--updates','3');self.assertEqual(child.returncode,2);self.assertEqual(child.stdout,'')

    def test_source_change_refuses_publication_and_failed_jobs_return_no_measurement(self):
        path=self.study('[2]')
        with patch('margam.execution.study_runner.source_matches',side_effect=[True,False]):
            with self.assertRaisesRegex(RuntimeError,'changed during'):probe_study(path)
        with patch.object(JobTransport,'run',side_effect=JobFailed('injected allocation failure')):
            with self.assertRaisesRegex(JobFailed,'allocation'):probe_study(path)
        for kwargs in (dict(trial=0),dict(trial=True),dict(trial=3),dict(device='invalid'),dict(wall_time_seconds=-1)):
            with self.assertRaises(ValueError):probe_study(path,**kwargs)


if __name__=='__main__':unittest.main()
