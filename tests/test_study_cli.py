"""Real headless CLI/native parity, isolation, exports and interruption reporting."""
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.model_document import capture_model
from margam.studies import catalogue
from margam.study_document import capture,write,read
from margam.execution.codec import encode_request
from margam.execution.graph import DocumentGraph,prepare_graph
from margam.execution.study_runner import run_study


class StudyCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        self.w=Window(state_directory=False);self.w.set_application_device('cpu');self.folder=tempfile.TemporaryDirectory();self.root=Path(self.folder.name)
        for key,value in [('points',8),('boundary_points',4),('layers',(4,)),('iterations',2)]:self.w.change(key,value)
    def tearDown(self):
        if self.w.worker is not None:self.w.cancel_training();self.wait()
        self.w.undo_stack.setClean();self.w.close();self.app.clipboard().clear();self.app.processEvents();self.folder.cleanup()
    def wait(self):
        deadline=time.monotonic()+45
        while self.w.worker is not None and time.monotonic()<deadline:self.app.processEvents();time.sleep(.005)
        self.assertIsNone(self.w.worker)
    def study(self,values='[2,3]',parameter='iterations',*,wall=0):
        model=capture_model(self.w);axis=next(item for item in catalogue(model) if item['parameter']==parameter)
        axes=[dict(node_id=axis['node_id'],parameter=parameter,mode='candidates',text=values)]
        study,files=capture(model,axes,self.w.project_workspace,execution=dict(device='cpu',wall_time_seconds=wall),scene=self.w.scene)
        path=self.root/'input.margam-study';write(path,study,files,overwrite=True);return path
    def cli(self,*args,timeout=100,env=None):
        return subprocess.run([sys.executable,'-m','margam.cli',*map(str,args)],capture_output=True,text=True,timeout=timeout,
            env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1',**(env or {})))
    def test_shared_graph_requests_and_validation_match_desktop(self):
        for key in ('interval_poisson','burgers_1d','expert_cavity','scale_cavity_400','dsgnar_poisson_1d'):
            from margam.templates.examples import EXAMPLE_PROBLEMS
            self.assertIn(key,EXAMPLE_PROBLEMS)
            self.w.undo_stack.setClean();self.assertTrue(self.w.load_example(key))
            graph=DocumentGraph(capture_model(self.w),self.w.project_workspace,'cpu')
            config,requests=prepare_graph(graph);native,n=self.w.scene.compile_config(),self.w.scene.output_requests(self.w.scene.compile_config())
            self.assertEqual(config,native)
            self.assertEqual({r.identifier:encode_request(r) for r in requests},{r.identifier:encode_request(r) for r in n})
        self.w.load_burgers_reference_example()
        graph=DocumentGraph(capture_model(self.w),self.w.project_workspace,'cpu');config,requests=prepare_graph(graph)
        self.assertEqual({r.identifier:encode_request(r) for r in requests},{r.identifier:encode_request(r) for r in self.w.scene.output_requests(config)})
        comparison=next(n for n in self.w.scene.nodes() if n.kind=='comparison');comparison.settings['enabled']=False
        for target in (self.w.scene,DocumentGraph(capture_model(self.w),self.w.project_workspace,'cpu')):
            with self.assertRaisesRegex(ValueError,'disabled Comparison'):prepare_graph(target)
    def test_cli_mixed_grid_continues_valid_trials_and_refuses_existing_output(self):
        path=self.study('[[4],[-1],[8]]','layers');output=self.root/'mixed'
        result=self.cli('study','run',path,'--output',output,'--device','cpu');self.assertEqual(result.returncode,1,result.stderr)
        report=json.loads((output/'summary.json').read_text());self.assertTrue(report['headless']);self.assertEqual(report['counts'],dict(completed=2,invalid=1))
        self.assertEqual([row['status'] for row in report['trials']],['completed','invalid','completed'])
        self.assertIsNone(report['trials'][1]['metrics']);self.assertFalse((output/'DP0002/artifacts').exists())
        self.assertTrue((output/'DP0003/artifacts/run.margam-run.pt').exists())
        from margam.ui.run_document import load_run
        saved=load_run(output/'DP0001/artifacts/run.margam-run.pt');self.assertEqual(saved['result']['steps'],2)
        self.assertEqual(saved['result']['outputs'],{});self.assertEqual(report['trials'][0]['metrics'],{})
        before=(output/'summary.json').read_bytes();result=self.cli('study','run',path,'--output',output)
        self.assertEqual(result.returncode,2);self.assertEqual((output/'summary.json').read_bytes(),before)
    def test_owned_reference_exports_native_parity_archive_reopen_and_resume(self):
        import torch
        from margam.ui.run_document import load_run
        self.w.load_burgers_reference_example()
        for key,value in [('points',8),('boundary_points',4),('boundary_sampling',{}),('layers',(4,)),('iterations',2)]:self.w.change(key,value)
        path=self.study('[2]');self.assertTrue(self.w.start_training());self.wait();original=self.w.last_run.model.training_state()
        reference=deepcopy(self.w.result['outputs']);output=self.root/'reference'
        result=self.cli('study','run',path,'--output',output,'--export');self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads((output/'summary.json').read_text());self.assertEqual(report['counts'],{'completed':1});self.assertTrue(report['headless'])
        archive=output/'DP0001/artifacts/run.margam-run.pt';saved=load_run(archive)
        self.assertEqual(saved['checkpoint']['loss_history'],original['loss_history'])
        for key,value in saved['checkpoint']['network'].items():self.assertTrue(torch.equal(value,original['network'][key]))
        for key,data in saved['result']['outputs'].items():
            for field in ('points','values','reference','point_ids'):
                if field in data:self.assertEqual(data[field],reference[key][field])
        detail=json.loads((output/'DP0001/trial.json').read_text());self.assertEqual(len(detail['exports']),2)
        csv=next(item for item in detail['exports'] if item['kind']=='export');pvd=next(item for item in detail['exports'] if item['kind']=='paraview')
        import csv as csv_module
        import xml.etree.ElementTree as ET
        with (archive.parent/csv['path']).open() as stream:rows=list(csv_module.DictReader(stream))
        self.assertEqual(len(rows),2827);self.assertIn('ref_u',rows[0]);self.assertIn('point_id',rows[0])
        frames=ET.parse(archive.parent/pvd['path']).getroot().findall('.//DataSet');self.assertEqual(len(frames),11)
        self.assertEqual([float(item.attrib['timestep']) for item in frames],sorted({float(row['t']) for row in rows}))
        self.w.undo_stack.setClean();self.assertTrue(self.w.open_saved_run(archive));self.assertTrue(self.w.resume_training(1));self.wait();self.assertEqual(self.w.result['steps'],3)
        output2=self.root/'no-exports';result=self.cli('study','run',path,'--output',output2);self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual({p.name for p in (output2/'DP0001/artifacts').iterdir()},{'run.margam-run.pt','publication.json'})
    @unittest.skipUnless(__import__('torch').cuda.is_available(),'CUDA required')
    def test_cuda_cli_matches_native_worker_and_records_device_override(self):
        import torch
        from margam.ui.run_document import load_run
        self.assertTrue(self.w.set_application_device('cuda'));path=self.study('[2]')
        self.assertTrue(self.w.start_training());self.wait();native=self.w.last_run.model.training_state()
        output=self.root/'cuda';result=self.cli('study','run',path,'--output',output,'--device','cuda')
        self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads((output/'summary.json').read_text());self.assertTrue(report['headless'])
        self.assertEqual(report['saved_execution']['device'],'cpu');self.assertEqual(report['effective_execution']['device'],'cuda')
        data=load_run(output/'DP0001/artifacts/run.margam-run.pt')
        self.assertEqual(data['checkpoint']['loss_history'],native['loss_history'])
        for key,value in data['checkpoint']['network'].items():self.assertTrue(torch.equal(value,native['network'][key]))

    def test_unavailable_gpu_and_time_limit_have_no_fake_metrics(self):
        path=self.study('[1]');output=self.root/'gpu'
        result=self.cli('study','run',path,'--output',output,'--device','cuda',env={'CUDA_VISIBLE_DEVICES':''})
        self.assertEqual(result.returncode,1,result.stderr);report=json.loads((output/'summary.json').read_text());self.assertEqual(report['counts'],{'incompatible':1})
        self.assertIsNone(report['trials'][0]['metrics']);self.assertFalse((output/'DP0001/artifacts').exists())
        output=self.root/'limit';result=self.cli('study','run',path,'--output',output,'--wall-time','.001')
        self.assertEqual(result.returncode,1,result.stderr);report=json.loads((output/'summary.json').read_text());self.assertEqual(report['counts'],{'failed':1})
        detail=json.loads((output/'DP0001/trial.json').read_text());self.assertEqual(detail['failure_kind'],'job_time_limit');self.assertIsNone(detail['metrics'])
    def test_injected_numerical_failure_and_source_change_are_isolated(self):
        from margam.execution.transport import JobFailed
        path=self.study('[1,2]');output=self.root/'failure'
        from margam.execution.transport import JobTransport
        actual_run=JobTransport.run;attempts=0
        def fail_once(transport,*args,**kwargs):
            nonlocal attempts
            attempts+=1
            if attempts==1:raise JobFailed('injected numerical failure')
            return actual_run(transport,*args,**kwargs)
        with patch('margam.execution.study_runner.JobTransport.run',new=fail_once):report=run_study(path,output)
        self.assertEqual(report['counts'],{'failed':1,'completed':1});self.assertIsNone(report['trials'][0]['metrics'])
        self.assertTrue((output/'DP0002/artifacts/run.margam-run.pt').exists())
        output=self.root/'source'
        with patch('margam.execution.study_runner.source_matches',return_value=False):report=run_study(path,output)
        self.assertEqual(report['counts'],{'failed':1,'not_started':1});self.assertEqual(report['state'],'aborted')
        output=self.root/'source-after-job'
        with patch('margam.execution.study_runner.source_matches',side_effect=[True,True,False,False]):report=run_study(path,output)
        self.assertEqual(report['counts'],{'failed':1,'not_started':1});self.assertFalse((output/'DP0001/artifacts').exists())
        self.assertIsNone(report['trials'][0]['metrics'])

    def test_artifact_failure_discards_staging_and_next_trial_completes(self):
        self.w.load_burgers_reference_example()
        for key,value in [('points',8),('boundary_points',4),('boundary_sampling',{}),('layers',(4,))]:self.w.change(key,value)
        path=self.study('[1,2]');output=self.root/'artifact-failure'
        from margam.components.output_export import export_data
        attempts=0
        def fail_after_write(*args,**kwargs):
            nonlocal attempts
            attempts+=1;result=export_data(*args,**kwargs)
            if attempts==1:raise OSError('injected export publication failure')
            return result
        with patch('margam.components.output_export.export_data',new=fail_after_write):report=run_study(path,output,exports=True)
        self.assertEqual(report['counts'],{'failed':1,'completed':1});self.assertIsNone(report['trials'][0]['metrics'])
        self.assertFalse((output/'DP0001/artifacts').exists());self.assertFalse((output/'DP0001/.artifacts').exists())
        self.assertTrue((output/'DP0002/artifacts/run.margam-run.pt').is_file())

    @unittest.skipUnless(os.name=='posix','POSIX signal cleanup check')
    def test_real_interrupt_reaps_child_and_preserves_completed_trial(self):
        path=self.study('[1,100000,2]');output=self.root/'interrupt'
        process=subprocess.Popen([sys.executable,'-m','margam.cli','study','run',str(path),'--output',str(output)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        child=None
        try:
            end=time.monotonic()+45
            while time.monotonic()<end:
                events=output/'DP0002/events.jsonl'
                if events.exists():
                    lines=events.read_text().splitlines()
                    ready=[json.loads(line) for line in lines if line.strip() and json.loads(line)['kind']=='ready']
                    if ready:child=ready[0]['pid'];break
                if process.poll() is not None:break
                time.sleep(.02)
            self.assertIsNotNone(child)
            process.send_signal(signal.SIGINT);stdout,stderr=process.communicate(timeout=20)
            self.assertEqual(process.returncode,130,stderr)
            report=json.loads((output/'summary.json').read_text());self.assertEqual(report['counts'],{'completed':1,'interrupted':1,'not_started':1})
            self.assertTrue((output/'DP0001/artifacts/run.margam-run.pt').exists());self.assertFalse((output/'DP0002/artifacts').exists())
            with self.assertRaises(ProcessLookupError):os.kill(child,0)
        finally:
            if process.poll() is None:process.kill();process.communicate()
    def test_inspect_and_import_are_display_free_and_do_not_import_torch(self):
        path=self.study('[1]');result=self.cli('study','inspect',path);self.assertEqual(result.returncode,0,result.stderr)
        code="import sys; from margam.cli import main; assert main(['study','inspect',sys.argv[1]])==0; assert 'torch' not in sys.modules and 'PySide6' not in sys.modules"
        result=subprocess.run([sys.executable,'-c',code,str(path)],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
