"""Headless execution, bounded data-only IPC, cancellation and process-death recovery."""
from dataclasses import replace
from pathlib import Path
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import time
import unittest
from unittest.mock import patch
import torch
from margam.execution.codec import encode_config,encode_request,save_packet,load_packet
from margam.execution.transport import JobTransport,JobFailed
from margam.components.outputs import OutputRequest,ReferenceSource
from margam.templates.poisson import PoissonConfig,build_pinn,make_interval_reference


def job(iterations=3,requests=()):
    c=PoissonConfig(shape='Interval',device='cpu',points=6,layers=(4,),iterations=iterations)
    return dict(operation='train',config=encode_config(c),requests=[encode_request(r) for r in requests],checkpoint=None,metadata={},dtype='float32')


class IsolatedJobTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_actual_child_is_headless_matches_engine_and_preserves_parent_rng(self):
        c=PoissonConfig(shape='Interval',device='cpu',points=6,layers=(4,),iterations=3)
        reference=replace(make_interval_reference(c),spec=dict(kind='preset'))
        request=OutputRequest('compared','comparison',5,reference)
        direct=build_pinn(c);direct.train(3,0);before=torch.get_rng_state().clone()
        transport=JobTransport();events=[];packet=transport.run(job(requests=[request]),events.append)
        self.assertNotEqual(packet['pid'],os.getpid());self.assertTrue(packet['headless']);self.assertIsNone(transport.process)
        self.assertEqual(packet['checkpoint']['loss_history'],direct.loss_history)
        self.assertEqual([event['kind'] for event in events],['ready','progress'])
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        for key,value in direct.network.state_dict().items():
            self.assertTrue(torch.equal(value,packet['checkpoint']['network'][key]));self.assertEqual(packet['checkpoint']['network'][key].device.type,'cpu')
        from margam.components.outputs import evaluate_outputs
        expected=evaluate_outputs(direct,c,[request])['compared']
        self.assertEqual(packet['result']['outputs']['compared']['values'],expected['values'])
        self.assertEqual(packet['result']['outputs']['compared']['reference'],expected['reference'])
        if os.name=='posix':
            with self.assertRaises(ProcessLookupError):os.kill(packet['pid'],0)

    def test_cancel_before_launch_and_forced_stop_discard_running_job(self):
        transport=JobTransport();transport.cancel();packet=transport.run(job(100000))
        self.assertTrue(packet['result']['cancelled']);self.assertLess(packet['result']['steps'],100000)
        transport=JobTransport();errors=[]
        def run():
            try:transport.run(job(100000))
            except Exception as error:errors.append(error)
        thread=Thread(target=run);thread.start();deadline=time.monotonic()+15
        while transport.pid is None and time.monotonic()<deadline:time.sleep(.01)
        self.assertIsNotNone(transport.pid);pid=transport.pid;transport.terminate();thread.join(15)
        self.assertFalse(thread.is_alive());self.assertEqual(len(errors),1);self.assertIsInstance(errors[0],JobFailed)
        self.assertIn('force-stopped',str(errors[0]));self.assertIsNone(transport.process)
        if os.name=='posix':
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)

    def test_callbacks_and_oversized_packets_are_rejected_without_overwriting_snapshot(self):
        with self.assertRaisesRegex(ValueError,'data-only'):encode_request(OutputRequest('x','comparison',3,ReferenceSource('custom',lambda p:p)))
        with TemporaryDirectory() as directory:
            path=Path(directory)/'data.pt';save_packet(path,dict(value=torch.ones(2)));original=path.read_bytes()
            self.assertTrue(torch.equal(load_packet(path)['value'],torch.ones(2)))
            with patch('margam.execution.codec.LIMIT',1000):
                with self.assertRaises((ValueError,RuntimeError)):save_packet(path,dict(value='x'*10000))
            self.assertEqual(path.read_bytes(),original);self.assertFalse(path.with_suffix('.partial').exists())
            with self.assertRaises(ValueError):save_packet(Path(directory)/'callable.pt',dict(value=lambda:None))

    def test_dead_or_malformed_child_does_not_return_a_result(self):
        real_popen=subprocess.Popen
        for source in ("import os;os._exit(13)","print('not a numerical event',flush=True)"):
            with self.subTest(source=source):
                transport=JobTransport()
                def launch(command,**kwargs):return real_popen([sys.executable,'-c',source],**kwargs)
                with patch('margam.execution.transport.subprocess.Popen',side_effect=launch):
                    with self.assertRaises((JobFailed,ValueError)):transport.run(job())
                self.assertIsNone(transport.process)

    def test_parent_pipe_closure_stops_child_and_late_force_discards_result(self):
        payload=job(100000);payload.update(format='margam-job',version=1,id='parent-exit-fixture')
        with TemporaryDirectory() as directory:
            save_packet(Path(directory)/'input.pt',payload)
            process=subprocess.Popen([sys.executable,'-m','margam.execution.worker',directory],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            process.stdin.close()
            try:
                self.assertEqual(process.wait(timeout=15),0)
                packet=load_packet(Path(directory)/'output.pt');self.assertTrue(packet['result']['cancelled'])
            finally:
                if process.poll() is None:process.kill();process.wait()
                process.stdout.close();process.stderr.close()
        transport=JobTransport();original=load_packet
        def force_during_transfer(path):
            data=original(path);transport.terminate();return data
        with patch('margam.execution.transport.load_packet',side_effect=force_during_transfer):
            with self.assertRaisesRegex(JobFailed,'result transfer'):transport.run(job())
        self.assertIsNone(transport.process)

    def test_desktop_force_stop_abnormal_exit_and_next_run(self):
        code=r"""
import os,time
from pathlib import Path
from tempfile import TemporaryDirectory
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
app=QApplication([]);w=Window(state_directory=False)
def finish():
 deadline=time.monotonic()+30
 while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.002)
 assert w.worker is None,w.log.toPlainText()
try:
 w.set_application_device('cpu');w.change('layers',(4,));w.change('iterations',100000)
 assert w.start_training();assert w.force_stop_action.isEnabled()
 w.force_stop_action.trigger();finish()
 assert w.result is None and not w.force_stop_action.isEnabled() and w.train_button.isEnabled()
 assert 'force-stopped' in w.log.toPlainText()
 assert w.start_training();deadline=time.monotonic()+10
 while w.worker.child_pid is None and time.monotonic()<deadline:app.processEvents();time.sleep(.002)
 assert w.worker.child_pid is not None
 w.worker.transport.process.kill();finish()
 assert w.result is None and 'exited unexpectedly' in w.log.toPlainText()
 w.change('iterations',3);assert w.start_training();finish()
 assert w.result is not None and w.last_run.child_pid!=os.getpid() and w.last_run.headless
 assert w.last_run.model.storage_device=='cpu'
 with TemporaryDirectory() as directory:
  path=Path(directory)/'isolated.pt';assert w.save_current_run(path)
  w.undo_stack.setClean();assert w.open_saved_run(path);assert w.resume_training(2);finish()
  assert w.result['steps']==5
finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=90,
            env=dict(os.environ,QT_QPA_PLATFORM='offscreen',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr);self.assertNotIn('Traceback',result.stderr)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_child_keeps_retained_state_on_cpu_and_resumes_exactly(self):
        payload=job(3);payload['config']['device']='cuda'
        first=JobTransport().run(payload);self.assertTrue(all(v.device.type=='cpu' for v in first['checkpoint']['network'].values()))
        payload['checkpoint']=first['checkpoint'];payload['config']['iterations']=5
        resumed=JobTransport().run(payload)
        payload['checkpoint']=None;full=JobTransport().run(payload)
        self.assertEqual(resumed['checkpoint']['loss_history'],full['checkpoint']['loss_history'])
        for key,value in resumed['checkpoint']['network'].items():self.assertTrue(torch.equal(value,full['checkpoint']['network'][key]))
        self.assertEqual(resumed['result']['device'],'cuda')
