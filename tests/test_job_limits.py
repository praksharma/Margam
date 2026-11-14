"""Wall-time discard, timer cleanup and local desktop preferences, without OOMs."""
from copy import deepcopy
from pathlib import Path
import os,subprocess,sys,time,unittest
from threading import Timer as RealTimer
from unittest.mock import patch
import torch
from margam.execution.transport import JobTransport,JobFailed,MAX_WALL_TIME_SECONDS
from margam.execution.codec import encode_config
from margam.templates.poisson import PoissonConfig


def job(device='cpu',iterations=3):
    return dict(operation='train',config=encode_config(PoissonConfig(shape='Interval',device=device,points=6,layers=(4,),iterations=iterations)),requests=[],checkpoint=None,metadata={},dtype='float32')


class ManualTimer:
    instances=[]
    def __init__(self,seconds,function):self.function=function;self.cancelled=False;self.joined=False;self.instances.append(self)
    def start(self):pass
    def cancel(self):self.cancelled=True
    def join(self,timeout):self.joined=True


class JobLimitTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);ManualTimer.instances=[]

    def assert_reaped(self,transport):
        self.assertIsNone(transport.process)
        if os.name=='posix':
            with self.assertRaises(ProcessLookupError):os.kill(transport.pid,0)

    def test_actual_deadline_bounds_child_startup_and_does_not_adopt_state(self):
        transport=JobTransport(wall_time_seconds=.1);before=torch.get_rng_state().clone();started=time.monotonic()
        with self.assertRaisesRegex(JobFailed,'time limit'):transport.run(job(iterations=100000))
        self.assertTrue(transport.deadline_exceeded);self.assertFalse(transport.terminated)
        self.assertLess(time.monotonic()-started,15);self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assert_reaped(transport)

    def expire_after_progress(self,device):
        transport=JobTransport(wall_time_seconds=60);events=[]
        def event(value):
            events.append(value)
            if value['kind']=='progress' and not transport.deadline_exceeded:ManualTimer.instances[-1].function()
        with patch('margam.execution.transport.Timer',ManualTimer):
            with self.assertRaisesRegex(JobFailed,'discarded'):transport.run(job(device,100000),event)
        self.assertTrue(any(e['kind']=='progress' for e in events))
        self.assertTrue(transport.deadline_exceeded);self.assert_reaped(transport)
        self.assertTrue(ManualTimer.instances[-1].cancelled and ManualTimer.instances[-1].joined)

    def test_running_expiry_discards_real_updates_and_cleans_timer(self):self.expire_after_progress('cpu')

    def test_unlimited_and_finite_success_preserve_updates_and_cancel_watchdog(self):
        with patch('margam.execution.transport.Timer',side_effect=AssertionError('Unlimited must create no timer')):
            unlimited=JobTransport().run(job())
        timers=[]
        def timer(seconds,callback):
            item=RealTimer(seconds,callback);timers.append(item);return item
        with patch('margam.execution.transport.Timer',side_effect=timer):
            transport=JobTransport(wall_time_seconds=30);finite=transport.run(job())
        self.assertFalse(timers[0].is_alive());self.assertFalse(transport.deadline_exceeded)
        self.assertEqual(unlimited['checkpoint']['loss_history'],finite['checkpoint']['loss_history'])
        for key,value in unlimited['checkpoint']['network'].items():self.assertTrue(torch.equal(value,finite['checkpoint']['network'][key]))
        self.assert_reaped(transport)

    def test_full_cancel_pipe_cannot_block_the_deadline(self):
        from threading import Thread
        real_popen=subprocess.Popen;transport=JobTransport(wall_time_seconds=.3);errors=[]
        def launch(command,**kwargs):return real_popen([sys.executable,'-c','import time; time.sleep(4)'],**kwargs)
        def run():
            try:transport.run(job())
            except Exception as error:errors.append(error)
        with patch('margam.execution.transport.subprocess.Popen',side_effect=launch):
            thread=Thread(target=run);thread.start();deadline=time.monotonic()+5
            while transport.pid is None and time.monotonic()<deadline:time.sleep(.002)
            self.assertIsNotNone(transport.pid)
            started=time.monotonic()
            # This fixture never reads stdin. Repeated Stop commands fill its pipe.
            for _ in range(5000):transport.cancel()
            thread.join(5)
        self.assertFalse(thread.is_alive());self.assertLess(time.monotonic()-started,3)
        self.assertEqual(len(errors),1);self.assertIsInstance(errors[0],JobFailed)
        self.assertTrue(transport.deadline_exceeded);self.assert_reaped(transport)

    def test_finished_child_is_not_expired_during_parent_result_loading(self):
        from margam.execution.codec import load_packet
        transport=JobTransport(wall_time_seconds=60)
        def load(path):
            ManualTimer.instances[-1].function() # Child is already reaped; parent decoding is outside budget.
            return load_packet(path)
        with patch('margam.execution.transport.Timer',ManualTimer),patch('margam.execution.transport.load_packet',side_effect=load):
            result=transport.run(job())
        self.assertEqual(result['result']['steps'],3);self.assertFalse(transport.deadline_exceeded)
        self.assert_reaped(transport)

    def test_invalid_limits_and_watchdog_launch_failure_reap_child(self):
        for value in (True,None,-1,float('nan'),float('inf'),MAX_WALL_TIME_SECONDS+1,10**1000):
            with self.subTest(value=value),self.assertRaises(ValueError):JobTransport(wall_time_seconds=value)
        class BrokenTimer(ManualTimer):
            def start(self):raise RuntimeError('Cannot start timer fixture')
            def join(self,timeout):raise AssertionError('An unstarted timer cannot be joined')
        transport=JobTransport(wall_time_seconds=1)
        with patch('margam.execution.transport.Timer',BrokenTimer):
            with self.assertRaisesRegex(RuntimeError,'Cannot start timer'):transport.run(job())
        self.assert_reaped(transport)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_finite_budget_and_running_expiry(self):
        timed=JobTransport(wall_time_seconds=60).run(job('cuda'))
        untimed=JobTransport().run(job('cuda'))
        self.assertEqual(timed['checkpoint']['loss_history'],untimed['checkpoint']['loss_history'])
        for key,value in timed['checkpoint']['network'].items():self.assertTrue(torch.equal(value,untimed['checkpoint']['network'][key]))
        self.expire_after_progress('cuda')

    def test_desktop_preferences_busy_guards_failure_recovery_and_evaluation(self):
        code=r'''
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib,time
from PySide6.QtWidgets import QApplication,QDialog
from margam.ui.app import Window
from margam.ui.model_document import capture_model
from margam.ui.run_summary import execution_plan
from margam.ui.job_limits import JobLimitDialog
app=QApplication([])
with TemporaryDirectory() as directory:
 w=Window(state_directory=directory)
 try:
  before=capture_model(w);index=w.undo_stack.index()
  assert w.job_time_limit_seconds==0 and not w.job_time_limit_action.icon().isNull()
  assert w.set_job_time_limit(120)
  assert capture_model(w)==before and w.undo_stack.index()==index
  assert '120 seconds' in execution_plan(w)[2]
  dialog=JobLimitDialog(120,w);assert dialog.seconds.value()==120;dialog.seconds.setValue(0);assert dialog.seconds.text()=='No limit';dialog.close()
  with patch('margam.ui.job_limits.JobLimitDialog') as factory:
   factory.return_value.exec.return_value=QDialog.DialogCode.Rejected
   factory.return_value.seconds.value.return_value=60
   assert not w.request_job_time_limit() and w.job_time_limit_seconds==120
  w.set_busy(True);assert not w.set_job_time_limit(60) and not w.job_time_limit_action.isEnabled();w.set_busy(False)
  for value in (True,1.5,-1,604801):assert not w.set_job_time_limit(value)
 finally:w.undo_stack.setClean();w.close();app.processEvents()
 w=Window(state_directory=directory)
 try:assert w.job_time_limit_seconds==120
 finally:w.undo_stack.setClean();w.close();app.processEvents()
 w=Window(state_directory=False);w.show();app.processEvents()
 def finish():
  deadline=time.monotonic()+30
  while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.003)
  assert w.worker is None,w.log.toPlainText()
 try:
  assert w.job_time_limit_seconds==0;w.set_application_device('cpu')
  w.change('shape','Interval');w.change('points',6);w.change('layers',(4,));w.change('iterations',3)
  assert w.start_training();finish();assert w.result['steps']==3
  archive=Path(directory)/'saved-run.pt';assert w.save_current_run(archive);saved=hashlib.sha256(archive.read_bytes()).hexdigest()
  assert w.set_job_time_limit(1) and w.resume_training(50000)
  assert w.worker.transport.wall_time_seconds==1;finish()
  assert w.result is None and w.train_button.isEnabled() and 'time limit' in w.log.toPlainText()
  assert hashlib.sha256(archive.read_bytes()).hexdigest()==saved
  assert w.set_job_time_limit(60);w.undo_stack.setClean();assert w.open_saved_run(archive)
  assert w.job_time_limit_seconds==60 and w.resume_training(2)
  assert w.worker.transport.wall_time_seconds==60;finish();assert w.result['steps']==5
  node=w.add_node('Solution Plot');root=next(n for n in w.scene.nodes() if n.kind=='method')
  w.scene.connect_nodes(root,node);w.change_output(node,'resolution',5)
  assert w.evaluate_trained_outputs() and w.worker.transport.wall_time_seconds==60;finish()
  assert w.result['steps']==5 and node.identifier in w.result['outputs']
  result=w.result;before=capture_model(w);index=w.undo_stack.index()
  assert w.set_job_time_limit(0) and w.result is result and capture_model(w)==before and w.undo_stack.index()==index
 finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=100,
            env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)


if __name__=='__main__':unittest.main()
