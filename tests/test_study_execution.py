"""Real GUI-controlled CLI lifecycle, preserved canvas and process-group cleanup."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
import test_study_cli as fixture
from margam.ui.model_document import capture_model
from margam.execution.study_state import file_identity
from margam.execution.study_runner import resume_study


@unittest.skipUnless(os.name=='posix','POSIX local study process groups required')
class StudyExecutionTests(unittest.TestCase):
    setUpClass=classmethod(fixture.StudyCLITests.setUpClass.__func__)
    def setUp(self):
        fixture.StudyCLITests.setUp(self);self.errors=[];self.hook=sys.excepthook;sys.excepthook=lambda *args:self.errors.append(args)
    def tearDown(self):
        dialog=getattr(self.w,'study_execution_dialog',None)
        if dialog is not None:
            if dialog.active:dialog.stop(force=True);self.wait_until(lambda:not dialog.active)
            dialog.shutdown_review()
        try:fixture.StudyCLITests.tearDown(self);self.assertEqual(self.errors,[])
        finally:sys.excepthook=self.hook
    wait=fixture.StudyCLITests.wait
    study=fixture.StudyCLITests.study
    def wait_until(self,predicate,timeout=60):
        deadline=time.monotonic()+timeout
        while not predicate() and time.monotonic()<deadline:self.app.processEvents();time.sleep(.005)
        self.app.processEvents();self.assertTrue(predicate())
    def setup_job(self,values='[1,2]',device=None):
        path=self.study(values);dialog=self.w.show_study_execution();dialog.source.setText(str(path));dialog.output.setText(str(self.root/'results'))
        if device is not None:dialog.device.setCurrentIndex(dialog.device.findData(device))
        self.assertTrue(dialog.review());self.wait_until(lambda:dialog.reader is None);self.assertIsNotNone(dialog.prepared,dialog.status.text())
        return dialog,path
    def test_review_freezes_input_and_serial_cpu_execution_keeps_canvas_results_rng(self):
        import torch
        # Preserve a real existing canvas result/checkpoint while the study runs.
        self.assertTrue(self.w.start_training());self.wait();before=capture_model(self.w);result=self.w.result;last=self.w.last_run;undo=self.w.undo_stack.index();rng=torch.get_rng_state().clone()
        dialog,path=self.setup_job();path.write_bytes(b'changed after review')
        self.assertTrue(dialog.start());self.assertTrue(self.w.is_running());self.assertTrue(self.w.scene.locked)
        self.assertFalse(self.w.start_training());self.assertFalse(self.w.set_application_device('cuda'));self.assertTrue(self.w.force_stop_action.isEnabled())
        self.wait_until(lambda:not dialog.active);report=json.loads((self.root/'results/summary.json').read_text())
        self.assertEqual(report['counts'],{'completed':2});self.assertTrue(report['headless']);self.assertEqual(report['effective_execution']['device'],'cpu')
        self.assertEqual(before,capture_model(self.w));self.assertEqual(undo,self.w.undo_stack.index());self.assertIs(result,self.w.result);self.assertIs(last,self.w.last_run)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertFalse(self.w.scene.locked);self.assertFalse(dialog.start())
        self.assertTrue(dialog.open_results());browser=self.w.study_results_dialog;self.wait_until(lambda:browser.worker is None);self.assertEqual(browser.table.rowCount(),2);browser.close()
    @unittest.skipUnless(__import__('torch').cuda.is_available(),'CUDA required')
    def test_cuda_override_and_original_device_preference(self):
        dialog,_=self.setup_job('[1]',device='cuda');original=self.w.application_device
        self.assertTrue(dialog.start());self.wait_until(lambda:not dialog.active)
        report=json.loads((self.root/'results/summary.json').read_text());self.assertEqual(report['effective_execution']['device'],'cuda');self.assertTrue(report['headless'])
        self.assertEqual(self.w.application_device,original)
    def test_real_stop_resume_and_completed_attempt_files_preserved(self):
        dialog,_=self.setup_job('[1,300,2]');self.assertTrue(dialog.start())
        events=self.root/'results/DP0002/events.jsonl';child=None
        def ready():
            nonlocal child
            if events.exists():
                for line in events.read_text().splitlines():
                    try:value=json.loads(line)
                    except ValueError:continue
                    if value['kind']=='ready':child=value['pid'];return True
            return False
        self.wait_until(ready);os.kill(child,signal.SIGSTOP);self.w.cancel_training();self.w.cancel_training();self.wait_until(lambda:not dialog.active)
        report=json.loads((self.root/'results/summary.json').read_text());self.assertEqual(report['counts'],{'completed':1,'interrupted':1,'not_started':1})
        archive=self.root/'results/DP0001/artifacts/run.margam-run.pt';before=archive.read_bytes()
        with self.assertRaises(ProcessLookupError):os.kill(child,0)
        dialog.mode.setCurrentIndex(dialog.mode.findData('resume'));self.assertTrue(dialog.review());self.wait_until(lambda:dialog.reader is None)
        self.assertFalse(dialog.device.isEnabled());self.assertTrue(dialog.start());self.wait_until(lambda:not dialog.active)
        report=json.loads((self.root/'results/summary.json').read_text());self.assertEqual(report['counts'],{'completed':3});self.assertEqual(report['trials'][1]['attempt'],2);self.assertEqual(before,archive.read_bytes())
    def test_force_stop_owned_group_and_dialog_hide(self):
        dialog,_=self.setup_job('[100000]');self.assertTrue(dialog.start());self.wait_until(lambda:dialog.worker.process is not None)
        process=dialog.worker.process;temporary=dialog.worker.temp_root;self.assertEqual(os.getpgid(process.pid),process.pid)
        child=None;events=self.root/'results/DP0001/events.jsonl'
        def ready():
            nonlocal child
            if events.exists():
                for line in events.read_text().splitlines():
                    try:value=json.loads(line)
                    except ValueError:continue
                    if value['kind']=='ready':child=value['pid'];return True
            return False
        self.wait_until(ready);self.assertEqual(os.getpgid(child),process.pid)
        dialog.close();self.assertTrue(dialog.active);self.assertIs(dialog,self.w.show_study_execution())
        self.w.force_stop_training();self.wait_until(lambda:not dialog.active);self.assertIsNotNone(process.poll());self.assertFalse(self.w.is_running());self.assertFalse(Path(temporary).exists())
        from margam.execution.study_state import lease
        def unlocked():
            try:
                with lease(self.root/'results'):return True
            except RuntimeError:return False
        self.wait_until(unlocked)
    def test_failure_before_launch_and_cancelled_review_do_not_lock_canvas(self):
        dialog=self.w.show_study_execution();dialog.source.setText(str(self.root/'missing'));dialog.output.setText(str(self.root/'results'))
        self.assertFalse(dialog.start());self.assertTrue(dialog.review());self.wait_until(lambda:dialog.reader is None)
        self.assertIsNone(dialog.prepared);self.assertFalse(self.w.is_running());self.assertFalse((self.root/'results').exists())
        path=self.study('[1]');dialog.source.setText(str(path));self.assertTrue(dialog.review());dialog.close();self.assertIsNone(dialog.reader);self.assertIsNone(dialog.prepared)
        dialog=self.w.show_study_execution();self.assertFalse(dialog.start());self.assertFalse(self.w.scene.locked)
    def test_invalid_trial_is_reported_and_does_not_stop_valid_trials(self):
        path=self.study('[[4],[-1],[8]]','layers');dialog=self.w.show_study_execution()
        dialog.source.setText(str(path));dialog.output.setText(str(self.root/'mixed'))
        self.assertTrue(dialog.review());self.wait_until(lambda:dialog.reader is None);self.assertTrue(dialog.start());self.wait_until(lambda:not dialog.active)
        report=json.loads((self.root/'mixed/summary.json').read_text());self.assertEqual(report['counts'],{'completed':2,'invalid':1})
        self.assertIn('trial issues',dialog.status.text());self.assertIsNone(dialog.prepared);self.assertFalse(self.w.scene.locked)

    def test_reviewed_execution_identity_rejects_changed_resume(self):
        dialog,_=self.setup_job('[1]');dialog.start();self.wait_until(lambda:not dialog.active)
        output=self.root/'results';before=(output/'summary.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'changed since review'):resume_study(output,expected_execution_sha256='0'*64)
        self.assertEqual(before,(output/'summary.json').read_bytes())
        dialog.mode.setCurrentIndex(dialog.mode.findData('resume'));dialog.review();self.wait_until(lambda:dialog.reader is None)
        self.assertEqual(dialog.prepared['execution_sha256'],file_identity(output,'execution.json')['sha256'])
    def test_gui_parent_death_interrupts_cli_and_reaps_numerical_child(self):
        path=self.study('[100000]');output=self.root/'owner-loss'
        code="from PySide6.QtCore import QCoreApplication; from margam.ui.study_execution import StudyProcess; import sys; app=QCoreApplication([]); worker=StudyProcess(['study','run',sys.argv[1],'--output',sys.argv[2]],sys.argv[2]); worker.finished.connect(app.quit); worker.start(); app.exec()"
        owned_temp=self.root/'owner-temp';owned_temp.mkdir()
        parent=subprocess.Popen([sys.executable,'-c',code,str(path),str(output)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=dict(os.environ,TMPDIR=str(owned_temp)))
        child=None
        try:
            def ready():
                nonlocal child
                events=output/'DP0001/events.jsonl'
                if events.exists():
                    for line in events.read_text().splitlines():
                        try:value=json.loads(line)
                        except ValueError:continue
                        if value['kind']=='ready':child=value['pid'];return True
                return False
            self.wait_until(ready);parent.kill();parent.wait(timeout=10)
            def stopped():
                try:return json.loads((output/'summary.json').read_text())['state']=='interrupted'
                except (OSError,ValueError):return False
            self.wait_until(stopped,timeout=20)
            with self.assertRaises(ProcessLookupError):os.kill(child,0)
            from margam.execution.study_state import lease
            with lease(output):pass
        finally:
            if parent.poll() is None:parent.kill();parent.wait()

    def test_stop_is_idempotent_and_pending_stop_is_delivered_after_launch(self):
        from margam.ui.study_execution import StudyProcess
        from unittest.mock import Mock
        worker=StudyProcess([],self.root)
        worker.request_stop();self.assertTrue(worker.stop_requested);self.assertFalse(worker.interrupt_sent)
        process=Mock();process.poll.return_value=None;process.pid=123456;worker.process=process
        worker.request_stop();worker.request_stop();worker.request_stop()
        process.send_signal.assert_called_once_with(signal.SIGINT)
        with patch('margam.ui.study_execution.os.killpg') as kill:
            worker.request_stop(force=True);kill.assert_called_once_with(process.pid,signal.SIGKILL)
        worker.deleteLater()

    def test_window_close_stops_job_before_discard_prompt_and_closes_safely(self):
        dialog,_=self.setup_job('[100000]');self.assertTrue(dialog.start());self.wait_until(lambda:dialog.worker.process is not None)
        with patch.object(self.w,'confirm_discard',return_value=False):
            self.w.close();self.assertTrue(self.w._closing);self.wait_until(lambda:not dialog.active);self.app.processEvents()
        self.assertFalse(self.w._closing);self.assertFalse(self.w.is_running());self.assertFalse(self.w.scene.locked)


if __name__=='__main__':unittest.main()
