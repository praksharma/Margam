"""Reviewed disposable GUI probes use the CLI owner lifecycle and preserve state."""
import json
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import torch
try:
    from . import test_study_execution as fixture
except ImportError:
    import test_study_execution as fixture
from margam.ui.model_document import capture_model


@unittest.skipUnless(os.name=='posix','POSIX owned process groups required')
class ResourceProbeUITests(unittest.TestCase):
    setUpClass=classmethod(fixture.StudyExecutionTests.setUpClass.__func__)
    setUp=fixture.StudyExecutionTests.setUp
    wait=fixture.StudyExecutionTests.wait
    study=fixture.StudyExecutionTests.study
    wait_until=fixture.StudyExecutionTests.wait_until
    def tearDown(self):
        dialog=getattr(self.w,'resource_probe_dialog',None)
        if dialog is not None:
            if dialog.active:dialog.stop(force=True);self.wait_until(lambda:not dialog.active)
            dialog.shutdown_review()
        fixture.StudyExecutionTests.tearDown(self)
    def setup_probe(self,values='[2,3]',trial=1,updates=1):
        path=self.study(values);dialog=self.w.show_resource_probe();dialog.source.setText(str(path));dialog.trial.setValue(trial);dialog.updates.setValue(updates)
        self.assertTrue(dialog.review());self.wait_until(lambda:dialog.reader is None);self.assertIsNotNone(dialog.prepared,dialog.status.text());return dialog,path

    def test_review_freezes_file_and_probe_preserves_canvas_results_and_rng(self):
        self.assertTrue(self.w.start_training());self.wait();before=capture_model(self.w);result=self.w.result;last=self.w.last_run;undo=self.w.undo_stack.index();rng=torch.get_rng_state().clone()
        dialog,path=self.setup_probe(trial=2);frozen=dialog.prepared['path'];path.write_bytes(b'replaced after review')
        self.assertTrue(dialog.start());self.assertTrue(self.w.is_running());self.assertTrue(self.w.scene.locked);self.assertFalse(self.w.start_training());self.assertFalse(self.w.set_application_device('cuda'));self.assertTrue(self.w.force_stop_action.isEnabled());self.assertTrue(dialog.worker.args[-2:]==['--wall-time','60'])
        self.wait_until(lambda:not dialog.active);self.assertIsNotNone(dialog.result,dialog.text.toPlainText());self.assertTrue(dialog.result['trial_id'].startswith('DP0002-'));self.assertEqual(dialog.result['completed_updates'],1);self.assertEqual(dialog.result['configured_updates'],3);self.assertFalse(frozen.exists())
        self.assertEqual(before,capture_model(self.w));self.assertEqual(undo,self.w.undo_stack.index());self.assertIs(result,self.w.result);self.assertIs(last,self.w.last_run);self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertFalse(self.w.scene.locked);self.assertFalse(dialog.start());self.assertIn('Measured scope',dialog.text.toPlainText())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_application_cuda_preference_drives_probe_without_changing_saved_intent(self):
        self.assertTrue(self.w.set_application_device('cuda'));dialog,path=self.setup_probe();saved=path.read_bytes();cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
        self.assertTrue(dialog.start());self.wait_until(lambda:not dialog.active);self.assertIsNotNone(dialog.result,dialog.text.toPlainText());self.assertEqual(dialog.result['requested_device'],'cuda');self.assertIn('cuda',dialog.result['device']);self.assertGreater(dialog.result['cuda_peak_allocated_bytes'],0);self.assertEqual(path.read_bytes(),saved);self.assertEqual(self.w.application_device,'cuda');self.assertTrue(torch.equal(cpu,torch.get_rng_state()));self.assertTrue(torch.equal(gpu,torch.cuda.get_rng_state()))

    def test_review_errors_edits_device_change_and_failed_thread_start(self):
        dialog=self.w.show_resource_probe();self.assertFalse(dialog.review());dialog.source.setText(str(self.root/'missing'));self.assertTrue(dialog.review());self.wait_until(lambda:dialog.reader is None);self.assertIsNone(dialog.prepared)
        dialog,path=self.setup_probe();dialog.updates.setValue(2);self.assertIsNone(dialog.prepared);self.assertFalse(dialog.start())
        dialog.trial.setValue(9);self.assertTrue(dialog.review());self.wait_until(lambda:dialog.reader is None);self.assertIsNone(dialog.prepared)
        dialog.trial.setValue(1);dialog.updates.setValue(1);self.assertTrue(dialog.review());self.wait_until(lambda:dialog.reader is None)
        self.assertTrue(self.w.set_application_device('auto'));self.assertFalse(dialog.start());self.assertIsNone(dialog.prepared)
        self.w.set_application_device('cpu');self.assertTrue(dialog.review());self.wait_until(lambda:dialog.reader is None)
        with patch('margam.ui.resource_probe.StudyProcess.start',side_effect=RuntimeError('injected launch failure')):self.assertFalse(dialog.start())
        self.assertFalse(dialog.active);self.assertFalse(self.w.is_running());self.assertFalse(self.w.scene.locked);self.assertIsNone(dialog.worker);self.assertIsNone(dialog.prepared)
        with patch('margam.ui.resource_probe.Reader.start',side_effect=RuntimeError('injected review failure')):self.assertFalse(dialog.review())
        self.assertIsNone(dialog.reader)
        with patch('margam.ui.resource_probe.prepare_probe',side_effect=MemoryError()):
            self.assertTrue(dialog.review());self.wait_until(lambda:dialog.reader is None)
        self.assertIsNone(dialog.prepared);self.assertIn('MemoryError',dialog.status.text())

    def test_cancel_finished_review_and_main_shutdown_join_reader(self):
        from margam.ui.resource_probe import prepare_probe
        path=self.study('[2]');dialog=self.w.show_resource_probe();dialog.source.setText(str(path))
        self.assertTrue(dialog.review());dialog.reader.wait();temporary=dialog.reader.value['temporary'].name;dialog.close();self.app.processEvents();self.assertFalse(Path(temporary).exists());self.assertIsNone(dialog.prepared);self.assertIsNone(dialog.reader)
        self.assertIs(self.w.show_resource_probe(),dialog)
        def slow(*args):
            check=args[-1]
            while not check():time.sleep(.002)
            raise InterruptedError('cancelled')
        with patch('margam.ui.resource_probe.prepare_probe',side_effect=slow):
            self.assertTrue(dialog.review());self.w.undo_stack.setClean();self.w.close();self.assertIsNone(dialog.reader)

    def test_stop_force_hide_and_main_window_close_discard_measurements(self):
        dialog,_=self.setup_probe('[1000]',updates=1000);self.assertTrue(dialog.start());worker=dialog.worker
        self.wait_until(lambda:worker.process is not None);dialog.close();self.assertTrue(dialog.active);self.assertIs(self.w.show_resource_probe(),dialog)
        self.w.cancel_training();self.wait_until(lambda:not dialog.active);self.assertIsNone(dialog.result);self.assertIsNotNone(worker.process.poll())
        dialog,_=self.setup_probe('[1000]',updates=1000);self.assertTrue(dialog.start());self.wait_until(lambda:dialog.worker.process is not None);self.w.force_stop_training();self.wait_until(lambda:not dialog.active);self.assertIsNone(dialog.result);self.assertFalse(self.w.is_running())
        dialog,_=self.setup_probe('[1000]',updates=1000);self.assertTrue(dialog.start())
        with patch.object(self.w,'confirm_discard',return_value=False):
            self.w.close();self.wait_until(lambda:not dialog.active);self.app.processEvents()
        self.assertIsNone(dialog.result);self.assertIsNone(dialog.worker);self.assertFalse(self.w._closing)

    @unittest.skipUnless(Path('/proc/self/task').exists(),'Linux process ownership inspection required')
    def test_gui_owner_loss_stops_probe_cli_and_native_child(self):
        import subprocess,sys,signal
        path=self.study('[1000]');marker=self.root/'owned-cli.pid'
        code="from PySide6.QtCore import QCoreApplication,QTimer; from margam.ui.study_execution import StudyProcess; from pathlib import Path; import sys; app=QCoreApplication([]); worker=StudyProcess(['study','probe',sys.argv[1],'--updates','1000','--wall-time','0'],None); timer=QTimer(); timer.timeout.connect(lambda:Path(sys.argv[2]).write_text(str(worker.process.pid)) if worker.process is not None else None); timer.start(50); worker.finished.connect(app.quit); worker.start(); app.exec()"
        owned_temp=self.root/'probe-owner-temp';owned_temp.mkdir()
        parent=subprocess.Popen([sys.executable,'-c',code,str(path),str(marker)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=dict(os.environ,TMPDIR=str(owned_temp)))
        cli=None;child=None
        try:
            def ready():
                nonlocal cli,child
                try:
                    cli=int(marker.read_text());children=Path(f'/proc/{cli}/task/{cli}/children').read_text().split()
                    if children:child=int(children[0]);return True
                except (OSError,ValueError):pass
                return False
            self.wait_until(ready);self.assertEqual(os.getpgid(child),cli);parent.kill();parent.wait(timeout=10)
            def stopped():
                try:os.kill(child,0);return False
                except ProcessLookupError:return True
            self.wait_until(stopped,timeout=20)
        finally:
            if parent.poll() is None:parent.kill();parent.wait()
            if cli is not None:
                try:os.killpg(cli,signal.SIGKILL)
                except ProcessLookupError:pass

    def test_cli_failure_and_invalid_report_never_adopt_measurement(self):
        dialog,_=self.setup_probe()
        def fail(worker):worker.returncode=2;worker.tail=b'injected CLI failure'
        with patch('margam.ui.resource_probe.StudyProcess.run',fail):
            self.assertTrue(dialog.start());self.wait_until(lambda:not dialog.active)
        self.assertIsNone(dialog.result);self.assertFalse(self.w.scene.locked)
        dialog,_=self.setup_probe()
        def bad(worker):worker.returncode=0;worker.tail=b'[]'
        with patch('margam.ui.resource_probe.StudyProcess.run',bad):
            self.assertTrue(dialog.start());self.wait_until(lambda:not dialog.active)
        self.assertIsNone(dialog.result);self.assertIn('Invalid probe',dialog.text.toPlainText());self.assertFalse(self.w.scene.locked)


if __name__=='__main__':unittest.main()
