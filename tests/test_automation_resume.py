"""Explicit native-checkpoint continuation without file access or implicit starts."""
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import torch
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.automation_core import AutomationError
from margam.ui.model_document import capture_model


class AutomationResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        self.w=Window(state_directory=False);self.w.set_application_device('cpu')
        self.c=self.w.automation_server.bridge.commands
        for name,value in [('points',12),('boundary_points',4),('layers',(4,)),('iterations',2)]:self.w.change(name,value)
    def tearDown(self):
        if self.w.is_running():self.w.cancel_training();self.wait()
        self.w.undo_stack.setClean();self.w.close();self.app.clipboard().clear();self.app.processEvents()
    def wait(self):
        end=time.monotonic()+45
        while self.w.worker is not None and time.monotonic()<end:self.app.processEvents();time.sleep(.005)
        self.assertIsNone(self.w.worker)
    def read(self):return self.c.execute('read_model',{})
    def args(self,steps=1):
        data=self.read()
        return dict(session_id=data['session_id'],expected_revision=data['revision'],run_id=data['continuation']['run_id'],additional_steps=steps)
    def error(self,code,args):
        with self.assertRaises(AutomationError) as caught:self.c.execute('resume_training',args)
        self.assertEqual(caught.exception.code,code)
    def train(self):self.assertTrue(self.w.start_training());self.wait();self.assertIsNotNone(self.w.result)

    def test_no_checkpoint_invalid_counts_busy_stale_and_cancelled_requests(self):
        self.assertFalse(self.read()['continuation']['available'])
        before=capture_model(self.w);self.error('resume_unavailable',self.args());self.assertEqual(capture_model(self.w),before)
        self.train();args=self.args();before=capture_model(self.w);index=self.w.undo_stack.index();retained=self.w.last_run
        for count in (True,0,-1,1.5,'2',100000):self.error('invalid_arguments',dict(args,additional_steps=count))
        self.error('stale_session',dict(args,session_id='other'))
        self.error('stale_revision',dict(args,expected_revision='other'))
        self.error('stale_checkpoint',dict(args,run_id='other'))
        self.w.scene.locked=True
        self.error('busy',args);self.w.scene.locked=False
        with self.assertRaises(AutomationError) as caught:self.c.execute('resume_training',args,cancelled=lambda:True)
        self.assertEqual(caught.exception.code,'request_cancelled')
        self.assertEqual(capture_model(self.w),before);self.assertEqual(self.w.undo_stack.index(),index);self.assertIs(self.w.last_run,retained)
        with patch.object(self.w,'commit_inspector',side_effect=lambda:self.w.change('iterations',3)):
            self.error('stale_revision',args)
        self.assertIsNone(self.w.worker)

    def test_read_only_metadata_limits_and_permission_revocation(self):
        self.train();model=self.w.last_run.model
        with patch.object(model,'training_state',side_effect=AssertionError('Inspection must not clone tensor state')):
            first=self.read()['continuation'];self.assertEqual(first,self.read()['continuation']);self.assertEqual(first['steps'],2)
            with patch.object(model,'iter',100000):self.assertFalse(self.read()['continuation']['available'])
            with patch.object(model,'dsgnar_summary',create=True,return_value={'converged':True}):self.assertFalse(self.read()['continuation']['available'])
        bridge=self.w.automation_server.bridge;permissions=bridge.permissions;bridge.set_active(True)
        args=self.args();future=bridge.submit('resume_training',args)
        while not future.done():self.app.processEvents()
        self.assertEqual(future.result()['error']['code'],'permission_denied')
        permissions.set_group('Training',True);future=bridge.submit('resume_training',args);permissions.set_tool('resume_training',False)
        while not future.done():self.app.processEvents()
        self.assertEqual(future.result()['error']['code'],'permission_denied');self.assertIsNone(self.w.worker)

    def test_manual_replacement_requires_new_checkpoint_id_even_with_same_revision(self):
        self.train();args=self.args();self.train()
        self.assertEqual(self.c.identity()['revision'],args['expected_revision'])
        self.assertNotEqual(self.read()['continuation']['run_id'],args['run_id']);self.error('stale_checkpoint',args)

    def test_resumed_job_cancellation_replacement_and_launch_rejection(self):
        self.train();args=self.args(10000)
        with patch.object(self.w,'resume_training',return_value=False):self.error('training_rejected',args)
        self.assertIsNone(self.c.job)
        start=self.c.execute('resume_training',args)
        self.c.execute('cancel_training',dict(session_id=start['session_id'],job_id=start['job_id']))
        self.wait()
        status=self.c.execute('training_status',dict(session_id=start['session_id'],job_id=start['job_id']))
        self.assertEqual(status['state'],'Partial');self.assertGreaterEqual(status['step'],2)
        self.assertTrue(self.read()['continuation']['available'])
        with patch.object(self.w,'resume_training',return_value=False):self.error('training_rejected',self.args())
        self.assertEqual(self.c.job['job_id'],start['job_id'])

    def workflow(self,device,archive=False):
        self.assertTrue(self.w.set_application_device(device));self.train()
        if archive:
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'run.pt';self.assertTrue(self.w.save_current_run(path));self.w.undo_stack.setClean()
                self.assertTrue(self.w.open_saved_run(path))
            self.assertIsNone(self.w.last_run);self.assertIsNotNone(self.w.loaded_checkpoint)
        args=self.args(3);before=self.w.undo_stack.index()
        start=self.c.execute('resume_training',args);self.assertEqual(start['operation'],'resume_training')
        self.assertEqual(start['resumed_from'],args['run_id']);self.assertEqual(start['step'],2)
        self.assertNotEqual(start['revision'],args['expected_revision']);self.assertTrue(start['current']);self.wait()
        status=self.c.execute('training_status',dict(session_id=start['session_id'],job_id=start['job_id']))
        self.assertEqual(status['state'],'Completed');self.assertEqual(status['step'],5);self.assertTrue(status['current'])
        self.assertEqual(self.w.undo_stack.index(),before+1);self.assertEqual(self.w.config.iterations,5)
        state=self.w.last_run.model.training_state()
        rows=self.c.execute('read_results',dict(session_id=start['session_id'],job_id=start['job_id'],limit=2))
        self.assertEqual(rows['total_rows'],5);self.assertEqual(len(rows['rows']),2)
        self.assertNotEqual(self.read()['continuation']['run_id'],args['run_id'])
        # Compare like-for-like native workers, not a parent-process CUDA context.
        self.train();direct=self.w.last_run.model.training_state()
        self.assertEqual(state['loss_history'],direct['loss_history'])
        for name,value in state['network'].items():self.assertTrue(torch.equal(value,direct['network'][name]),name)
        self.w.undo_edit();self.assertEqual(self.w.config.iterations,2);self.assertIsNone(self.w.result)
        with self.assertRaises(AutomationError) as caught:
            self.c.execute('read_results',dict(session_id=start['session_id'],job_id=start['job_id']))
        self.assertEqual(caught.exception.code,'stale_results')

    def test_cpu_retained_run_exact_continuation(self):self.workflow('cpu')
    def test_cpu_desktop_opened_archive_exact_continuation(self):self.workflow('cpu',archive=True)
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_retained_run_exact_continuation(self):self.workflow('cuda')


if __name__=='__main__':unittest.main()
