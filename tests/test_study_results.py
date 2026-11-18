"""Read-only collection, verification and desktop opening preserve native contracts."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
try:
    from . import test_study_cli as fixture
except ImportError:
    import test_study_cli as fixture
from margam.execution.study_runner import run_study,atomic_json
from margam.study_results import collect,prepare_open
from margam.ui.model_document import capture_model


class StudyResultTests(unittest.TestCase):
    setUpClass=classmethod(fixture.StudyCLITests.setUpClass.__func__)
    def setUp(self):
        fixture.StudyCLITests.setUp(self);self.async_errors=[];self.old_hook=sys.excepthook
        sys.excepthook=lambda *args:self.async_errors.append(args)
    def tearDown(self):
        try:
            fixture.StudyCLITests.tearDown(self)
            self.assertEqual(self.async_errors,[],'Unhandled Qt callback error')
        finally:sys.excepthook=self.old_hook
    wait=fixture.StudyCLITests.wait
    study=fixture.StudyCLITests.study

    def snapshot(self,path):return {p.relative_to(path).as_posix():(p.read_bytes(),p.stat().st_mtime_ns) for p in path.rglob('*') if p.is_file()}
    def settle(self,dialog):
        deadline=time.monotonic()+45
        while dialog.worker is not None and time.monotonic()<deadline:self.app.processEvents();time.sleep(.005)
        self.assertIsNone(dialog.worker);self.app.processEvents()
    def run_grid(self,values='[1,2]',parameter='iterations',device='cpu'):
        output=self.root/'run';report=run_study(self.study(values,parameter),output,device=device);return output,report

    def test_collection_is_data_only_and_preserves_model_rng_and_files(self):
        import torch
        output,report=self.run_grid('[[4],[-1],[8]]','layers');before=self.snapshot(output);model=capture_model(self.w);index=self.w.undo_stack.index();rng=torch.get_rng_state().clone()
        with patch('margam.execution.study_runner.runtime',side_effect=AssertionError('No runtime/device probe')):
            result=collect(output)
        self.assertEqual(result['report']['counts'],{'completed':2,'invalid':1});self.assertEqual(result['records'][1]['row']['metrics'],None)
        self.assertEqual([r['choices'][0]['value'] for r in result['records']],[[4],[-1],[8]])
        self.assertEqual(model,capture_model(self.w));self.assertEqual(index,self.w.undo_stack.index());self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(before,self.snapshot(output))
        code="import sys; from margam.study_results import collect; result=collect(sys.argv[1]); assert len(result['records'])==3; assert 'torch' not in sys.modules and 'PySide6' not in sys.modules"
        child=subprocess.run([sys.executable,'-c',code,str(output)],text=True,capture_output=True,timeout=20)
        self.assertEqual(child.returncode,0,child.stderr)

    def test_verified_private_copy_remains_usable_after_original_deleted(self):
        from margam.ui.run_document import load_run
        output,_=self.run_grid('[1]');collection=collect(output);before=self.snapshot(output)
        temporary,path=prepare_open(collection,0)
        try:
            self.assertEqual(before,self.snapshot(output));original=output/'DP0001/artifacts/run.margam-run.pt';original.unlink()
            self.assertEqual(load_run(path)['result']['steps'],1)
        finally:temporary.cleanup()
        self.assertFalse(path.exists())

    def test_refresh_changes_corruption_and_unfinished_open_refused(self):
        output,report=self.run_grid('[[4],[-1]]','layers');collection=collect(output)
        with self.assertRaisesRegex(ValueError,'completed'):prepare_open(collection,1)
        report['trials'][1]['reason']='new reason';atomic_json(output/'summary.json',report)
        with self.assertRaisesRegex(ValueError,'Refresh'):prepare_open(collection,0)
        collection=collect(output);archive=output/'DP0001/artifacts/run.margam-run.pt';archive.write_bytes(archive.read_bytes()+b'changed')
        # Browsing reports outcomes, and explicitly opening verifies actual payloads.
        self.assertEqual(collect(output)['records'][0]['row']['status'],'completed')
        with self.assertRaisesRegex(ValueError,'checksum'):prepare_open(collection,0)

    def test_resumed_attempt_history_and_verified_open(self):
        from margam.execution.transport import JobTransport
        from margam.execution.study_runner import resume_study
        from margam.ui.run_document import load_run
        path=self.study('[1]');output=self.root/'resumed'
        with patch.object(JobTransport,'run',side_effect=KeyboardInterrupt):report=run_study(path,output)
        self.assertEqual(report['state'],'interrupted');resume_study(output)
        result=collect(output);row=result['records'][0]['row']
        self.assertEqual(row['directory'],'DP0001/attempts/0002');self.assertEqual(row['history'][0]['status'],'interrupted')
        before=self.snapshot(output);temporary,path=prepare_open(result,0)
        try:self.assertEqual(load_run(path)['result']['steps'],1)
        finally:temporary.cleanup()
        self.assertEqual(before,self.snapshot(output))

    def test_bad_summary_identity_paths_metrics_and_symlinks_rejected(self):
        output,report=self.run_grid('[1]');original=(output/'summary.json').read_bytes()
        for mutate in (lambda r:r.update(version=1),lambda r:r['trials'][0].update(directory='../escape'),lambda r:r['trials'][0].update(attempt=True),
            lambda r:r['trials'][0].update(trial_id='wrong'),lambda r:r.update(counts={'failed':1}),lambda r:r['trials'][0].update(metrics={'bad':{'value':-1}}),lambda r:r['trials'][0].update(metrics={'bad':{'norm':'relative_l2','status':'defined','fields':['u'],'rows':1}})):
            value=deepcopy(report);mutate(value);atomic_json(output/'summary.json',value);before=self.snapshot(output)
            with self.assertRaises((ValueError,KeyError,TypeError)):collect(output)
            self.assertEqual(before,self.snapshot(output));(output/'summary.json').write_bytes(original)
        from margam.execution.study_state import execution_record
        anchor=(output/'execution.json').read_bytes()
        changed=deepcopy(report);changed['runtime']={}
        atomic_json(output/'summary.json',changed);atomic_json(output/'execution.json',execution_record(changed))
        with self.assertRaisesRegex(ValueError,'runtime metadata'):collect(output)
        (output/'summary.json').write_bytes(original);(output/'execution.json').write_bytes(anchor)
        other=self.root/'summary';other.write_bytes(original);(output/'summary.json').unlink();(output/'summary.json').symlink_to(other)
        with self.assertRaisesRegex(ValueError,'symlink'):collect(output)

    def test_dialog_browsing_selection_refresh_and_native_discard(self):
        output,_=self.run_grid('[[4],[-1]]','layers');before=capture_model(self.w);index=self.w.undo_stack.index();files=self.snapshot(output)
        dialog=self.w.show_study_results();self.assertTrue(dialog.open_folder(output));self.settle(dialog)
        self.assertEqual(dialog.table.rowCount(),2);self.assertEqual(dialog.table.item(1,2).text(),'invalid')
        self.assertEqual(capture_model(self.w),before);self.assertEqual(self.w.undo_stack.index(),index);self.assertEqual(files,self.snapshot(output))
        dialog.table.selectRow(1);self.assertFalse(dialog.open_selected());dialog.table.selectRow(0)
        with patch.object(self.w,'confirm_discard',return_value=False):self.assertTrue(dialog.open_selected());self.settle(dialog)
        self.assertEqual(capture_model(self.w),before);self.assertIsNone(self.w.result)
        with patch.object(self.w,'confirm_discard',return_value=True):self.assertTrue(dialog.open_selected());self.settle(dialog)
        self.assertEqual(self.w.result['steps'],2);self.assertIsNotNone(self.w.loaded_checkpoint)
        self.assertEqual(files,self.snapshot(output))
        for attribute in ('locked',):
            with patch.object(self.w.scene,attribute,True):self.assertFalse(dialog.open_selected())
        with patch.object(self.w.project_workspace,'busy',True):self.assertFalse(dialog.open_selected())
        with patch.object(self.w,'is_running',return_value=True):self.assertFalse(dialog.open_selected())
        original=(output/'summary.json').read_bytes();(output/'summary.json').write_text('{}')
        self.assertTrue(dialog.refresh());self.settle(dialog);self.assertIsNone(dialog.collection);self.assertFalse(dialog.open_selected())
        (output/'summary.json').write_bytes(original)
        self.assertTrue(dialog.refresh());self.settle(dialog);dialog.close()
        self.assertIs(dialog,self.w.show_study_results());dialog.close()

    @unittest.skipUnless(__import__('torch').cuda.is_available(),'CUDA required')
    def test_cuda_archive_collection_cpu_restore_and_owned_exports(self):
        self.w.load_burgers_reference_example()
        for key,value in [('points',8),('boundary_points',4),('boundary_sampling',{}),('layers',(4,))]:self.w.change(key,value)
        path=self.study('[1]');output=self.root/'run';run_study(path,output,device='cuda',exports=True)
        dialog=self.w.show_study_results();dialog.open_folder(output);self.settle(dialog)
        self.assertEqual(dialog.collection['report']['effective_execution']['device'],'cuda');self.assertGreater(dialog.metric_choice.count(),0)
        before=self.w.application_device
        with patch.object(self.w,'confirm_discard',return_value=True):dialog.open_selected();self.settle(dialog)
        self.assertEqual(self.w.application_device,before);self.assertEqual(self.w.result['steps'],1)
        self.assertTrue(self.w.result['outputs']);self.assertIsNotNone(self.w.project_workspace._temporary);self.assertTrue(any(self.w.project_workspace.directory.rglob('*.csv')));dialog.close()

    def test_background_cancellation_and_window_close_are_safe(self):
        from margam.ui.study_results import StudyResults
        dialog=self.w.show_study_results()
        def slow(check):
            while not check():time.sleep(.002)
            raise InterruptedError('cancelled')
        self.assertTrue(dialog.start_read(slow,'collect'));dialog.cancel_reading();self.settle(dialog)
        self.assertIsNone(dialog.collection)
        self.assertTrue(dialog.start_read(lambda check:object(),'collect'));dialog.worker.wait()
        dialog.cancel_reading();self.settle(dialog);self.assertIsNone(dialog.collection)
        self.assertTrue(dialog.start_read(slow,'collect'));dialog.close();self.assertIsNone(dialog.worker)
        # Reopening immediately must ignore an old queued finished signal.
        self.assertTrue(dialog.start_read(lambda check:None,'collect'));dialog.shutdown();self.app.processEvents();self.assertIsNone(dialog.worker)
        self.assertTrue(dialog.start_read(slow,'collect'));self.w.undo_stack.setClean();self.w.close();self.assertIsNone(dialog.worker)


if __name__=='__main__':unittest.main()
