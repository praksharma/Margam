"""Study CSV exports reported snapshots without opening archives or changing state."""
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
try:
    from . import test_study_results as fixture
except ImportError:
    import test_study_results as fixture
from margam.study_results import collect,export_csv
from margam.ui.model_document import capture_model


class StudyCSVTests(unittest.TestCase):
    setUpClass=classmethod(fixture.StudyResultTests.setUpClass.__func__)
    setUp=fixture.StudyResultTests.setUp
    tearDown=fixture.StudyResultTests.tearDown
    wait=fixture.StudyResultTests.wait
    study=fixture.StudyResultTests.study
    run_grid=fixture.StudyResultTests.run_grid
    snapshot=fixture.StudyResultTests.snapshot
    settle=fixture.StudyResultTests.settle

    def test_all_trials_outputs_undefined_metrics_json_and_no_mutation(self):
        import torch
        output,_=self.run_grid('[[4],[-1]]','layers');data=collect(output);original=deepcopy(data)
        data['records'][0]['choices'][0]['block']='=HYPERLINK("bad")\nquoted, label'
        data['records'][0]['row']['metrics']={'a':dict(value=.125,status='defined',norm='relative_l2',fields=['u'],rows=3),'b':dict(value=None,status='zero_reference',norm='relative_l2',fields=['v'],rows=3)}
        data['records'][1]['row']['reason']='=formula,\n"quoted"'
        before=deepcopy(data);files=self.snapshot(output);model=capture_model(self.w);undo=self.w.undo_stack.index();rng=torch.get_rng_state().clone();target=self.root/'table.csv'
        with patch('margam.ui.run_document.load_run',side_effect=AssertionError('No payload load')),patch('margam.study_results.collect',side_effect=AssertionError('No implicit refresh')):
            self.assertEqual(export_csv(data,target),2)
        with target.open(newline='') as stream:rows=list(csv.DictReader(stream))
        self.assertEqual(rows[0]['metric:a:value'],'0.125');self.assertEqual(rows[0]['metric:b:value'],'');self.assertEqual(json.loads(rows[0]['metric:b:status_json']),'zero_reference')
        self.assertEqual(rows[1]['metric:a:value'],'');self.assertEqual(rows[1]['actual_steps'],'');self.assertEqual(rows[1]['reported_status'],'invalid')
        self.assertEqual(json.loads(rows[0]['choices_json']),data['records'][0]['choices']);self.assertEqual(json.loads(rows[1]['reason_json']),data['records'][1]['row']['reason']);self.assertTrue(rows[1]['reason_json'].startswith('"'))
        self.assertEqual(json.loads(rows[0]['metric:a:status_json']),'defined')
        self.assertEqual(rows[0]['summary_sha256'],original['summary_sha256']);self.assertEqual(rows[0]['metadata_scope'],'reported_snapshot_payloads_unverified')
        self.assertEqual(before,data);self.assertEqual(files,self.snapshot(output));self.assertEqual(model,capture_model(self.w));self.assertEqual(undo,self.w.undo_stack.index());self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_atomic_no_clobber_symlink_write_failure_and_nonfinite_cleanup(self):
        output,_=self.run_grid('[1]');data=collect(output);target=self.root/'table.csv';export_csv(data,target);saved=target.read_bytes()
        with self.assertRaises(FileExistsError):export_csv(data,target)
        self.assertEqual(saved,target.read_bytes());link=self.root/'link.csv';link.symlink_to(target)
        with self.assertRaises(FileExistsError):export_csv(data,link)
        self.assertEqual(saved,target.read_bytes())
        with patch('os.link',side_effect=OSError('publication failure')):
            with self.assertRaisesRegex(OSError,'publication failure'):export_csv(data,self.root/'failed.csv')
        self.assertFalse(list(self.root.glob('.margam-study-csv-*')));self.assertFalse((self.root/'failed.csv').exists())
        data['records'][0]['choices'][0]['value']=float('nan')
        with self.assertRaises(ValueError):export_csv(data,self.root/'bad.csv')
        self.assertFalse((self.root/'bad.csv').exists())

    def test_gui_export_cancel_busy_failed_refresh_and_snapshot_semantics(self):
        output,_=self.run_grid('[1]');dialog=self.w.show_study_results();self.assertFalse(dialog.export_table(self.root/'empty.csv'))
        dialog.open_folder(output);self.settle(dialog);self.assertTrue(dialog.export_button.isEnabled());before=capture_model(self.w)
        with patch('margam.ui.study_results.QFileDialog.getSaveFileName',return_value=('','')):self.assertFalse(dialog.export_table())
        # Export the displayed snapshot even when progress on disk has changed.
        (output/'summary.json').write_text('{}');self.assertTrue(dialog.export_table(self.root/'snapshot.csv'));self.assertEqual(before,capture_model(self.w))
        self.assertFalse(dialog.export_table(self.root/'snapshot.csv'));self.assertIn('failed',dialog.status.text())
        dialog.refresh();self.assertFalse(dialog.export_table(self.root/'busy.csv'));self.settle(dialog);self.assertFalse(dialog.export_button.isEnabled());self.assertFalse(dialog.export_table(self.root/'stale.csv'))


if __name__=='__main__':unittest.main()
