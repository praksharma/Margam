"""Desktop Slurm export uses existing bundles without mutating the open project."""
import json
import os
import subprocess
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import torch
try:
    from . import test_study_cli as fixture
except ImportError:
    import test_study_cli as fixture
from margam.slurm import read_job
from margam.ui.model_document import capture_model


class SlurmExportUITests(unittest.TestCase):
    setUpClass=classmethod(fixture.StudyCLITests.setUpClass.__func__)
    setUp=fixture.StudyCLITests.setUp
    tearDown=fixture.StudyCLITests.tearDown
    wait=fixture.StudyCLITests.wait
    study=fixture.StudyCLITests.study

    def configured(self):
        d=self.w.show_slurm_export();d.source.setText(str(self.study('[1]')));d.output.setText(str(self.root/'job bundle'))
        d.cpus.setValue(1);d.memory.setValue(1024);d.minutes.setValue(5);return d

    def test_required_budgets_errors_and_no_overwrite(self):
        d=self.w.show_slurm_export();self.assertFalse(d.export_bundle())
        d.source.setText(str(self.study('[1]')));d.output.setText(str(self.root/'new'))
        self.assertFalse(d.export_bundle());self.assertFalse((self.root/'new').exists())
        d.cpus.setValue(1);d.memory.setValue(1024);d.minutes.setValue(5)
        d.partition.setText('bad;command');self.assertFalse(d.export_bundle());self.assertFalse((self.root/'new').exists())
        d.partition.clear();self.assertTrue(d.export_bundle());before={p.name:p.read_bytes() for p in (self.root/'new').iterdir()}
        self.assertFalse(d.export_bundle());self.assertEqual(before,{p.name:p.read_bytes() for p in (self.root/'new').iterdir()})
        d.output.setText(str(self.root/'bad'));d.source.setText(str(self.root/'missing'));self.assertFalse(d.export_bundle());self.assertFalse((self.root/'bad').exists())

    def test_export_uses_saved_snapshot_preserves_model_rng_and_no_launch(self):
        d=self.configured();saved=Path(d.source.text()).read_bytes();self.w.change('iterations',19)
        before=capture_model(self.w);undo=self.w.undo_stack.index();rng=torch.get_rng_state().clone()
        d.gpu.setChecked(True);d.exports.setChecked(True);d.partition.setText('gpu');d.account.setText('research')
        with patch('subprocess.Popen',side_effect=AssertionError('Export never launches a process')):
            self.assertTrue(d.export_bundle(),d.status.text())
        job=read_job(d.output.text());self.assertEqual(job['execution'],dict(device='cuda',exports=True));self.assertEqual(job['resources']['account'],'research')
        self.assertEqual(saved,(Path(d.output.text())/'study.margam-study').read_bytes())
        self.assertEqual(before,capture_model(self.w));self.assertEqual(undo,self.w.undo_stack.index());self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertIsNone(self.w.worker)
        self.assertIn('Nothing submitted',d.status.text());d.close();self.assertIs(self.w.show_slurm_export(),d)

    def test_desktop_export_local_bash_collection_and_native_open(self):
        from margam.study_results import collect,prepare_open
        d=self.configured();self.assertTrue(d.export_bundle(),d.status.text());bundle=Path(d.output.text())
        child=subprocess.run(['bash',str(bundle/'job.sbatch')],cwd=self.root,text=True,capture_output=True,timeout=100,
            env=dict(os.environ,SLURM_SUBMIT_DIR=str(bundle),MARGAM_PYTHON=sys.executable))
        self.assertEqual(child.returncode,0,child.stderr);result=collect(bundle/'results');self.assertEqual(result['report']['counts'],{'completed':1})
        temporary,path=prepare_open(result,0)
        try:
            with patch.object(self.w,'confirm_discard',return_value=True):self.assertTrue(self.w.open_saved_run(path))
        finally:temporary.cleanup()
        self.assertEqual(self.w.result['steps'],1);self.assertIsNotNone(self.w.loaded_checkpoint)


if __name__=='__main__':unittest.main()
