"""Paired finite studies share bounded preview, portable and native trial resolution."""
from copy import deepcopy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import torch
import test_study_cli as fixture
from margam.studies import catalogue,preview,resolve_axes
from margam.study_document import capture,write,read,trial_model,validate,canonical,digest
from margam.execution.study_runner import run_study,resume_study
from margam.study_results import collect,prepare_open


class PairedStudyTests(unittest.TestCase):
    setUpClass=classmethod(fixture.StudyCLITests.setUpClass.__func__)
    setUp=fixture.StudyCLITests.setUp
    tearDown=fixture.StudyCLITests.tearDown
    wait=fixture.StudyCLITests.wait

    def model(self):
        from margam.ui.model_document import capture_model
        return capture_model(self.w)
    def axes(self,rows):
        available=catalogue(self.model());return [dict(node_id=next(item for item in available if item['parameter']==key)['node_id'],parameter=key,mode=mode,text=text) for key,mode,text in rows]
    def save(self,rows):
        axes=self.axes(rows);study,files=capture(self.model(),axes,self.w.project_workspace,execution=dict(device='cpu',wall_time_seconds=0),scene=self.w.scene,combination='paired')
        path=self.root/'paired.margam-study';write(path,study,files,overwrite=True);return path,study
    def rows(self):return [('network_type','candidates','["fcnn","siren"]'),('layers','candidates','[[4],[8,8]]'),('iterations','fixed','1')]

    def test_order_broadcast_cartesian_difference_limits_and_no_mutation(self):
        model=self.model();before=deepcopy(model);rng=torch.get_rng_state().clone();undo=self.w.undo_stack.index();axes=self.axes(self.rows())
        with patch('margam.templates.poisson.build_pinn',side_effect=AssertionError('No allocation')):
            paired=preview(model,axes,combination='paired');cartesian=preview(model,axes)
        self.assertEqual(paired['total'],2);self.assertEqual(cartesian['total'],4)
        self.assertEqual([[c['value'] for c in row['choices']] for row in paired['trials']], [['fcnn',[4],1],['siren',[8,8],1]])
        repeated=self.axes([('network_type','candidates','["fcnn","fcnn","siren"]'),('layers','candidates','[[4],[8],[8]]')])
        self.assertEqual(preview(model,repeated,combination='paired')['total'],3)
        with self.assertRaisesRegex(ValueError,'duplicate'):preview(model,repeated)
        bad=self.axes([('layers','candidates','[[4],[8],[16]]'),('iterations','candidates','[1,2]')])
        with self.assertRaisesRegex(ValueError,'matching lengths'):preview(model,bad,combination='paired')
        with self.assertRaises(ValueError):preview(model,axes,combination='conditional')
        large=self.axes([('iterations','range','[1,64,1]'),('seed','range','[1,64,1]')]);self.assertEqual(resolve_axes(model,large,'paired')[1],64)
        with self.assertRaisesRegex(ValueError,'512'):resolve_axes(model,large)
        self.assertEqual(model,before);self.assertEqual(self.model(),before);self.assertEqual(undo,self.w.undo_stack.index());self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_version_three_identity_roundtrip_defaults_and_malformed_modes(self):
        path,study=self.save(self.rows());saved,_=read(path);self.assertEqual(saved,study);self.assertEqual(saved['version'],3)
        self.assertEqual(saved['combination'],'paired');self.assertEqual(set(saved['network_defaults']),{'fcnn','siren'})
        for i in range(2):
            _,model=trial_model(saved,i);network=next(n for n in model['nodes'] if n['kind']=='network');self.assertEqual(network['settings']['parameters']['layers'],[[4],[8,8]][i])
        self.assertEqual(trial_model(saved,1)[0],f"DP0002-{saved['study_id']}")
        with self.assertRaises(ValueError):trial_model(saved,2)
        from margam.components.network_config import OPTION_SCHEMAS
        with patch.dict(OPTION_SCHEMAS['siren']['omega'],default=71.):
            validate(saved);_,frozen=trial_model(saved,1)
            network=next(n for n in frozen['nodes'] if n['kind']=='network')
            self.assertEqual(network['settings']['parameters']['network_options']['omega'],30.)
        for changes in [dict(combination='cartesian'),dict(combination='unknown'),dict(network_defaults={}),dict(version=2)]:
            changed=deepcopy(saved);changed.update(changes);changed['study_id']=digest(canonical({k:v for k,v in changed.items() if k!='study_id'}))
            with self.assertRaises(ValueError):validate(changed)
        _,numeric=self.save([('iterations','candidates','[1,2]')]);self.assertEqual(numeric['version'],3);self.assertEqual(numeric['network_defaults'],{})

    def test_gui_save_open_preview_and_mismatch_diagnostic(self):
        path,study=self.save(self.rows());dialog=self.w.show_design_points()
        with patch.object(self.w,'confirm_discard',return_value=True):self.assertTrue(dialog.open_study(path))
        self.assertEqual(dialog.combination.currentData(),'paired');self.assertTrue(dialog.refresh());self.assertEqual(dialog.result['total'],2)
        copy=self.root/'again.margam-study';self.assertTrue(dialog.save_study(copy));self.assertEqual(read(copy)[0]['study_id'],study['study_id'])
        dialog.combination.setCurrentIndex(dialog.combination.findData('cartesian'));self.assertIsNone(dialog.result);self.assertTrue(dialog.refresh());self.assertEqual(dialog.result['total'],4)
        dialog.combination.setCurrentIndex(dialog.combination.findData('paired'));dialog.axes.cellWidget(1,2).setText('[[4],[8],[16]]');self.assertFalse(dialog.refresh());self.assertIn('matching lengths',dialog.status.text())

    def run_case(self,device):
        path,study=self.save(self.rows());output=self.root/'results';report=run_study(path,output,device=device);self.assertEqual(report['counts'],{'completed':2})
        result=collect(output);self.assertEqual(len(result['records']),2);self.assertEqual(result['records'][1]['choices'][1]['value'],[8,8])
        temporary,archive=prepare_open(result,1)
        try:
            from margam.ui.run_document import load_run
            loaded=load_run(archive);self.assertEqual(loaded['result']['steps'],1)
        finally:temporary.cleanup()
        before=(output/'summary.json').read_bytes();resume_study(output);self.assertEqual(before,(output/'summary.json').read_bytes())
    def test_cpu_execution_collection_native_open_and_noop_resume(self):self.run_case('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_execution_collection_native_open_and_noop_resume(self):self.run_case('cuda')

    def test_paired_cli_inspection_gui_controller_and_collection(self):
        import time
        path,study=self.save(self.rows());child=fixture.StudyCLITests.cli(self,'study','inspect',path)
        self.assertEqual(child.returncode,0,child.stderr);self.assertEqual(json.loads(child.stdout)['trials'],2)
        dialog=self.w.show_study_execution();dialog.source.setText(str(path));dialog.output.setText(str(self.root/'desktop'))
        before=self.model();undo=self.w.undo_stack.index();rng=torch.get_rng_state().clone()
        def settle(predicate):
            deadline=time.monotonic()+60
            while not predicate() and time.monotonic()<deadline:self.app.processEvents();time.sleep(.005)
            self.assertTrue(predicate())
        try:
            self.assertTrue(dialog.review());settle(lambda:dialog.reader is None);self.assertIsNotNone(dialog.prepared,dialog.status.text())
            self.assertTrue(dialog.start());settle(lambda:not dialog.active)
            self.assertEqual(collect(self.root/'desktop')['report']['counts'],{'completed':2})
            self.assertEqual(before,self.model());self.assertEqual(undo,self.w.undo_stack.index());self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(dialog.open_results());browser=self.w.study_results_dialog;settle(lambda:browser.worker is None)
            self.assertEqual(browser.table.rowCount(),2);self.assertTrue(browser.export_table(self.root/'paired.csv'));browser.close()
        finally:
            if dialog.active:dialog.stop(force=True);settle(lambda:not dialog.active)
            dialog.shutdown_review();dialog.close()

    def test_invalid_pairs_keep_indices_and_slurm_bundle(self):
        from margam.slurm import export_job,run_job
        path,study=self.save([('layers','candidates','[[4],[-1]]'),('iterations','candidates','[1,2]')]);bundle=self.root/'bundle'
        export_job(path,bundle,cpus=1,memory_mb=1024,minutes=5);report=run_job(bundle)
        self.assertEqual(report['counts'],{'completed':1,'invalid':1});result=collect(bundle/'results');self.assertEqual(result['records'][1]['row']['trial_id'],f"DP0002-{study['study_id']}");self.assertIsNone(result['records'][1]['row']['metrics'])


if __name__=='__main__':unittest.main()
