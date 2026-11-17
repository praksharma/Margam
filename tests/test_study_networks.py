"""Concrete network-family sweeps reuse native validators and default schemas."""
from copy import deepcopy
import json
import sys
import unittest
from unittest.mock import patch
try:
    from . import test_study_cli as fixture
except ImportError:
    import test_study_cli as fixture
from margam.studies import catalogue,values_for,preview,config_from_document
from margam.study_document import capture,write,read,trial_model
from margam.components.network_config import network_options
from margam.execution.study_runner import run_study
from margam.study_results import collect,prepare_open
from margam.ui.model_document import capture_model


class StudyNetworkTests(unittest.TestCase):
    setUpClass=classmethod(fixture.StudyCLITests.setUpClass.__func__)
    setUp=fixture.StudyCLITests.setUp
    tearDown=fixture.StudyCLITests.tearDown
    wait=fixture.StudyCLITests.wait
    study=fixture.StudyCLITests.study
    cli=fixture.StudyCLITests.cli
    def axis(self,key,text):
        row=next(row for row in catalogue(capture_model(self.w)) if row['parameter']==key)
        return dict(node_id=row['node_id'],parameter=key,mode='candidates',text=text)
    def test_categorical_values_are_explicit_and_bounded(self):
        self.assertEqual(values_for('network_type','fixed','"siren"'),['siren'])
        self.assertEqual(values_for('network_type','candidates','["fcnn","dgm"]'),['fcnn','dgm'])
        for mode,text in [('range','[1,2,1]'),('candidates','["unknown"]'),('candidates','["fcnn","fcnn"]'),('fixed','1'),('candidates','[]'),('candidates','"fcnn"'),('fixed','"__import__(os)"')]:
            with self.subTest(mode=mode,text=text),self.assertRaises(ValueError):values_for('network_type',mode,text)
    def test_shared_materialization_defaults_options_and_keeps_base_unchanged(self):
        self.w.change_network_type('siren');self.w.change_network_option('omega',7.)
        model=capture_model(self.w);before=deepcopy(model);undo=self.w.undo_stack.index()
        axes=[self.axis('network_type','["siren","dgm"]'),self.axis('layers','[[4,4],[4,8]]')]
        with patch('margam.templates.poisson.build_pinn',side_effect=AssertionError('No network')),patch('margam.templates.poisson.sample_problem',side_effect=AssertionError('No samples')):
            result=preview(model,axes,scene=self.w.scene)
        self.assertEqual((result['total'],result['valid'],result['invalid']),(4,3,1));self.assertIn('equal',result['trials'][3]['reason'])
        self.assertIsNone(result['trials'][3]['updates']);self.assertEqual(result['trials'][0]['networks'][0]['options']['omega'],30.)
        study,files=capture(model,axes,self.w.project_workspace,execution=dict(device='cpu',wall_time_seconds=0),scene=self.w.scene)
        for index,row in enumerate(result['trials']):
            _,candidate=trial_model(study,index);config=config_from_document(candidate)
            self.assertEqual(config.network_options,row['networks'][0]['options'])
            self.assertEqual(config.network_type,row['networks'][0]['architecture'])
        # Existing numeric-only studies retain custom architecture options exactly.
        old_axes=[self.axis('iterations','[1,2]')];old,_=capture(model,old_axes,self.w.project_workspace,execution=dict(device='cpu',wall_time_seconds=0),scene=self.w.scene)
        _,candidate=trial_model(old,0);self.assertEqual(config_from_document(candidate).network_options['omega'],7.)
        self.assertEqual(model,before);self.assertEqual(capture_model(self.w),before);self.assertEqual(undo,self.w.undo_stack.index())
    def test_dialog_and_portable_roundtrip_show_defaults_for_compact_and_expanded(self):
        dialog=self.w.show_design_points();dialog.add_axis();axis=self.axis('network_type','["fcnn","siren"]')
        combo=dialog.axes.cellWidget(0,0);combo.setCurrentIndex(combo.findData(json.dumps([axis['node_id'],'network_type'])))
        dialog.axes.cellWidget(0,1).setCurrentIndex(1);dialog.axes.cellWidget(0,2).setText(axis['text']);self.assertTrue(dialog.refresh())
        self.assertIn('resolved options',dialog.details.toPlainText());self.assertIn('custom options',dialog.details.toPlainText())
        path=self.root/'families.margam-study';self.assertTrue(dialog.save_study(path));study,_=read(path)
        self.assertEqual(study['axes'][0]['text'],axis['text']);self.assertEqual(study['version'],2)
        with patch.object(self.w,'confirm_discard',return_value=True):self.assertTrue(dialog.open_study(path))
        self.assertTrue(dialog.refresh())
        method=next(node for node in self.w.scene.nodes() if node.kind=='method');self.w.collapse_recipe(method)
        model=capture_model(self.w);row=next(row for row in catalogue(model) if row['parameter']=='network_type')
        result=preview(model,[dict(node_id=row['node_id'],parameter='network_type',mode='candidates',text='["fcnn","siren"]')],scene=self.w.scene)
        self.assertEqual(result['valid'],2);dialog.close()
    def test_frozen_defaults_survive_changed_library_defaults_and_reject_incomplete_maps(self):
        from margam.components.network_config import OPTION_SCHEMAS
        from margam.study_document import validate,canonical,digest
        model=capture_model(self.w);axes=[self.axis('network_type','["siren"]')]
        study,files=capture(model,axes,self.w.project_workspace,execution=dict(device='cpu',wall_time_seconds=0),scene=self.w.scene)
        self.assertEqual(study['network_defaults']['siren']['omega'],30.)
        path=self.root/'frozen.margam-study';write(path,study,files)
        with patch.dict(OPTION_SCHEMAS['siren']['omega'],default=71.):
            reopened,_=read(path);_,candidate=trial_model(reopened,0)
            self.assertEqual(config_from_document(candidate).network_options['omega'],30.)
            result=preview(model,axes,network_defaults=reopened['network_defaults']);self.assertEqual(result['trials'][0]['networks'][0]['options']['omega'],30.)
            dialog=self.w.show_design_points()
            with patch.object(self.w,'confirm_discard',return_value=True):self.assertTrue(dialog.open_study(path))
            self.assertTrue(dialog.refresh());self.assertEqual(dialog.result['trials'][0]['networks'][0]['options']['omega'],30.)
            again=self.root/'again.margam-study';self.assertTrue(dialog.save_study(again));self.assertEqual(read(again)[0]['network_defaults'],study['network_defaults'])
            dialog.clear_draft();self.assertIsNone(dialog.network_defaults);dialog.close()
        for mutate in (lambda value:value.update(network_defaults=[]),lambda value:value['network_defaults'].update(siren={}),lambda value:value.update(version=1)):
            value=deepcopy(study);mutate(value);value['study_id']=digest(canonical({key:item for key,item in value.items() if key!='study_id'}))
            with self.assertRaises((ValueError,TypeError)):validate(value)

    def run_families(self,device):
        from margam.ui.run_document import load_run
        path=self.study('["fcnn","fourier","siren"]','network_type');output=self.root/'families'
        result=self.cli('study','run',path,'--output',output,'--device',device);self.assertEqual(result.returncode,0,result.stderr)
        report=collect(output);self.assertEqual(report['report']['counts'],{'completed':3});self.assertTrue(report['report']['headless'])
        for index,kind in enumerate(('fcnn','fourier','siren')):
            detail=json.loads((output/f'DP{index+1:04d}/trial.json').read_text());self.assertEqual(detail['config']['network_type'],kind)
            self.assertEqual(detail['config']['network_options'],network_options(kind,{}))
            self.assertEqual(report['records'][index]['choices'][0]['value'],kind)
            temporary,archive=prepare_open(report,index)
            try:
                if kind=='siren':
                    from margam.components.network_config import OPTION_SCHEMAS
                    with patch.dict(OPTION_SCHEMAS['siren']['omega'],default=71.):
                        historical,copy=prepare_open(report,index);historical.cleanup()
                saved=load_run(archive);self.assertEqual(saved['result']['steps'],2);self.assertEqual(saved['result']['device'],device)
            finally:temporary.cleanup()
        # Resume verifies family-specific models/options and leaves completed trials alone.
        before=(output/'summary.json').read_bytes();result=self.cli('study','resume',output);self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(before,(output/'summary.json').read_bytes())
    def test_cpu_families_native_archives_and_completed_resume(self):self.run_families('cpu')
    @unittest.skipUnless(__import__('torch').cuda.is_available(),'CUDA required')
    def test_cuda_families_native_archives_and_completed_resume(self):self.run_families('cuda')
    def test_version_two_launches_through_slurm_bundle_and_gui_controller(self):
        from margam.slurm import export_job,read_job,run_job
        import time
        path=self.study('["fcnn","siren"]','network_type');bundle=self.root/'batch'
        export_job(path,bundle,cpus=1,memory_mb=1024,minutes=5);self.assertEqual(read(bundle/'study.margam-study')[0]['version'],2)
        self.assertEqual(run_job(bundle)['counts'],{'completed':2})
        dialog=self.w.show_study_execution();dialog.source.setText(str(path));dialog.output.setText(str(self.root/'desktop'))
        self.assertTrue(dialog.review());deadline=time.monotonic()+45
        while dialog.reader is not None and time.monotonic()<deadline:self.app.processEvents();time.sleep(.005)
        self.assertIsNotNone(dialog.prepared,dialog.status.text());self.assertTrue(dialog.start());deadline=time.monotonic()+60
        while dialog.active and time.monotonic()<deadline:self.app.processEvents();time.sleep(.005)
        if dialog.active:dialog.stop(force=True)
        self.assertFalse(dialog.active);self.assertEqual(collect(self.root/'desktop')['report']['counts'],{'completed':2});dialog.close()

    def test_invalid_width_family_combination_does_not_block_other_trials(self):
        model=capture_model(self.w);axes=[self.axis('network_type','["fcnn","dgm"]'),self.axis('layers','[[4,4],[4,8]]')]
        study,files=capture(model,axes,self.w.project_workspace,execution=dict(device='cpu',wall_time_seconds=0),scene=self.w.scene)
        path=self.root/'mixed.margam-study';write(path,study,files);report=run_study(path,self.root/'mixed')
        self.assertEqual(report['counts'],{'completed':3,'invalid':1});self.assertIsNone(report['trials'][3]['metrics'])


if __name__=='__main__':unittest.main()
