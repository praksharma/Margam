"""Portable study identities, immutable inputs, guarded restoration and native parity."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
import unittest
import warnings
from unittest.mock import patch
import zipfile
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.model_document import capture_model
from margam.studies import catalogue,preview,config_from_document
from margam import study_document as sd


class StudyDocumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        self.w=Window(state_directory=False);self.w.set_application_device('cpu');self.folder=tempfile.TemporaryDirectory();self.root=Path(self.folder.name)
    def tearDown(self):
        if self.w.worker is not None:self.w.cancel_training();self.wait()
        self.w.undo_stack.setClean();self.w.close();self.app.clipboard().clear();self.app.processEvents();self.folder.cleanup()
    def wait(self):
        deadline=time.monotonic()+45
        while self.w.worker is not None and time.monotonic()<deadline:self.app.processEvents();time.sleep(.003)
        self.assertIsNone(self.w.worker)
    def axes(self):
        item=next(row for row in catalogue(capture_model(self.w)) if row['parameter']=='iterations')
        return [dict(node_id=item['node_id'],parameter='iterations',mode='candidates',text='[2,3]')]
    def capture(self,**kwargs):
        return sd.capture(capture_model(self.w),self.axes(),self.w.project_workspace,execution=dict(device='auto',wall_time_seconds=300),scene=self.w.scene,**kwargs)
    def dialog(self):
        dialog=self.w.show_design_points();dialog.add_axis();axis=self.axes()[0];combo=dialog.axes.cellWidget(0,0)
        combo.setCurrentIndex(combo.findData(json.dumps([axis['node_id'],axis['parameter']])));dialog.axes.cellWidget(0,1).setCurrentIndex(1);dialog.axes.cellWidget(0,2).setText('[2,3]');return dialog
    def test_roundtrip_stable_trials_intent_and_atomic_no_clobber(self):
        before=capture_model(self.w);index=self.w.undo_stack.index();study,files=self.capture();path=self.root/'study.margam-study'
        sd.write(path,study,files);loaded,contents=sd.read(path)
        self.assertEqual(loaded,study);self.assertEqual(contents,files);self.assertEqual(capture_model(self.w),before);self.assertEqual(self.w.undo_stack.index(),index)
        ids=[]
        for n in (0,1):
            identifier,model=sd.trial_model(loaded,n);ids.append(identifier)
            self.assertEqual(config_from_document(model).iterations,n+2)
        self.assertNotEqual(*ids);self.assertEqual(sd.trial_model(sd.read(path)[0],0)[0],ids[0])
        for n in (-1,True,2):
            with self.assertRaises(ValueError):sd.trial_model(study,n)
        original=path.read_bytes()
        with self.assertRaises(FileExistsError):sd.write(path,study,files)
        with self.assertRaises(ValueError):sd.write(path,study,files,overwrite=1)
        with self.assertRaises(ValueError):self.capture(include_links=1)
        with patch('margam.study_document.os.replace',side_effect=OSError('injected publication failure')):
            with self.assertRaises(OSError):sd.write(path,study,files,overwrite=True)
        self.assertEqual(path.read_bytes(),original);self.assertFalse(list(self.root.glob('.margam-study-*')))
        def race(source,destination):
            Path(destination).write_bytes(b'competing writer');raise FileExistsError('publication race')
        other=self.root/'race.margam-study'
        with patch('margam.study_document.os.link',side_effect=race):
            with self.assertRaises(FileExistsError):sd.write(other,study,files)
        self.assertEqual(other.read_bytes(),b'competing writer');self.assertFalse(list(self.root.glob('.margam-study-*')))
        self.w.config.layers=(99,);self.w.config.device='cuda'
        self.assertEqual(self.capture()[0],study) # Inspector cache/device cannot override graph intent.
    def test_owned_reference_portability_reopen_resave_and_native_output_parity(self):
        import torch
        self.assertTrue(self.w.load_burgers_reference_example())
        for key,value in [('iterations',2),('points',8),('boundary_points',4),('boundary_sampling',{}),('layers',(4,))]:self.w.change(key,value)
        self.assertTrue(self.w.start_training());self.wait();self.assertIsNotNone(self.w.result)
        original_state=self.w.last_run.model.training_state();original_outputs=deepcopy(self.w.result['outputs']);original_run=self.w.last_run
        document=capture_model(self.w);index=self.w.undo_stack.index();dialog=self.dialog();self.app.processEvents()
        dialog.device.setCurrentIndex(dialog.device.findData('cuda'));dialog.time_limit.setValue(300)
        path=self.root/'portable.margam-study';self.assertTrue(dialog.save_study(path),dialog.status.text())
        self.assertIs(self.w.last_run,original_run);self.assertEqual(capture_model(self.w),document);self.assertEqual(self.w.undo_stack.index(),index)
        study,files=sd.read(path);self.assertEqual(len(files),1);self.assertEqual(study['execution'],dict(device='cuda',wall_time_seconds=300))
        source_root=Path(self.w.project_workspace._temporary.name);self.w.undo_stack.setClean();self.assertTrue(dialog.open_study(path),dialog.status.text())
        self.assertFalse(source_root.exists());self.assertIsNone(self.w.result);self.assertIsNone(self.w.loaded_checkpoint);self.assertIsNone(self.w.worker)
        self.assertEqual(self.w.application_device,'cpu');self.assertEqual(self.w.job_time_limit_seconds,0);self.assertEqual(dialog.axes_values(),study['axes'])
        path2=self.root/'moved.margam-study';self.assertTrue(dialog.save_study(path2),dialog.status.text());self.assertEqual(sd.read(path2)[0]['study_id'],study['study_id'])
        self.assertTrue(dialog.refresh());self.assertEqual(dialog.result['total'],2)
        self.assertTrue(self.w.start_training());self.wait();self.assertIsNotNone(self.w.result)
        state=self.w.last_run.model.training_state();self.assertEqual(state['loss_history'],original_state['loss_history'])
        for key,value in state['network'].items():self.assertTrue(torch.equal(value,original_state['network'][key]))
        for key,output in self.w.result['outputs'].items():
            for field in ('points','values','reference','point_ids'):
                if field in output:self.assertEqual(output[field],original_outputs[key][field])
        code="""import sys
from margam.study_document import read,restore,trial_model
from margam.studies import preview
from types import SimpleNamespace
study,files=read(sys.argv[1]);workspace=restore(files)
try:
 scene=SimpleNamespace(project_workspace=lambda:workspace,execution_device=lambda:'cpu')
 assert trial_model(study,0)[0].endswith(study['study_id'])
 assert 'torch' not in sys.modules and 'PySide6' not in sys.modules
 assert preview(study['model'],study['axes'],scene=scene)['valid']==2
 assert 'PySide6' not in sys.modules
finally:workspace.close()
"""
        result=subprocess.run([sys.executable,'-c',code,str(path2)],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
    def test_linked_copy_is_explicit_and_does_not_rebind_historical_fingerprints(self):
        from margam.ui.source_fingerprints import accepted_fingerprint
        self.w.load_example('burgers_1d');node=self.w.add_node('Evaluation points')
        path=Path(__file__).resolve().parents[1]/'assets/evaluation/burgers-reference/reference.csv'
        mapping=dict(path=str(path),coordinates='x,t',values='',ids='point_id');node.settings['source_data']=mapping
        record=accepted_fingerprint(self.w.project_workspace,mapping);node.settings['data_fingerprints']={'source_data':record}
        before=capture_model(self.w)
        with self.assertRaisesRegex(ValueError,'Include linked'):self.capture()
        study,files=self.capture(include_links=True);copied=next(row for row in study['model']['nodes'] if row['id']==node.identifier)
        self.assertTrue(copied['settings']['source_data']['path'].startswith('margam-owned:study_inputs/'))
        self.assertEqual(copied['settings']['data_fingerprints']['source_data'],record);self.assertEqual(len(files),1)
        self.assertEqual(capture_model(self.w),before)
        link=self.root/'alias.csv';link.symlink_to(path);node.settings['source_data']['path']=str(link)
        with self.assertRaises(OSError):self.capture(include_links=True)
        link.unlink();node.settings['source_data']['path']=str(link)
        with self.assertRaises(OSError):self.capture(include_links=True)
    def test_payload_checksum_size_missing_file_and_manifest_budgets(self):
        self.assertTrue(self.w.load_burgers_reference_example());study,files=self.capture()
        name=next(iter(files));original=files[name];path=self.root/'tampered.margam-study'
        for content in (b'x'+original[1:],original+b'\n'):
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('study.json',sd.canonical(study));archive.writestr('files/'+name,content)
            with self.assertRaisesRegex(ValueError,'checksum|size'):sd.read(path)
        with zipfile.ZipFile(path,'w') as archive:archive.writestr('study.json',sd.canonical(study))
        with self.assertRaisesRegex(ValueError,'missing|unexpected'):sd.read(path)
        sd.write(path,study,files,overwrite=True)
        with patch('margam.study_document.MANIFEST_LIMIT',16):
            with self.assertRaises(ValueError):sd.read(path)
        with patch('margam.study_document.MAX_FILES',0):
            with self.assertRaises(ValueError):sd.read(path)

    def test_malformed_tampered_duplicate_paths_and_byte_budgets(self):
        study,files=self.capture();path=self.root/'broken.margam-study'
        def put(entries):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',UserWarning)
                with zipfile.ZipFile(path,'w') as archive:
                    for name,content in entries:archive.writestr(name,content)
        for change in (dict(version=True),dict(study_id='0'*64),dict(execution=dict(device='cuda',wall_time_seconds=True))):
            invalid=dict(study,**change)
            with self.assertRaises(ValueError):sd.validate(invalid)
        for entries in ([('study.json',sd.canonical(study)),('extra',b'')],[('study.json',b'{"format":1,"format":2}')],[('../escape',b'bad')],[('study.json',sd.canonical(study)),('study.json',sd.canonical(study))]):
            put(entries)
            with self.assertRaises(ValueError):sd.read(path)
        for name in ('../escape','/absolute','C:/escape','nested/../bad','bad\\name'):
            from margam.ui.clipboard_files import read_owned
            with self.assertRaises(ValueError):read_owned(self.root,name)
            invalid=deepcopy(study);invalid['files']={name:dict(size=0,sha256=hashlib.sha256(b'').hexdigest())};invalid['study_id']=sd.digest(sd.canonical({k:v for k,v in invalid.items() if k!='study_id'}))
            with self.assertRaises(ValueError):sd.validate(invalid)
        entry=zipfile.ZipInfo('study.json');entry.external_attr=(stat.S_IFLNK|0o777)<<16;put([(entry,b'target')])
        with self.assertRaises(ValueError):sd.read(path)
        put([('study.json',sd.canonical(study)),('big',b'x'*17)])
        with patch('margam.study_document.FILE_LIMIT',16):
            with self.assertRaises(ValueError):sd.read(path)
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_BZIP2) as archive:archive.writestr('study.json',sd.canonical(study))
        with self.assertRaises(ValueError):sd.read(path)
        path.write_bytes(b'not a zip')
        with self.assertRaises(ValueError):sd.read(path)
    def test_shipped_study_and_capture_manifest_match(self):
        root=Path(__file__).resolve().parents[1];path=root/'assets/studies/poisson-grid.margam-study'
        record=json.loads((root/'assets/screenshots/manual-study-preview.json').read_text())['study']
        study,files=sd.read(path);self.assertEqual(files,{})
        self.assertEqual(sd.digest(path.read_bytes()),record['sha256']);self.assertEqual(study['study_id'],record['study_id'])
        report=preview(study['model'],study['axes']);self.assertEqual((report['total'],report['valid'],report['total_updates'],report['maximum_parameters']),(12,12,24000,1185))

    def test_ui_bad_open_busy_and_discard_keep_current_model_and_draft(self):
        dialog=self.dialog();before=capture_model(self.w);axes=dialog.axes_values();path=self.root/'bad.margam-study';path.write_bytes(b'invalid')
        self.assertFalse(dialog.open_study(path));self.assertEqual(capture_model(self.w),before);self.assertEqual(dialog.axes_values(),axes)
        self.w.scene.locked=True;self.assertFalse(dialog.save_study(self.root/'new.margam-study'));self.assertFalse(dialog.open_study(path));self.w.scene.locked=False
        self.assertFalse((self.root/'new.margam-study').exists())
        study,files=self.capture();sd.write(path,study,files,overwrite=True)
        with patch.object(self.w,'confirm_discard',return_value=False):self.assertFalse(dialog.open_study(path))
        self.assertEqual(capture_model(self.w),before);self.assertEqual(dialog.axes_values(),axes)
        def became_busy():self.w.scene.locked=True;return True
        with patch.object(self.w,'confirm_discard',side_effect=became_busy):self.assertFalse(dialog.open_study(path))
        self.w.scene.locked=False;self.assertEqual(capture_model(self.w),before)
        capture=sd.capture
        def busy_after_capture(*args,**kwargs):
            result=capture(*args,**kwargs);self.w.scene.locked=True;return result
        output=self.root/'busy.margam-study'
        with patch('margam.study_document.capture',side_effect=busy_after_capture):self.assertFalse(dialog.save_study(output))
        self.w.scene.locked=False;self.assertFalse(output.exists());self.assertEqual(capture_model(self.w),before)


if __name__=='__main__':unittest.main()
