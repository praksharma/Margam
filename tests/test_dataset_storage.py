"""Managed CSV/NPZ inputs, explicit links and complete portable-data lifecycle."""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
import torch
from margam.ui.project_workspace import ProjectWorkspace
from margam.components.prepared_data import read_points,LIMIT


def mapping(path):return dict(path=str(path),coordinates='x',normals='',values='')


class DatasetStorageTests(unittest.TestCase):
    def test_copy_is_fixed_and_failed_copy_rolls_back(self):
        with TemporaryDirectory() as folder:
            original=Path(folder)/'input.csv';original.write_text('x\n0.2\n0.8\n')
            workspace=ProjectWorkspace()
            try:
                validator=lambda source:read_points(source,1)
                owned=workspace.store_dataset(mapping(original),copy=True,validator=validator)
                self.assertTrue(owned['path'].startswith('margam-owned:'))
                saved=Path(workspace.resolve(owned['path']));self.assertEqual(saved.read_bytes(),original.read_bytes())
                original.write_text('x\n0.5\n');original.unlink()
                self.assertEqual(read_points(workspace.resolve_source(owned),1)[0].tolist(),[[.2],[.8]])
                self.assertEqual(workspace.store_dataset(owned,copy=True,validator=validator),owned)
                with self.assertRaisesRegex(ValueError,'external file'):workspace.store_dataset(owned,copy=False,validator=validator)
                bad=Path(folder)/'bad.csv';bad.write_text('not_x\n1\n')
                before=set(workspace.directory.rglob('*'))
                with self.assertRaises(ValueError):workspace.store_dataset(mapping(bad),copy=True,validator=validator)
                self.assertEqual(set(workspace.directory.rglob('*')),before)
            finally:workspace.close()
            # Resolving an owned URI must preserve ownership through a folder alias.
            real=Path(folder)/'real';real.mkdir();alias=Path(folder)/'alias'
            alias.symlink_to(real,target_is_directory=True)
            workspace=ProjectWorkspace(alias/'model.margam.json')
            original.write_text('x\n0.2\n')
            owned=workspace.store_dataset(mapping(original),copy=True,validator=validator)
            self.assertEqual(workspace.portable(workspace.resolve(owned['path'])),owned['path'])
            loop=Path(folder)/'loop.csv';loop.symlink_to(loop.name)
            self.assertEqual(workspace.portable(str(loop)),str(loop))

    def test_links_and_oversize_files_do_not_create_workspace(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'linked.csv';path.write_text('x\n0.2\n')
            workspace=ProjectWorkspace();validator=lambda source:read_points(source,1)
            link=workspace.store_dataset(mapping(path),copy=False,validator=validator)
            self.assertEqual(link['path'],str(path));self.assertIsNone(workspace._temporary)
            path.write_text('x\n0.4\n')
            self.assertEqual(read_points(link,1)[0].tolist(),[[.4]])
            with path.open('wb') as stream:stream.truncate(LIMIT+1)
            with self.assertRaisesRegex(ValueError,'64 MiB'):workspace.store_dataset(mapping(path),copy=True,validator=validator)
            self.assertIsNone(workspace._temporary)
            workspace.busy=True
            with self.assertRaisesRegex(ValueError,'editor'):workspace.store_dataset(mapping(path),copy=False,validator=validator)

    def run_process(self,args,*,cpu=True,timeout=90):
        env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1')
        if cpu:env['CUDA_VISIBLE_DEVICES']=''
        result=subprocess.run([sys.executable,*args],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,env=env,timeout=timeout)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)

    def test_dialog_cancel_owned_preview_undo_and_legacy_reference(self):
        self.run_process(['-c',r'''
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.data_mapping import DataMappingDialog
from margam.components.prepared_data import read_points
app=QApplication([]);w=Window(state_directory=False);w.change('shape','Interval')
try:
 with TemporaryDirectory() as directory:
  path=Path(directory)/'input.csv';path.write_text('x\n0.2\n0.8\n')
  source=dict(path=str(path),coordinates='x',normals='',values='')
  dialog=DataMappingDialog(1,None,w,coordinate_only=True)
  assert dialog.storage.choice.currentData()=='copy'
  dialog.fields['path'].setText(str(path));assert dialog.check()
  assert w.project_workspace._temporary is None
  dialog.reject();assert w.project_workspace._temporary is None
  link=DataMappingDialog(1,source,w);assert link.storage.choice.currentData()=='link';link.close()
  first=w.project_workspace.store_dataset(source,copy=True,validator=lambda data:read_points(data,1))
  w.change('interior_data',first)
  owned=DataMappingDialog(1,w.config.interior_data,w,coordinate_only=True)
  assert owned.fields['path'].text()=='Project copy: input.csv' and owned.source()['path']==first['path']
  assert owned.check() and owned.storage.choice.currentData()=='copy'
  owned.storage.choice.setCurrentIndex(1);assert not owned.check();owned.close()
  path.write_text('x\n0.3\n0.7\n')
  second=w.project_workspace.store_dataset(source,copy=True,validator=lambda data:read_points(data,1))
  w.change('interior_data',second)
  assert w.save_model(Path(directory)/'model.margam.json')
  w.undo_edit();assert read_points(w.config.interior_data,1)[0].tolist()==[[.2],[.8]]
  w.redo_edit();assert read_points(w.config.interior_data,1)[0].tolist()==[[.3],[.7]]
  # The older Reference block uses its own mapped reference-data path.
  ref=w.add_node('Reference');comparison=w.add_node('Comparison')
  w.scene.connect_nodes(next(n for n in w.scene.nodes() if n.kind=='method'),comparison)
  w.scene.connect_nodes(ref,comparison);w.change_output(comparison,'resolution',3)
  reference=Path(directory)/'reference.csv';reference.write_text('x,u\n0,0\n0.5,0.25\n1,0\n')
  def fill():
   dialog=app.activeModalWidget();dialog.fields['path'].setText(str(reference));dialog.fields['values'].setText('u')
   assert dialog.storage.choice.currentData()=='copy' and dialog.check();dialog.accept_checked()
  QTimer.singleShot(0,fill);w.edit_point_data(reference=ref)
  reference.unlink();path.unlink()
  assert ref.settings['reference_data']['path'].startswith('margam-owned:')
  assert w.validate_model(),w.log.toPlainText()
  w.change('iterations',1);assert w.start_training()
  deadline=time.monotonic()+20
  while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
  assert w.worker is None and w.result is not None,w.log.toPlainText()
  assert w.result['outputs'][comparison.identifier]['reference']==[0.,.25,0.]
finally:
 app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
'''])

    def test_portable_workflow_cpu(self):
        with TemporaryDirectory() as directory:self.run_process(['-m','examples.portable_data','--directory',directory])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_portable_workflow_cuda(self):
        with TemporaryDirectory() as directory:self.run_process(['-m','examples.portable_data','--directory',directory,'--device','cuda'],cpu=False)
