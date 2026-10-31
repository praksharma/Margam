"""Read-only data inventory, honest change baselines and mapping navigation."""
import os
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


class ProjectDataTests(unittest.TestCase):
    def run_ui(self,code):
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=75,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertNotIn('Traceback',result.stderr)

    def test_inventory_changes_missing_and_readonly(self):
        self.run_ui(r'''
from pathlib import Path
from tempfile import TemporaryDirectory
from copy import deepcopy
import shutil
import numpy as np
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.model_document import capture_model
from margam.ui.project_workspace import ProjectWorkspace
from margam.components.prepared_data import read_points
app=QApplication([]);w=Window(state_directory=False)
try:
 assert w.load_training_data_example()
 with TemporaryDirectory() as directory:
  directory=Path(directory)
  source=directory/'base.csv';shutil.copyfile('assets/training/triangle/base.csv',source)
  mappings=deepcopy(w.config.boundary_data);mappings['base']['path']=str(source);w.change('boundary_data',mappings)
  interior=directory/'interior.npz';np.savez(interior,points=[[.2,.2],[.1,.3]])
  owned=w.project_workspace.store_dataset(dict(path=str(interior),coordinates='points',normals='',values=''),copy=True,validator=lambda s:read_points(s,2))
  w.change('interior_data',owned)
  before=capture_model(w);history=(w.undo_stack.index(),w.undo_stack.isClean());result={'sentinel':True};w.result=result
  dialog=w.show_project_data();app.processEvents()
  assert len(dialog.records)==5
  assert all(row['status']=='Available' for row in dialog.results)
  assert sorted(row['rows'] for row in dialog.results)==[2,12,12,12,58]
  assert sum(row['storage']=='Project copy' for row in dialog.results)==1
  assert capture_model(w)==before and (w.undo_stack.index(),w.undo_stack.isClean())==history and w.result is result
  index=next(i for i,row in enumerate(dialog.records) if row['boundary']=='base')
  source.write_text(source.read_text()+'\n')
  dialog.refresh();assert dialog.results[index]['status']=='Changed since first check'
  dialog.refresh();assert dialog.results[index]['changed'] # Refresh must not accept a new baseline.
  source.unlink();dialog.refresh();assert dialog.results[index]['status']=='Missing'
  dialog.table.selectRow(index);assert 'Coordinates: x,y' in dialog.details.toPlainText()
  # Save As must retain session baselines for stable owned references.
  assert w.save_model(directory/'model.margam.json');dialog.refresh()
  own_index=next(i for i,row in enumerate(dialog.results) if row['storage']=='Project copy')
  assert dialog.results[own_index]['status']=='Available'
  shutil.rmtree(directory/'model_files')
  snapshot=capture_model(w);dialog.refresh()
  assert not (directory/'model_files').exists() and capture_model(w)==snapshot
  assert dialog.results[own_index]['status']=='Missing'
finally:
 w.result=None;app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
''')

    def test_navigation_stale_lock_and_empty_project(self):
        self.run_ui(r'''
from unittest.mock import patch
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
app=QApplication([]);w=Window(state_directory=False)
try:
 dialog=w.show_project_data();assert not dialog.records and 'No imported' in dialog.summary.text()
 assert w.load_training_data_example();dialog.refresh()
 index=next(i for i,r in enumerate(dialog.records) if r['boundary']=='slope');dialog.table.selectRow(index)
 node=dialog.select_block();assert node.kind=='boundary_data' and w.boundary_side=='slope' and node.isSelected()
 with patch.object(w,'edit_point_data') as edit:
  dialog.map_source();edit.assert_called_once_with('slope',reference=None)
 w.set_busy(True);assert not dialog.map_button.isEnabled()
 with patch.object(w,'edit_point_data') as edit:dialog.map_source();edit.assert_not_called()
 w.set_busy(False);dialog.refresh()
 w.change_bc('slope','value',1.);assert dialog.stale and not dialog.map_button.isEnabled()
 dialog.refresh();assert not dialog.stale
 w.reset_template();dialog.refresh();assert not dialog.records and not dialog.baselines
 # Unconfigured source blocks need attention; built-in samplers don't.
 source=w.add_node('Evaluation points');dialog.refresh()
 assert len(dialog.records)==1 and dialog.results[0]['status']=='Not mapped'
 w.delete_block(source);assert not dialog.select_button.isEnabled()
 dialog.refresh();assert not dialog.records
 # Both reference forms and their field/ID mappings appear without requiring a valid graph.
 w.load_evaluation_example()
 legacy=w.add_node('Reference')
 w.change_output(legacy,'reference_data',dict(path='assets/evaluation/reference-values.csv',coordinates='x',normals='',values='u'))
 dialog.refresh();assert len(dialog.records)==3 and all(r['rows']==4 for r in dialog.results)
 index=next(i for i,r in enumerate(dialog.records) if r['reader']=='reference_data');dialog.table.selectRow(index)
 assert 'Point IDs: point_id' in dialog.details.toPlainText() and 'Fields: u' in dialog.details.toPlainText()
 reference=next(n for n in w.scene.nodes() if n.kind=='reference_data')
 invalid=dict(reference.settings['source_data'],values='missing')
 w.change_output(reference,'source_data',invalid);dialog.refresh()
 index=next(i for i,r in enumerate(dialog.records) if r['reader']=='reference_data')
 assert dialog.results[index]['status']=='Invalid / unavailable' and dialog.results[index]['rows'] is None
finally:
 app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
''')

    @unittest.skipUnless(importlib.util.find_spec('jorvik'),'Optional Jorvik required')
    def test_jorvik_expected_fingerprint_and_unconnected_sources(self):
        self.run_ui(r'''
from pathlib import Path
from tempfile import TemporaryDirectory
from copy import deepcopy
from unittest.mock import patch
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.project_data import collect_sources,inspect_source
from tests.test_jorvik import fixture,SELECTION
app=QApplication([]);w=Window(state_directory=False)
try:
 with TemporaryDirectory() as directory:
  for node in list(w.scene.nodes()):
   if node.kind in ('domain','interior','boundary'):w.delete_block(node)
  geometry=w.add_node('Jorvik geometry');external=Path(directory)/'samples.nc';fixture(external)
  assert w.browse_jorvik(geometry,'export',external) and w.update_jorvik(geometry,SELECTION,confirm=False)
  # These data blocks are not wired to the PDE/BC yet, but remain inspectable.
  dialog=w.show_project_data();assert len(dialog.records)==3
  assert sorted(r['rows'] for r in dialog.results)==[2,3,4]
  record=dialog.records[0];assert record['expected']
  path=Path(w.project_workspace.resolve(record['source']['path']));path.write_bytes(path.read_bytes()+b'\n')
  outcome=inspect_source(record,w.project_workspace,{},{});assert outcome['changed']
  dialog.refresh();dialog.table.selectRow(0)
  assert dialog.map_button.text()=='Review Jorvik datasets…'
  with patch.object(w,'update_jorvik') as update:dialog.map_source();update.assert_called_once_with(geometry)
finally:
 app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
''')
