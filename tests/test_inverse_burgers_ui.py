"""Explicit inverse graph, portable observations, archive loading and inference."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class InverseBurgersUITests(unittest.TestCase):
    def test_workflow_and_ui_ownership(self):
        code=r'''
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from PySide6.QtWidgets import QApplication,QLabel
from margam.ui.app import Window
from margam.ui.model_document import capture_model,decode_document
from margam.ui.run_summary import execution_plan
from margam.recipes import COMPACT_RECIPES
app=QApplication([]);w=Window(state_directory=False)
def wait():
 while w.worker is not None:app.processEvents()
 app.processEvents()
try:
 w.set_application_device('cpu');assert w.load_inverse_example()
 assert not any(n.kind in ('boundary','bc') for n in w.scene.nodes())
 assert not w.output_views.entries
 assert any(label.text().startswith('PDE collocation: observed') for label in w.property_scroll.widget().findChildren(QLabel))
 doc=capture_model(w);assert doc['version']==30;decode_document(doc)
 assert w.config.interior_data['path'].startswith('margam-owned:') or 'imports' in w.config.interior_data['path'] or 'datasets' in w.config.interior_data['path']
 c,counts,text=execution_plan(w);assert c.method=='inverse_burgers' and counts['pde']==2000 and counts['boundary']==0
 source=next(n for n in w.scene.nodes() if n.kind=='observations');assert 'data_fingerprints' in source.settings
 overview=w.show_project_data();assert any(row['role']=='Observations' for row in overview.records);overview.close()
 w.change('iterations',2);w.change('layers',(4,))
 trainer=next(n for n in w.scene.nodes() if n.kind=='inverse_method')
 assert trainer.settings['parameters']['iterations']==2
 assert w.collapse_recipe(trainer)
 compact=next(n for n in w.scene.nodes() if n.kind=='inverse_recipe');assert w.validate_model()
 w.expand_recipe(compact);assert w.validate_model()
 assert w.start_training();wait();assert w.result is not None
 values,history=w.inverse_parameters();assert len(history)==2 and values['lambda_2']>0
 dialog=w.show_inverse_parameters();dialog.close()
 monitor=next(n for n in w.scene.nodes() if n.kind=='monitor');w.output_views.open(monitor)
 assert any(label=='Observations (raw)' for label,_,_ in w.output_views.entries[monitor.identifier][1].series)
 with TemporaryDirectory() as directory:
  root=Path(directory);path=root/'inverse.margam.json';assert w.save_model(path)
  archive=root/'inverse.pt';assert w.save_current_run(archive)
  assert w.open_saved_run(archive) and w.inverse_parameters()[0]==values
  assert w.evaluate_trained_outputs();wait();assert w.inverse_parameters()[0]==values
  assert w.resume_training(2);wait();assert w.result['steps']==4 and len(w.inverse_parameters()[1])==4
  export=next(n for n in w.scene.nodes() if n.kind=='paraview');assert w.export_block_data(export,root/'predictions.pvd')
  assert (root/'predictions.pvd').exists()
  w.undo_stack.setClean();assert w.load_model(path) and w.config.method=='inverse_burgers' and w.result is None
 # Training coefficients remain owned by PDE, never overwritten by learned estimates.
 pde=next(n for n in w.scene.nodes() if n.kind=='inverse_pde');w.scene.clearSelection();pde.setSelected(True)
 initial=deepcopy(pde.settings['parameters']['inverse_settings']);w.change_inverse_option('lambda_1',.2)
 assert pde.settings['parameters']['inverse_settings']['lambda_1']==.2
 w.undo_edit();assert pde.settings['parameters']['inverse_settings']==initial
finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=120,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertNotIn('Traceback',result.stderr)

    def test_observation_mapping_and_missing_fixture_guard(self):
        code=r'''
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
from PySide6.QtWidgets import QApplication,QMessageBox
from margam.ui.app import Window
from margam.ui.data_mapping import DataMappingDialog
from margam.ui.model_document import capture_model
from margam.templates.inverse_burgers import inverse_example_config
from margam.components.inverse_config import read_observations
app=QApplication([]);w=Window(state_directory=False)
try:
 w.set_application_device('cpu')
 before=capture_model(w);bad=inverse_example_config();bad.interior_data['path']='/nonexistent/inverse-fixture.csv'
 with patch('margam.templates.inverse_burgers.inverse_example_config',return_value=bad),patch.object(QMessageBox,'warning'):
  assert not w.load_inverse_example()
 assert capture_model(w)==before
 assert w.load_inverse_example()
 node=next(n for n in w.scene.nodes() if n.kind=='observations');w.show_block_properties(node)
 previous=deepcopy(node.settings);index=w.undo_stack.index()
 with TemporaryDirectory() as directory:
  path=Path(directory)/'measured.npz';np.savez(path,points=[[.1,.2],[.3,.4]],values=[.5,.6])
  def accept(dialog):
   dialog.fields['path'].setText(str(path));dialog.fields['coordinates'].setText('points');dialog.fields['values'].setText('values');dialog.storage.choice.setCurrentIndex(0)
   assert not dialog.fields['normals'].isEnabled() and dialog.check()
   dialog.accept();return dialog.DialogCode.Accepted
  with patch.object(DataMappingDialog,'exec',accept):w.edit_observations()
  assert w.undo_stack.index()==index+1
  assert node.settings['parameters']['interior_data']['path'].startswith('margam-owned:')
  assert len(read_observations(w.config)[0])==2 and w.validate_model()
  w.undo_edit();assert node.settings==previous and len(read_observations(w.config)[0])==2000
  w.redo_edit();path.unlink();assert len(read_observations(w.config)[0])==2 and w.validate_model()
  w.clear_point_data();assert not w.validate_model() and not node.settings.get('data_fingerprints')
  w.undo_edit();assert w.validate_model() and len(read_observations(w.config)[0])==2
finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=60,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertNotIn('Traceback',result.stderr)
