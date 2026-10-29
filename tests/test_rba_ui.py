"""RBA controls, compact recipes, raw/weighted monitors, archives and export."""
import os,subprocess,sys,unittest
from pathlib import Path


class RBAUITests(unittest.TestCase):
    def test_examples_mask_archives_and_physical_time_export(self):
        code=r"""import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import xml.etree.ElementTree as ET
from PySide6.QtWidgets import QApplication,QComboBox,QDoubleSpinBox
from margam.ui.app import Window
from margam.ui.model_document import capture_model,decode_document
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
def finish():
 deadline=time.monotonic()+30
 while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
 assert w.worker is None and w.result is not None,w.log.toPlainText()
try:
 for key in ('rba_allen_cahn','rba_helmholtz_2d'):
  assert w.load_example(key) and w.validate_model(),w.validation.text()
  assert w.example_actions[key] in w.rba_examples_menu.actions() and not w.config.adaptive_sampling
  trainer=next(n for n in w.scene.nodes() if n.kind=='rba_method');w.show_block_properties(trainer)
  assert w.property_scroll.findChild(QDoubleSpinBox,'rba_rate') and w.property_scroll.findChild(QComboBox,'rba_initialization')
  w.change('points',8);w.change('boundary_points',2);w.change('layers',(4,4));w.change('iterations',3);w.change('device','cpu')
  for node in w.scene.nodes():
   if node.kind in ('solution','comparison','export','paraview'):node.settings['resolution']=5
  trainer=next(n for n in w.scene.nodes() if n.kind=='rba_method');compact=w.collapse_recipe(trainer);assert compact;w.show_block_properties(compact)
  assert w.property_scroll.findChild(QDoubleSpinBox,'rba_decay');w.expand_recipe(compact)
  doc=capture_model(w);assert doc['version']==26;decode_document(doc)
  assert w.start_training();finish();assert w.result['steps']==3 and not w.last_run.model.attention.requires_grad
  monitor=next(n for n in w.scene.nodes() if n.kind=='monitor');w.open_block(monitor);app.processEvents()
  assert 'pde_weighted' in w.last_run.model.loss_history[-1]
  with patch('margam.ui.app.QMessageBox.information') as info:w.show_rba_summary();assert info.called
  with TemporaryDirectory() as directory:
   directory=Path(directory);model=directory/'model.json';assert w.save_model(model);archive=directory/'rba.pt';assert w.save_current_run(archive)
   w.undo_stack.setClean();assert w.open_saved_run(archive)
   with patch('margam.ui.app.QMessageBox.information') as info:w.show_rba_summary();assert info.called
   assert w.resume_training(additional=2);finish();assert w.result['steps']==5
   if key=='rba_allen_cahn':
    export=next(n for n in w.scene.nodes() if n.kind=='paraview');path=directory/'series.pvd';assert w.export_block_data(export,path)
    assert [float(row.get('timestep')) for row in ET.parse(path).findall('Collection/DataSet')]==[0.,.25,.5,.75,1.]
finally:app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stderr+'\n'+result.stdout)
