"""gPINN recipes, gradient controls, monitor histories, persistence and exports."""
import os,subprocess,sys,unittest
from pathlib import Path


class GPINNUITests(unittest.TestCase):
    def test_examples_recipes_worker_archives_and_temporal_export(self):
        code=r"""import time
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET
from PySide6.QtWidgets import QApplication,QDoubleSpinBox,QComboBox
from margam.ui.app import Window
from margam.ui.model_document import capture_model,decode_document
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
def finish():
 deadline=time.monotonic()+30
 while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
 assert w.worker is None and w.result is not None,w.log.toPlainText()
try:
 for key in ('gpinn_poisson_2d','gpinn_burgers','gpinn_burgers_rar'):
  assert w.load_example(key) and w.validate_model(),w.validation.text()
  assert w.example_actions[key] in w.gpinn_examples_menu.actions()
  trainer=next(n for n in w.scene.nodes() if n.kind=='gpinn_method');w.show_block_properties(trainer)
  assert w.property_scroll.findChild(QDoubleSpinBox,'gpinn_weight_x')
  modes=w.property_scroll.findChild(QComboBox,'loss_formulation');assert not modes.model().item(modes.findData('integral')).isEnabled()
  w.change('points',8);w.change('boundary_points',2);w.change('layers',(4,4));w.change('iterations',3);w.change('device','cpu')
  from margam.templates.poisson import BoundarySampling
  w.change('boundary_sampling',{side:BoundarySampling(2,w.config.boundary_recipe(side).method,w.config.boundary_recipe(side).seed) for side in w.config.sides})
  for node in w.scene.nodes():
   if node.kind in ('solution','comparison','export','paraview'):node.settings['resolution']=5
  if key.endswith('_rar'):
   settings=dict(w.config.adaptive_sampling,interval=2,candidates=20,batch=2,maximum=20);w.change('adaptive_sampling',settings)
  trainer=next(n for n in w.scene.nodes() if n.kind=='gpinn_method');compact=w.collapse_recipe(trainer);assert compact
  w.show_block_properties(compact);assert w.property_scroll.findChild(QDoubleSpinBox,'gpinn_weight_x');w.expand_recipe(compact)
  doc=capture_model(w);assert doc['version']==26;decode_document(doc)
  assert w.start_training();finish();assert w.result['steps']==3
  record=w.last_run.model.loss_history[-1];assert 'pde_grad_x' in record
  monitor=next(n for n in w.scene.nodes() if n.kind=='monitor');w.open_block(monitor);app.processEvents()
  with TemporaryDirectory() as directory:
   directory=Path(directory);path=directory/'gpinn.pt';assert w.save_current_run(path);w.undo_stack.setClean();assert w.open_saved_run(path)
   assert w.resume_training(additional=2);finish();assert w.result['steps']==5
   if key!='gpinn_poisson_2d':
    export=next(n for n in w.scene.nodes() if n.kind=='paraview');path=directory/'series.pvd';assert w.export_block_data(export,path)
    assert [float(row.get('timestep')) for row in ET.parse(path).findall('Collection/DataSet')]==[0.,.25,.5,.75,1.]
finally:app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stderr+'\n'+result.stdout)
