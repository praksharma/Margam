"""Real isolated DSGNAR desktop recipes, checkpoints and exports."""
import os,subprocess,sys,unittest
import torch
from pathlib import Path


class DSGNARUITests(unittest.TestCase):
    def _workflow(self,device):
        code=r"""import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from PySide6.QtWidgets import QApplication,QComboBox,QSpinBox,QMessageBox
from margam.ui.app import Window
from margam.ui.model_document import capture_model,decode_document
def fail(parent,title,message,*args):raise RuntimeError(str(title)+': '+str(message))
QMessageBox.critical=QMessageBox.warning=fail
app=QApplication([]);w=Window(state_directory=False);assert w.set_application_device('cpu')
def finish():
 deadline=time.monotonic()+60
 while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.003)
 assert w.worker is None and w.result is not None,w.log.toPlainText()
try:
 for key in ('dsgnar_poisson_1d','dsgnar_burgers','dsgnar_cavity','dsgnar_wave','dsgnar_multiscale'):
  assert w.load_example(key) and w.validate_model(),w.validation.text()
  assert w.example_actions[key] in w.dsgnar_examples_menu.actions()
  if key=='dsgnar_cavity':assert w.config.physics=='incompressible_2d' and w.config.field_names==('u','v','p')
  w.change('layers',(3,));w.change('points',5);w.change('boundary_points',2);w.change('iterations',2)
  settings=dict(w.config.dsgnar_settings,sketch_size=4,probes=3,batch_size=3);w.change('dsgnar_settings',settings)
  trainer=next(n for n in w.scene.nodes() if n.kind=='dsgnar_method');w.show_block_properties(trainer)
  assert w.property_scroll.findChild(QSpinBox,'dsgnar_sketch_size') and w.property_scroll.findChild(QComboBox,'dsgnar_precision')
  compact=w.collapse_recipe(trainer);assert compact;w.show_block_properties(compact);w.expand_recipe(compact)
  document=capture_model(w);assert document['version']==32;decode_document(document)
  for n in w.scene.nodes():
   if n.kind in ('solution','comparison','export','paraview'):n.settings['resolution']=3
  with TemporaryDirectory() as directory:
   directory=Path(directory);assert w.save_model(directory/'model.json');w.undo_stack.setClean();assert w.load_model(directory/'model.json')
   assert w.start_training();finish();assert w.result['steps']==2
   assert w.last_run.headless and w.last_run.model.storage_device=='cpu'
   with patch('margam.ui.app.QMessageBox.information') as info:w.show_dsgnar_summary();assert info.called
   assert w.save_current_run(directory/'run.pt');w.undo_stack.setClean();assert w.open_saved_run(directory/'run.pt')
   assert w.resume_training(1);finish();assert w.result['steps']==3
   from margam.templates.poisson import build_pinn
   import torch
   torch.set_num_threads(1)
   direct=build_pinn(w.config);direct.train(3,0)
   # Continuation is exact against the same fresh-process execution context.
   # Qt-host CUDA SVD/library plans can differ at the last bit.
   from margam.execution.codec import encode_config
   from margam.execution.transport import JobTransport
   full=JobTransport().run(dict(operation='train',config=encode_config(w.config),requests=[],checkpoint=None,metadata={},dtype=w.config.dsgnar_settings['precision']))['checkpoint']
   assert full['loss_history']==w.last_run.model.loss_history
   for name,value in full['network'].items():torch.testing.assert_close(value,w.last_run.model.network.state_dict()[name],rtol=0,atol=0)
   assert torch.equal(full['dsgnar']['generator'],w.last_run.model.training_state()['dsgnar']['generator'])
   assert len(direct.loss_history)==len(w.last_run.model.loss_history)
   for a,b in zip(direct.loss_history,w.last_run.model.loss_history):
    assert a.keys()==b.keys()
    torch.testing.assert_close(torch.tensor(list(a.values()),dtype=torch.float64),torch.tensor([b[k] for k in a],dtype=torch.float64),rtol=1e-11,atol=1e-12)
   for name,value in direct.network.state_dict().items():torch.testing.assert_close(value.cpu(),w.last_run.model.network.state_dict()[name],rtol=1e-11,atol=1e-12)
   assert w.evaluate_trained_outputs();finish()
   export=next(n for n in w.scene.nodes() if n.kind=='export');assert w.export_block_data(export,directory/'prediction.csv')
   if key in ('dsgnar_burgers','dsgnar_cavity','dsgnar_wave'):
    suffix='prediction.pvd' if key in ('dsgnar_burgers','dsgnar_wave') else 'prediction.vti'
    export=next(n for n in w.scene.nodes() if n.kind=='paraview');assert w.export_block_data(export,directory/suffix)
 # Earlier format-32 DSGNAR documents gain a zero stage grace in config/nodes.
 document=capture_model(w);document['config']['dsgnar_settings'].pop('stage_grace')
 for item in document['nodes']:
  values=item['settings'].get('parameters',{}).get('dsgnar_settings')
  if values is not None:values.pop('stage_grace',None)
 restored,nodes,_=decode_document(document);assert restored.dsgnar_settings['stage_grace']==0
 assert next(n for n in nodes if n['kind']=='dsgnar_method')['settings']['parameters']['dsgnar_settings']['stage_grace']==0
 # Legacy model 31 gains defaults without changing old recipes.
 document=capture_model(w);document['version']=31;document['config'].pop('dsgnar_settings')
 assert decode_document(document)[0].dsgnar_settings['precision']=='float64'
finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
"""
        code=code.replace("set_application_device('cpu')",f"set_application_device('{device}')")
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=240)
        self.assertEqual(result.returncode,0,result.stderr+'\n'+result.stdout)

    def test_examples_recipe_checkpoint_evaluation_and_export(self):self._workflow('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_isolated_examples_exact_resume_and_export(self):self._workflow('cuda')
