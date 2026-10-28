"""Scale library/recipe controls, worker diagnostics and restricted archives."""
import os
from pathlib import Path
import subprocess,sys,unittest


class ScaleUITests(unittest.TestCase):
    def test_recipe_examples_worker_and_resume(self):
        code=r"""import time
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import asdict
from PySide6.QtWidgets import QApplication,QDoubleSpinBox
from margam.ui.app import Window
from margam.ui.model_document import capture_model,decode_document
from margam.templates.poisson import build_pinn
from margam.templates.scale_examples import SCALE_EXAMPLES
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
def finish():
 deadline=time.monotonic()+30
 while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
 assert w.worker is None and w.result is not None,w.log.toPlainText()
try:
 with TemporaryDirectory() as directory:
  for key in SCALE_EXAMPLES:
   assert w.example_actions[key] in w.scale_examples_menu.actions()
   w.load_example(key);assert w.validate_model(),w.validation.text()
   trainer=next(n for n in w.scene.nodes() if n.kind=='scale_method');w.scene.clearSelection();trainer.setSelected(True);app.processEvents()
   widget=w.property_scroll.findChild(QDoubleSpinBox,'scale_tau_sc');assert widget;widget.setValue(.2)
   assert w.scene.compile_config().scale_settings['tau_sc']==.2
   w.change('points',8);w.change('boundary_points',2);w.change('layers',(4,4));w.change('iterations',3);w.change('device','cpu')
   options=dict(w.scene.compile_config().network_options);options['features']=2;w.change('network_options',options)
   before=asdict(w.scene.compile_config());compact=w.collapse_recipe(trainer);assert compact.kind=='scale_recipe'
   assert asdict(w.scene.compile_config())==before;trainer=w.expand_recipe(compact);assert asdict(w.scene.compile_config())==before
   decode_document(capture_model(w));assert w.start_training();finish()
   assert 'physical PDE' in w.log.toPlainText() and w.result['outputs']
   menu=w.create_block_menu(trainer);assert any(a.text()=='Scale correction diagnostics' for a in menu.actions());menu.deleteLater()
   path=Path(directory)/(key+'.pt');assert w.save_current_run(path);w.undo_stack.setClean();assert w.open_saved_run(path)
   assert w.resume_training(2);finish();full=build_pinn(w.config);full.train(5,0)
   assert w.last_run.model.loss_history==full.loss_history
finally:
 app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stderr+'\n'+result.stdout)
