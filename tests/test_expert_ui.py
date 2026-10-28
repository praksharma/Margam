"""Expert recipe editing, expansion, archive and deterministic worker resume."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class ExpertUITests(unittest.TestCase):
    def test_recipe_controls_network_replacement_archive_and_resume(self):
        code=r'''import time
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import asdict
from PySide6.QtWidgets import QApplication,QSpinBox
from margam.ui.app import Window
from margam.templates.poisson import build_pinn
from margam.ui.model_document import capture_model,decode_document
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
try:
 assert w.example_actions['expert_burgers'] in w.examples_1d_menu.actions()
 w.load_example('expert_burgers');assert w.validate_model(),w.validation.text()
 trainer=next(n for n in w.scene.nodes() if n.kind=='expert_method')
 w.scene.clearSelection();trainer.setSelected(True);app.processEvents()
 control=w.property_scroll.findChild(QSpinBox,'expert_balance_every');assert control
 control.setValue(2);assert w.scene.compile_config().expert_settings['balance_every']==2
 w.undo_edit();assert w.scene.compile_config().expert_settings['balance_every']==1000
 w.redo_edit();assert w.scene.compile_config().expert_settings['balance_every']==2
 w.change('points',32);w.change('iterations',5);w.change('layers',(8,8))
 w.change('network_options',dict(encoding='gaussian',features=4,scale=.16,weight_factorization=True))
 c=asdict(w.scene.compile_config());compact=w.collapse_recipe(trainer)
 assert compact.kind=='expert_recipe' and asdict(w.scene.compile_config())==c
 trainer=w.expand_recipe(compact);assert asdict(w.scene.compile_config())==c
 # Replaceable network is independent of Expert trainer semantics.
 network=next(n for n in w.scene.nodes() if n.kind=='network')
 w.scene.clearSelection();network.setSelected(True);w.change_network_type('fcnn')
 assert w.scene.compile_config().network_type=='fcnn';w.undo_edit();assert asdict(w.scene.compile_config())==c
 decode_document(capture_model(w))
 with TemporaryDirectory() as directory:
    path=Path(directory)/'expert.json';assert w.save_model(path) and w.load_model(path)
    assert asdict(w.scene.compile_config())==c and not w.output_views.entries
    assert w.start_training()
    def finish():
        deadline=time.monotonic()+20
        while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
        assert w.worker is None and w.result is not None,w.log.toPlainText()
    finish()
    assert w.expert_summary()['step']==5 and 'Expert update 5' in w.log.toPlainText()
    menu=w.create_block_menu(next(n for n in w.scene.nodes() if n.kind=='expert_method'))
    assert any(a.text()=='Expert training summary' for a in menu.actions());menu.deleteLater()
    archive=Path(directory)/'expert.pt';assert w.save_current_run(archive)
    w.undo_stack.setClean();assert w.open_saved_run(archive)
    assert w.expert_summary()['coefficients'] and w.resume_training(2);finish()
    continuous=build_pinn(w.config);continuous.train(7,display_every=0)
    assert w.last_run.model.loss_history==continuous.loss_history
finally:
 app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)
