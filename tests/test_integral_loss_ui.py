"""Loss formulation survives undo, compact recipes, files and background training."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class IntegralLossUITests(unittest.TestCase):
    def test_loss_choice_recipes_migration_and_worker(self):
        code=r'''import time
from dataclasses import asdict
from tempfile import TemporaryDirectory
from pathlib import Path
from PySide6.QtWidgets import QApplication,QComboBox
from margam.ui.app import Window
from margam.ui.model_document import capture_model,decode_document
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
try:
 for example,trainer_kind in [('interval_poisson','method'),('self_adaptive_poisson','sa_method'),('lra_poisson','lra_method')]:
    w.load_example(example)
    trainer=next(n for n in w.scene.nodes() if n.kind==trainer_kind)
    w.scene.clearSelection();trainer.setSelected(True);app.processEvents()
    choice=w.property_scroll.findChild(QComboBox,'loss_formulation');assert choice
    choice.setCurrentIndex(choice.findData('integral'))
    assert w.scene.compile_config().loss_formulation=='integral'
    w.undo_edit();assert w.scene.compile_config().loss_formulation=='mean'
    w.redo_edit();assert w.scene.compile_config().loss_formulation=='integral'
    w.change('x_max',3.);w.change('points',8);w.change('layers',(8,));w.change('iterations',3)
    before=asdict(w.scene.compile_config());compact=w.collapse_recipe(trainer)
    assert compact and asdict(w.scene.compile_config())==before
    w.scene.clearSelection();compact.setSelected(True);app.processEvents()
    choice=w.property_scroll.findChild(QComboBox,'loss_formulation');assert choice.currentData()=='integral'
    w.expand_recipe(compact);assert asdict(w.scene.compile_config())==before
    doc=capture_model(w);assert doc['version']==23;decode_document(doc)
    with TemporaryDirectory() as directory:
        p=Path(directory)/'integral.json';assert w.save_model(p) and w.load_model(p)
        assert asdict(w.scene.compile_config())==before
        assert w.start_training()
        deadline=time.monotonic()+20
        while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
        assert w.worker is None and w.result is not None,w.log.toPlainText()
        archive=Path(directory)/'integral.pt';assert w.save_current_run(archive)
        w.undo_stack.setClean();assert w.open_saved_run(archive)
        assert w.config.loss_formulation=='integral'
 # Old compact/expanded trainer instances migrate to mean, keeping exact wiring.
 w.load_example('interval_poisson');trainer=next(n for n in w.scene.nodes() if n.kind=='method')
 for compact in (False,True):
    if compact:trainer=w.collapse_recipe(trainer)
    old=capture_model(w);old['version']=13;old['config'].pop('loss_formulation')
    for n in old['nodes']:
        if 'loss_formulation' in n['settings'].get('parameters',{}):n['settings']['parameters'].pop('loss_formulation')
    c,nodes,wires=decode_document(old)
    assert c.loss_formulation=='mean'
    assert next(n for n in nodes if n['id']==trainer.identifier)['settings']['parameters']['loss_formulation']=='mean'
finally:
 app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)
