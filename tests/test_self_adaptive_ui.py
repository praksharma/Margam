"""SA recipe insertion, archive validation/resume and optional result views."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class SelfAdaptiveUITests(unittest.TestCase):
    def test_recipe_example_archive_and_resume(self):
        code=r'''
import time
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from PySide6.QtWidgets import QApplication,QMessageBox
from margam.ui.app import Window
from margam.ui.model_document import capture_model,decode_document
from margam.ui.run_document import load_run
from margam.recipes import self_adaptive
def unexpected_warning(*args):raise AssertionError(str(args[1:]))
QMessageBox.warning=unexpected_warning
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
assert any('Self-adaptive PINN' in a.text() for a in w.recipe_menu.actions())
root=w.insert_expanded_recipe(self_adaptive)
assert root.kind=='sa_method' and root.settings['parameters']['sa_lr']==.005
w.undo_edit()
assert w.load_example('self_adaptive_poisson')
assert w.validate_model() and not w.output_views.entries
root=next(n for n in w.scene.nodes() if n.kind=='sa_method')
w.scene.clearSelection();root.setSelected(True)
w.change('sa_boundary',True);w.change('sa_lr',.007);w.change('sa_initial',.75);w.change('sa_max',50.)
w.change('layers',(12,12));w.change('iterations',30)
config=asdict(w.scene.compile_config());document=capture_model(w)
compact=w.collapse_recipe(root)
assert compact.kind=='sa_recipe' and asdict(w.scene.compile_config())==config
w.undo_edit();assert capture_model(w)==document
w.redo_edit();decode_document(capture_model(w))
w.open_block(compact)
root=next(n for n in w.scene.nodes() if n.kind=='sa_method')
assert asdict(w.scene.compile_config())==config
compact=w.collapse_recipe(root)
w.scene.clearSelection();compact.setSelected(True)
assert w.copy_blocks() and w.paste_blocks()
pasted=next(n for n in w.scene.nodes() if n.kind=='sa_recipe' and n is not compact)
assert pasted.settings==compact.settings
w.undo_edit()
with TemporaryDirectory() as directory:
    path=Path(directory)/'sa.margam.json'
    assert w.save_model(path) and w.load_model(path)
    assert asdict(w.scene.compile_config())==config
    assert w.start_training()
    deadline=time.monotonic()+30
    while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
    assert w.worker is None and w.result is not None,w.log.toPlainText()
    assert w.result['steps']==30 and not w.output_views.entries
    summary=w.adaptive_summary()
    assert set(summary)=={'pde','left','right'} and summary['pde']['maximum']>.75
    run=Path(directory)/'sa.margam-run.pt'
    assert w.save_current_run(run)
    archived=load_run(run)
    assert 'adaptive' in archived['checkpoint']
    assert w.open_saved_run(run)
    assert w.adaptive_summary()==summary and not w.output_views.entries
    assert w.resume_training(20)
    deadline=time.monotonic()+30
    while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
    assert w.worker is None and w.result is not None and w.result['steps']==50,w.log.toPlainText()
    assert w.adaptive_summary()['pde']['maximum']>summary['pde']['maximum']
    compact=next(n for n in w.scene.nodes() if n.kind=='sa_recipe')
    with patch.object(QMessageBox,'information') as dialog:
        menu=w.create_block_menu(compact)
        next(a for a in menu.actions() if a.text()=='Adaptive weight summary').trigger()
        dialog.assert_called_once()
app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=80)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
