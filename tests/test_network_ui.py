"""Architecture settings across UI edits, recipes, old models and run archives."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class NetworkUITests(unittest.TestCase):
    def test_architecture_options_recipes_and_archives(self):
        code=r'''
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from PySide6.QtWidgets import QApplication,QComboBox,QMessageBox
from margam.ui.app import Window
from margam.ui.model_document import capture_model
from margam.ui.run_document import load_run
from margam.components.network_config import NETWORK_NAMES
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
def warning(*args):raise AssertionError(str(args[1:]))
QMessageBox.warning=warning
w.load_example('interval_poisson')
initial=capture_model(w)
# Earlier model definitions and compact recipes receive the new FCNN defaults.
for compact_mode in (False,True):
    if compact_mode:
        root=next(n for n in w.scene.nodes() if n.kind=='method');w.collapse_recipe(root)
    old=capture_model(w);old['version']=9
    for key in ('network_type','network_options'):old['config'].pop(key)
    for n in old['nodes']:
        if n['kind'] in ('network','recipe','sa_recipe'):
            for key in ('network_type','network_options'):n['settings']['parameters'].pop(key)
    assert w.replace_model(old)
    assert w.scene.compile_config().network_type=='fcnn'
    if compact_mode:w.expand_recipe(next(n for n in w.scene.nodes() if n.kind=='recipe'))
w.library.search.setText('siren');app.processEvents()
assert not w.library.tree.topLevelItem(2).isHidden()
w.library.search.clear()
with TemporaryDirectory() as directory:
    for kind in NETWORK_NAMES:
        w.load_example('self_adaptive_poisson')
        network=next(n for n in w.scene.nodes() if n.kind=='network')
        w.scene.clearSelection();network.setSelected(True)
        combo=w.property_scroll.findChild(QComboBox,'network_architecture')
        combo.setCurrentIndex(combo.findData(kind));app.processEvents()
        assert w.scene.compile_config().network_type==kind
        if kind=='siren':w.change_network_option('first_omega',10.)
        config=asdict(w.scene.compile_config())
        root=next(n for n in w.scene.nodes() if n.kind=='sa_method')
        compact=w.collapse_recipe(root)
        assert asdict(w.scene.compile_config())==config
        root=w.expand_recipe(compact);assert asdict(w.scene.compile_config())==config
        compact=w.collapse_recipe(root)
        path=Path(directory)/(kind+'.margam.json')
        assert w.save_model(path) and w.load_model(path)
        assert asdict(w.scene.compile_config())==config
        compact=next(n for n in w.scene.nodes() if n.kind=='sa_recipe')
        w.scene.clearSelection();compact.setSelected(True)
        assert w.copy_blocks() and w.paste_blocks()
        pasted=next(n for n in w.scene.nodes() if n.kind=='sa_recipe' and n is not compact)
        assert pasted.settings==compact.settings
        w.undo_edit()
        w.change('iterations',5);assert w.start_training()
        deadline=time.monotonic()+30
        while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.002)
        assert w.worker is None and w.result is not None,w.log.toPlainText()
        assert w.result['steps']==5 and not w.output_views.entries
        run=Path(directory)/(kind+'.margam-run.pt')
        assert w.save_current_run(run)
        payload=load_run(run)
        assert payload['checkpoint']['network_spec']['type']==kind
        w.undo_stack.setClean();assert w.open_saved_run(run)
        assert w.scene.compile_config().network_type==kind
        assert w.resume_training(2)
        deadline=time.monotonic()+30
        while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.002)
        assert w.worker is None and w.result is not None and w.result['steps']==7,w.log.toPlainText()
# Switching architecture is atomic and undo restores the earlier option dictionary.
w.load_example('interval_poisson')
network=next(n for n in w.scene.nodes() if n.kind=='network');w.scene.clearSelection();network.setSelected(True)
w.change_network_type('siren');w.change_network_option('first_omega',12.)
before=capture_model(w);w.change_network_type('dgm');w.undo_edit()
assert capture_model(w)==before
# A partial draft remains saveable while its layer editor is empty.
w.change('layers',());capture_model(w)
app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=110)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
