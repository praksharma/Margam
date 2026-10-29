"""Affine feature options stay in the replaceable network and persist through recipes."""
import os,subprocess,sys,unittest
from pathlib import Path


class AffineFourierUITests(unittest.TestCase):
    def test_edit_mapping_archive_and_recipe_roundtrip(self):
        code=r"""from tempfile import TemporaryDirectory
from pathlib import Path
from PySide6.QtWidgets import QApplication,QLineEdit
from margam.ui.app import Window
from margam.ui.model_document import capture_model,decode_document
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
try:
 w.load_example('scale_cavity_3200');network=next(n for n in w.scene.nodes() if n.kind=='network');w.show_block_properties(network)
 edit=w.property_scroll.findChild(QLineEdit,'network_option_input_transform');assert edit and edit.text()=='x; y; x-1; y-1'
 edit.setText('x; y; x-2; y-2');edit.editingFinished.emit();assert w.scene.compile_config().network_options['input_transform']=='x; y; x-2; y-2'
 assert w.validate_model();doc=capture_model(w);decode_document(doc)
 with TemporaryDirectory() as directory:
  path=Path(directory)/'affine.margam.json';assert w.save_model(path);assert w.load_model(path);assert capture_model(w)==doc
  trainer=next(n for n in w.scene.nodes() if n.kind=='scale_method');compact=w.collapse_recipe(trainer);assert compact
  assert compact.settings['parameters']['network_type']=='affine_fourier'
  assert compact.settings['parameters']['network_options']['input_transform']=='x; y; x-2; y-2'
  w.expand_recipe(compact);assert w.scene.compile_config().network_options['input_transform']=='x; y; x-2; y-2'
finally:app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=40)
        self.assertEqual(result.returncode,0,result.stderr)
