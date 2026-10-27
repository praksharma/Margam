"""Core instance settings are independent; execution assembles connected inputs."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class GraphInstanceTests(unittest.TestCase):
    def test_owned_settings_and_assembly(self):
        code=r'''
from tempfile import TemporaryDirectory
from pathlib import Path
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.graph_model import assemble
app=QApplication([]);w=Window(state_directory=False);w.show()
original=next(n for n in w.scene.nodes() if n.kind=='network')
w.show_block_properties(original);w.change('layers',(20,12));w.change('network_seed',123)
other=w.add_node('Neural network');w.change('layers',(8,8));w.change('network_seed',456)
assert original.settings['parameters']['layers']==[20,12]
assert other.settings['parameters']['layers']==[8,8]
w.show_block_properties(original)
assert w.config.layers==(20,12) and w.config.network_seed==123
with TemporaryDirectory() as directory:
    path=Path(directory)/'instances.margam.json';assert w.save_model(path) and w.load_model(path)
    networks=[n for n in w.scene.nodes() if n.kind=='network']
    assert sorted(n.settings['parameters']['network_seed'] for n in networks)==[123,456]
    other=next(n for n in networks if n.settings['parameters']['network_seed']==456)
    w.scene.clearSelection();other.setSelected(True);w.delete_selected()
    assert w.validate_model()
    config=w.scene.compile_config();assert config.layers==(20,12) and config.network_seed==123
    # The compiler's recipe comes from the connected blocks, not the inspector cache.
    w.config.layers=(99,99)
    assert w.scene.compile_config().layers==(20,12)
    domain=next(n for n in w.scene.nodes() if n.kind=='domain')
    w.show_block_properties(domain);w.change('x_max',2)
    assert w.scene.compile_config().x_max==2
w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
