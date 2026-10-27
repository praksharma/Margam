"""Copied subgraphs get fresh IDs and internal wires, with reversible paste/cut."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class ClipboardTests(unittest.TestCase):
    def test_copy_cut_paste(self):
        code=r'''
from PySide6.QtCore import QMimeData
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
w.load_heat_template()
method=next(n for n in w.scene.nodes() if n.kind=='method')
monitor=next(n for n in w.scene.nodes() if n.kind=='monitor')
w.change_output(monitor,'history_limit',80)
w.scene.clearSelection();method.setSelected(True);monitor.setSelected(True)
assert w.copy_blocks()
count=len(w.scene.nodes());edges=len(w.scene.wires);config=w.config
assert w.paste_blocks()
pasted=[n for n in w.scene.nodes() if n.isSelected()]
assert len(pasted)==2 and not {n.identifier for n in pasted}&{method.identifier,monitor.identifier}
assert len(w.scene.wires)==edges+1 and w.config is config
assert next(n for n in pasted if n.kind=='monitor').settings['history_limit']==80
assert not w.output_views.entries
w.undo_edit();assert len(w.scene.nodes())==count and len(w.scene.wires)==edges
w.redo_edit();assert all(n.scene() is w.scene for n in pasted)
assert w.cut_blocks();assert len(w.scene.nodes())==count
w.undo_edit();assert all(n.scene() is w.scene for n in pasted)
w.set_busy(True)
assert not w.paste_blocks() and not w.cut_blocks()
w.set_busy(False)
index=w.undo_stack.index()
mime=QMimeData();mime.setData('application/x-margam-model-fragment',b'{bad')
app.clipboard().setMimeData(mime)
assert not w.paste_blocks() and w.undo_stack.index()==index
app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],
            env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
