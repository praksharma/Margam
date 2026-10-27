"""Pan mode changes the cursor/camera, including over blocks, without model edits."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class PanModeTests(unittest.TestCase):
    def test_hand_cursor_drag_and_exit(self):
        code=r'''
from PySide6.QtCore import Qt,QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
v=w.view;positions={n.identifier:n.pos() for n in w.scene.nodes()};history=w.undo_stack.index()
w.pan_action.trigger()
assert v.pan_mode and v.viewport().cursor().shape()==Qt.CursorShape.OpenHandCursor
node=next(n for n in w.scene.nodes() if n.kind=='domain')
point=v.mapFromScene(node.mapToScene(node.rect().center()));before=v.mapToScene(point)
QTest.mousePress(v.viewport(),Qt.MouseButton.LeftButton,pos=point)
assert v.viewport().cursor().shape()==Qt.CursorShape.ClosedHandCursor
QTest.mouseMove(v.viewport(),point+QPoint(80,40),20)
QTest.mouseRelease(v.viewport(),Qt.MouseButton.LeftButton,pos=point+QPoint(80,40))
assert v.viewport().cursor().shape()==Qt.CursorShape.OpenHandCursor
assert (v.mapToScene(point)-before).manhattanLength()>20
assert {n.identifier:n.pos() for n in w.scene.nodes()}==positions and w.undo_stack.index()==history
QTest.mouseDClick(v.viewport(),Qt.MouseButton.LeftButton,pos=QPoint(50,50))
assert w.quick_insert is None
w.name.setFocus();app.processEvents()
assert v.pan_mode and v.viewport().cursor().shape()==Qt.CursorShape.OpenHandCursor
v.setFocus();QTest.keyClick(v,Qt.Key.Key_Escape)
assert not v.pan_mode and not w.pan_action.isChecked()
assert v.viewport().cursor().shape()==Qt.CursorShape.ArrowCursor
w.pan_action.trigger();w.pan_action.trigger()
assert not v.pan_mode
w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
