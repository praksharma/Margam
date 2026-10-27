"""Navigation changes the camera, never block positions or numerical settings."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class CanvasNavigationTests(unittest.TestCase):
    def test_zoom_pan_and_fit(self):
        code=r'''
from PySide6.QtCore import Qt, QPoint, QPointF
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from margam.ui.app import Window
app=QApplication([]);w=Window(state_directory=False);w.show()
for _ in range(3):app.processEvents()
view=w.view
positions={n.identifier:n.pos() for n in w.scene.nodes()}
point=QPoint(210,160)
before=view.mapToScene(point)
scale=view.transform().m11()
event=QWheelEvent(QPointF(point),QPointF(view.viewport().mapToGlobal(point)),QPoint(),QPoint(0,120),Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.NoScrollPhase,False)
app.sendEvent(view.viewport(),event)
assert view.transform().m11()>scale
assert (view.mapToScene(point)-before).manhattanLength()<3/view.transform().m11()
w.actual_size_action.trigger()
assert abs(view.transform().m11()-1)<1e-10
w.zoom_in_action.trigger();assert view.transform().m11()>1
w.zoom_out_action.trigger();assert abs(view.transform().m11()-1)<1e-10
for key in [Qt.MouseButton.MiddleButton,Qt.MouseButton.LeftButton]:
    view.setFocus()
    if key==Qt.MouseButton.LeftButton:QTest.keyPress(view,Qt.Key.Key_Space)
    start=view.viewport().rect().center();end=start+QPoint(90,50)
    center_before=view.mapToScene(start)
    QTest.mousePress(view.viewport(),key,Qt.KeyboardModifier.NoModifier,start)
    QTest.mouseMove(view.viewport(),end,20)
    QTest.mouseRelease(view.viewport(),key,Qt.KeyboardModifier.NoModifier,end)
    if key==Qt.MouseButton.LeftButton:QTest.keyRelease(view,Qt.Key.Key_Space)
    assert (view.mapToScene(start)-center_before).manhattanLength()>50
assert {n.identifier:n.pos() for n in w.scene.nodes()}==positions
view.setFocus();QTest.keyClick(view,Qt.Key.Key_Space)
assert view.viewport().rect().contains(view.mapFromScene(w.scene.itemsBoundingRect()).boundingRect())
for _ in range(30):view.zoom_by(2)
assert abs(view.transform().m11()-4)<1e-10
for _ in range(30):view.zoom_by(.5)
assert abs(view.transform().m11()-.1)<1e-10
w.fit_model();w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],
                              env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),
                              capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertNotIn('Traceback',result.stderr)
