"""Concrete crowded-diagram detours and presentation-only routing lifecycle."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
from margam.ui.wire_routing import route,crosses,MAX_BLOCKS


class RoutingTests(unittest.TestCase):
    def test_clear_route_and_note_detour_are_deterministic(self):
        boxes=[(0,0,170,80),(600,0,770,80)]
        self.assertEqual(route((170,40),(600,40),boxes,0,1),[(170,40),(600,40)])
        boxes.append((300,-20,470,100))
        path=route((170,40),(600,40),boxes,0,1)
        self.assertIsNotNone(path);self.assertEqual(path,route((170,40),(600,40),boxes,0,1))
        self.assertEqual(path[0],(170,40));self.assertEqual(path[-1],(600,40))
        for a,b in zip(path,path[1:]):
            self.assertTrue(a[0]==b[0] or a[1]==b[1]);self.assertFalse(crosses(a,b,boxes[2]))

    def test_backward_and_vertical_connections_avoid_endpoint_bodies(self):
        for start,end,boxes in [((770,40),(0,40),[(600,0,770,80),(0,0,170,80)]),
                                ((170,40),(0,300),[(0,0,170,80),(0,260,170,340)])]:
            path=route(start,end,boxes,0,1);self.assertIsNotNone(path)
            for a,b in zip(path,path[1:]):
                for rect in boxes:self.assertFalse(crosses(a,b,rect))

    def test_overlaps_invalid_coordinates_and_bounded_fallback(self):
        boxes=[(0,0,170,80),(600,0,770,80),(-10,-10,200,100)]
        self.assertIsNone(route((170,40),(600,40),boxes,0,1))
        self.assertIsNone(route((float('nan'),40),(600,40),boxes,0,1))
        self.assertIsNone(route((170,40),(600,40),boxes,10,1))
        self.assertIsNone(route((170,40),(600,40),boxes+[(0,0,1,1)]*MAX_BLOCKS,0,1))

    def run_ui(self,code):
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=120,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertNotIn('Traceback',result.stderr)

    def test_native_selection_drag_undo_and_retained_results(self):
        self.run_ui(r"""
from copy import deepcopy
from PySide6.QtCore import Qt,QPoint,QPointF
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from margam.ui.app import Window
from margam.ui.model_document import capture_model
from examples.capture_wire_routing import arrange_fixture,wire_hits
app=QApplication([]);w=Window(state_directory=False);w.resize(1600,960);w.show()
try:
 w.set_application_device('cpu');w.load_heat_template();w.change('iterations',2);w.change('layers',(4,))
 assert w.start_training()
 while w.worker is not None:app.processEvents()
 result=w.result;model=w.last_run.model;assert result is not None
 note,wire=arrange_fixture(w);before=capture_model(w);index=w.undo_stack.index()
 curve=wire.path();assert wire_hits(wire,note)
 w.route_wires_action.setChecked(True)
 assert wire.route_status=='routed' and not wire_hits(wire,note)
 assert capture_model(w)==before and w.undo_stack.index()==index
 assert w.result is result and w.last_run.model is model
 w.fit_model()
 for _ in range(4):app.processEvents()
 point=w.view.mapFromScene(wire.path().pointAtPercent(.5))
 QTest.mouseClick(w.view.viewport(),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,point);app.processEvents()
 assert wire.isSelected()
 w.scene.clearSelection();note.setSelected(True);old=QPointF(note.pos());oldpath=wire.path()
 w.view.actual_size();w.view.centerOn(note)
 start=w.view.mapFromScene(note.pos()+QPointF(50,20));end=start+QPoint(0,240)
 QTest.mousePress(w.view.viewport(),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start)
 QTest.mouseMove(w.view.viewport(),end,20)
 QTest.mouseRelease(w.view.viewport(),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,end);app.processEvents()
 assert note.pos()!=old and wire.path()!=oldpath and w.undo_stack.index()==index+1
 assert not wire_hits(wire,note) and w.result is result
 w.undo_edit();assert note.pos()==old and wire.path()==oldpath and w.result is result
 w.redo_edit();assert note.pos()!=old and w.result is result
 w.undo_edit();w.set_busy(True)
 assert w.route_wires_action.isEnabled();w.route_wires_action.setChecked(False)
 assert wire.route_status=='curved' and wire.path()==curve and w.result is result
 w.set_busy(False);w.route_wires_action.setChecked(True)
 # Notes are obstacles immediately on insertion/removal, including Undo/Redo.
 w.scene.clearSelection();note.setSelected(True);w.delete_selected()
 assert wire.route_status=='routed' and wire.path()!=oldpath and w.result is result
 w.undo_edit();assert wire.path()==oldpath and w.result is result
finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
""")

    def test_local_preference_and_fallback_tooltip(self):
        self.run_ui(r"""
from tempfile import TemporaryDirectory
from pathlib import Path
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.model_document import capture_model
from examples.capture_wire_routing import arrange_fixture,wire_hits
app=QApplication([])
with TemporaryDirectory() as folder:
 w=Window(state_directory=Path(folder)/'state')
 try:
  note,wire=arrange_fixture(w);w.route_wires_action.setChecked(True)
  # Deliberately enclose the source port: fixed stub cannot escape a different block.
  note.setPos(wire.source.scenePos().x()-100,wire.source.scenePos().y()-60)
  assert wire.route_status=='fallback' and 'Routing fallback' in wire.toolTip()
  saved=capture_model(w);w.scene.reroute_wires();assert capture_model(w)==saved
 finally:w.undo_stack.setClean();w.close()
 other=Window(state_directory=Path(folder)/'state')
 try:assert other.scene.route_wires and other.route_wires_action.isChecked()
 finally:other.undo_stack.setClean();other.close()
 isolated=Window(state_directory=False)
 try:assert not isolated.scene.route_wires
 finally:isolated.undo_stack.setClean();isolated.close()
app.clipboard().clear();app.processEvents()
""")
