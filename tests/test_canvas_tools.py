"""Concrete note persistence, blank drafts and presentation-only interaction tools."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class CanvasToolsTests(unittest.TestCase):
    def run_ui(self,code):
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=120,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertNotIn('Traceback',result.stderr)

    def test_notes_are_presentation_only_and_portable(self):
        self.run_ui(r'''
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from PySide6.QtWidgets import QApplication,QPlainTextEdit
from margam.ui.app import Window
from margam.ui.model_document import capture_model,decode_document
app=QApplication([]);w=Window(state_directory=False)
def wait():
 while w.worker is not None:app.processEvents()
try:
 w.set_application_device('cpu');w.load_heat_template();w.change('iterations',2);w.change('layers',(4,))
 assert w.start_training();wait();result=w.result;model=w.last_run.model
 note=w.add_node('Note');assert note is not None and w.result is result and w.last_run.model is model
 text='Assumptions\nUnits: m, s\nu = 0 on the wall\n<b>plain text</b>\nmargam-owned: is a URI, not a file here.'
 assert w.change_note(note,text);assert w.result is result and w.last_run.model is model
 w.undo_edit();assert note.settings['annotation_text']=='Write a note…' and w.result is result
 w.redo_edit();assert note.settings['annotation_text']==text
 assert not note.inputs and note.output is None and note.detail==text and '&lt;b&gt;' in note.toolTip()
 assert w.validate_model()
 # Annotation editing is bounded and coalesces per editor without touching weights.
 w.scene.clearSelection();note.setSelected(True)
 edit=next(e for e in w.property_scroll.widget().findChildren(QPlainTextEdit) if e.property('search_key')=='annotation_text')
 index=w.undo_stack.index();edit.setPlainText('A');edit.setPlainText('AB');assert w.undo_stack.index()==index+1
 w.undo_edit();assert note.settings['annotation_text']==text and w.result is result
 assert w.change_note(note,text+'\nSaved note')
 edit=next(e for e in w.property_scroll.widget().findChildren(QPlainTextEdit) if e.property('search_key')=='annotation_text')
 assert edit.toPlainText()==note.settings['annotation_text']
 original=note.settings['annotation_text'];edit.setPlainText('q'*5000)
 assert len(edit.toPlainText())==4096 and len(note.settings['annotation_text'])==4096
 w.undo_edit();assert note.settings['annotation_text']==original
 assert not w.change_note(note,'\x00') and not w.change_note(note,'x'*4097)
 # Note-only copy/paste/cut/delete preserves predictions, including undo/redo.
 assert w.copy_blocks() and w.paste_blocks()
 notes=[n for n in w.scene.nodes() if n.kind=='annotation'];assert len(notes)==2 and len({n.identifier for n in notes})==2
 assert all(n.settings['annotation_text']==note.settings['annotation_text'] for n in notes) and w.result is result
 assert w.cut_blocks() and len([n for n in w.scene.nodes() if n.kind=='annotation'])==1 and w.result is result
 w.undo_edit();assert len([n for n in w.scene.nodes() if n.kind=='annotation'])==2 and w.result is result
 doc=capture_model(w);assert doc['version']==31;decode_document(doc)
 legacy=deepcopy(doc);legacy['version']=28;legacy['nodes']=[n for n in legacy['nodes'] if n['kind']!='annotation'];decode_document(legacy)
 bad=deepcopy(doc);next(n for n in bad['nodes'] if n['kind']=='annotation')['settings']['annotation_text']='x'*4097
 try:decode_document(bad);assert False
 except ValueError:pass
 bad=deepcopy(doc);bad['version']=28
 try:decode_document(bad);assert False
 except ValueError:pass
 with TemporaryDirectory() as directory:
  directory=Path(directory);path=directory/'noted.margam.json';assert w.save_model(path)
  run=directory/'noted.pt';assert w.save_current_run(run)
  assert w.load_model(path);assert len([n for n in w.scene.nodes() if n.kind=='annotation'])==2
  assert w.open_saved_run(run);assert len([n for n in w.scene.nodes() if n.kind=='annotation'])==2 and w.result is not None
finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
''')

    def test_blank_drafts_and_local_preferences(self):
        self.run_ui(r'''
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.model_document import capture_model
app=QApplication([])
with TemporaryDirectory() as folder:
 root=Path(folder);w=Window(state_directory=root/'preferences')
 try:
  w.change('points',81);before=capture_model(w)
  with patch.object(w,'confirm_discard',return_value=False):assert not w.new_blank_model()
  assert capture_model(w)==before
  w.set_busy(True);assert not w.new_blank_model() and not w.new_model_action.isEnabled();w.set_busy(False)
  with patch.object(w,'confirm_discard',return_value=True):assert w.new_blank_model()
  assert not w.scene.nodes() and not w.scene.wires and w.model_path is None and w.undo_stack.isClean()
  assert w.new_model_action.shortcut().toString()=='Ctrl+N' and not w.load_action.shortcut().toString()
  dialog=w.show_run_summary();assert not dialog.train.isEnabled();dialog.close()
  assert not w.validate_model()
  # Empty and notes-only drafts remain saveable; they never become executable.
  assert w.save_model(root/'empty.margam.json')
  note=w.add_node('Note');w.change_note(note,'Plan a new physics problem')
  assert not w.validate_model() and w.save_model(root/'notes.margam.json')
  assert w.load_model(root/'empty.margam.json') and not w.scene.nodes()
  before=capture_model(w);index=w.undo_stack.index()
  w.show_grid_action.setChecked(True);w.snap_grid_action.setChecked(True);w.set_dependency_mode('both')
  assert capture_model(w)==before and w.undo_stack.index()==index
 finally:w.undo_stack.setClean();w.close()
 other=Window(state_directory=root/'preferences')
 try:
  assert other.scene.show_grid and other.scene.snap_grid and other.dependency_mode=='both'
  other.scene.clearSelection();assert not any(getattr(n,'dependency_highlight',False) for n in other.scene.nodes())
 finally:other.undo_stack.setClean();other.close()
app.clipboard().clear();app.processEvents()
''')

    def test_group_drag_context_and_dependency_paths(self):
        self.run_ui(r'''
from copy import deepcopy
from unittest.mock import patch
from PySide6.QtCore import Qt,QPoint,QPointF,QEvent,QTimer
from PySide6.QtWidgets import QApplication,QGraphicsSceneContextMenuEvent,QMenu
from PySide6.QtTest import QTest
from margam.ui.app import Window
from margam.ui.model_document import capture_model
from margam.ui.canvas_tools import dependency_paths
app=QApplication([]);w=Window(state_directory=False);w.show()
for _ in range(4):app.processEvents()
try:
 nodes=w.scene.nodes();source=next(n for n in nodes if n.kind=='domain');pde=next(n for n in nodes if n.kind=='pde');boundary=next(n for n in nodes if n.kind=='boundary')
 w.scene.clearSelection();pde.setSelected(True)
 before=capture_model(w);history=w.undo_stack.index();sentinel={'keep':True};w.result=sentinel
 w.set_dependency_mode('upstream');assert source.dependency_highlight and boundary.dependency_highlight
 assert not next(n for n in nodes if n.kind=='method').dependency_highlight
 w.set_dependency_mode('downstream');assert not source.dependency_highlight and next(n for n in nodes if n.kind=='method').dependency_highlight
 assert capture_model(w)==before and w.undo_stack.index()==history and w.result is sentinel
 found,edges=dependency_paths({'a'},[('a','b'),('b','a'),('b','c'),('d','e')],'both')
 assert found=={'a','b','c'} and edges=={0,1,2}
 w.set_dependency_mode('off');assert not any(n.dependency_highlight for n in nodes)
 # Snap a native drag as one undoable group translation, preserving offsets.
 w.scene.clearSelection();source.setSelected(True);boundary.setSelected(True)
 source.setPos(13,171);boundary.setPos(243,281);offset=boundary.pos()-source.pos()
 originals={n.identifier:QPointF(n.pos()) for n in (source,boundary)}
 w.snap_grid_action.setChecked(True);w.show_grid_action.setChecked(True)
 view=w.view;view.actual_size();view.centerOn(source)
 start=view.mapFromScene(source.pos()+QPointF(50,20));end=start+QPoint(53,37)
 count=w.undo_stack.index()
 QTest.mousePress(view.viewport(),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start)
 QTest.mouseMove(view.viewport(),end,20);QTest.mouseRelease(view.viewport(),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,end);app.processEvents()
 assert abs(source.pos().x()%20)<1e-8 and abs(source.pos().y()%20)<1e-8
 assert boundary.pos()-source.pos()==offset and w.undo_stack.index()==count+1 and w.result is sentinel
 w.undo_edit();assert all(n.pos()==originals[n.identifier] for n in (source,boundary));w.redo_edit()
 # Right-press on a selected node must not destroy the group.
 start=view.mapFromScene(source.pos()+QPointF(50,20))
 QTest.mousePress(view.viewport(),Qt.MouseButton.RightButton,Qt.KeyboardModifier.NoModifier,start)
 QTest.mouseRelease(view.viewport(),Qt.MouseButton.RightButton,Qt.KeyboardModifier.NoModifier,start)
 assert source.isSelected() and boundary.isSelected()
 from PySide6.QtGui import QContextMenuEvent
 captured=[]
 def dismiss():
  popup=app.activePopupWidget();assert isinstance(popup,QMenu)
  captured.extend(a.text() for a in popup.actions());popup.close()
 QTimer.singleShot(0,dismiss)
 app.sendEvent(view.viewport(),QContextMenuEvent(QContextMenuEvent.Reason.Mouse,start,view.viewport().mapToGlobal(start)))
 assert 'Align' in captured and 'Distribute' in captured and source.isSelected() and boundary.isSelected()
 menu=w.create_group_menu();assert any(a.text()=='Align' for a in menu.actions())
 next(a for a in menu.actions() if a.text()=='Align').menu().actions()[0].trigger()
 assert source.pos().x()==boundary.pos().x() and w.result is sentinel
 assert source.isSelected() and boundary.isSelected()
 w.set_busy(True);menu=w.create_group_menu()
 assert not w.layout_actions['left'].isEnabled() and w.fit_selection_action.isEnabled()
 w.set_busy(False);assert not view.grab().isNull()
finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
''')
