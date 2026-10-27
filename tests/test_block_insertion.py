"""Native drop events and quick insert share placement, validation and locking."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class BlockInsertionTests(unittest.TestCase):
    def run_ui(self, code):
        result = subprocess.run([sys.executable, '-c', code],
            cwd=Path(__file__).resolve().parents[1],
            env=dict(os.environ, QT_QPA_PLATFORM='offscreen', PYTHONDONTWRITEBYTECODE='1'),
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_drop_and_validation(self):
        self.run_ui(r'''
from PySide6.QtCore import Qt, QPoint, QPointF, QMimeData
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.block_catalogue import BLOCK_MIME
app=QApplication([]);w=Window();w.show();app.processEvents()
tree=w.library.tree
results=tree.topLevelItem(5)
mime=tree.mimeData([results.child(1)])
assert bytes(mime.data(BLOCK_MIME))==b'Training Monitor'
assert not tree.mimeData([results]).hasFormat(BLOCK_MIME)
assert not tree.mimeData([results.child(0),results.child(1)]).hasFormat(BLOCK_MIME)
view=w.view
view.zoom_by(1.7);view.translate(135,-80)
position=QPoint(155,145)
scene_position=view.mapToScene(position)
old_positions={n.identifier:n.pos() for n in w.scene.nodes()}
old_zoom=view.transform().m11()
def send_drop(data):
    enter=QDragEnterEvent(position,Qt.DropAction.CopyAction,data,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
    app.sendEvent(view.viewport(),enter)
    move=QDragMoveEvent(position,Qt.DropAction.CopyAction,data,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
    app.sendEvent(view.viewport(),move)
    drop=QDropEvent(QPointF(position),Qt.DropAction.CopyAction,data,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
    app.sendEvent(view.viewport(),drop)
    return enter,move,drop
assert all(event.isAccepted() for event in send_drop(mime))
node=next(n for n in w.scene.nodes() if n.kind=='monitor')
assert (node.mapToScene(node.rect().center())-scene_position).manhattanLength()<1e-8
assert all(n.pos()==old_positions[n.identifier] for n in w.scene.nodes() if n.identifier in old_positions)
assert view.transform().m11()==old_zoom
assert len(w.scene.wires)==7 and not w.output_views.entries
assert w.validate_model()  # An unconnected optional output is allowed.
count=len(w.scene.nodes())
for payload in (b'not a block',b'\xff',b'x'*101):
    bad=QMimeData();bad.setData(BLOCK_MIME,payload)
    assert not any(event.isAccepted() for event in send_drop(bad))
text=QMimeData();text.setText('Rectangle')
assert not any(event.isAccepted() for event in send_drop(text))
w.set_busy(True)
assert not any(event.isAccepted() for event in send_drop(mime))
assert w.add_node('Training Monitor',scene_position) is None
assert len(w.scene.nodes())==count
w.set_busy(False)
# Same existing validation for duplicate core blocks, whichever insertion path.
core=tree.mimeData([tree.topLevelItem(0).child(0)])
assert all(event.isAccepted() for event in send_drop(core))
assert not w.validate_model()
assert 'exactly one domain' in w.validation.text()
assert w.add_node('unknown title') is None
w.undo_stack.setClean();w.close();app.processEvents()
''')

    def test_quick_insert_keyboard_and_lifecycle(self):
        self.run_ui(r'''
from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
app=QApplication([]);w=Window();w.show();app.processEvents()
view=w.view;position=QPoint(120,120)
assert view.itemAt(position) is None
expected=view.mapToScene(position)
count=len(w.scene.nodes())
QTest.mouseClick(view.viewport(),Qt.MouseButton.LeftButton,pos=position)
QTest.mouseDClick(view.viewport(),Qt.MouseButton.LeftButton,pos=position)
app.processEvents()
popup=w.quick_insert
assert popup is not None and popup.isVisible()
popup.library.search.setText('training mon')
QTest.keyClick(popup.library.search,Qt.Key.Key_Return);app.processEvents()
assert w.quick_insert is None and len(w.scene.nodes())==count+1
node=next(n for n in w.scene.nodes() if n.kind=='monitor')
assert (node.mapToScene(node.rect().center())-expected).manhattanLength()<1e-8
assert not w.output_views.entries
view.setFocus();QTest.keyClick(view,Qt.Key.Key_Tab);app.processEvents()
assert w.quick_insert is not None
popup=w.quick_insert;popup.library.search.setText('no match at all')
QTest.keyClick(popup.library.search,Qt.Key.Key_Return)
assert w.quick_insert is popup and len(w.scene.nodes())==count+1
QTest.keyClick(popup.library.search,Qt.Key.Key_Escape);app.processEvents()
assert w.quick_insert is None
# Navigate from search into the matching tree and insert with Enter.
expected=view.mapToScene(view.viewport().rect().center())
view.setFocus();QTest.keyClick(view,Qt.Key.Key_Tab);app.processEvents()
popup=w.quick_insert;popup.library.search.setText('reference')
QTest.keyClick(popup.library.search,Qt.Key.Key_Down)
QTest.keyClick(popup.library.tree,Qt.Key.Key_Return);app.processEvents()
assert w.quick_insert is None
node=next(n for n in w.scene.nodes() if n.kind=='reference')
assert (node.mapToScene(node.rect().center())-expected).manhattanLength()<1e-8
# Tab in the inspector must retain ordinary focus navigation.
w.name.setFocus();QTest.keyClick(w.name,Qt.Key.Key_Tab);app.processEvents()
assert w.quick_insert is None
view.setFocus();QTest.keyClick(view,Qt.Key.Key_Tab);app.processEvents()
assert w.quick_insert is not None
w.set_busy(True);app.processEvents()
assert w.quick_insert is None
QTest.keyClick(view,Qt.Key.Key_Tab);app.processEvents()
assert w.quick_insert is None
w.set_busy(False)
# A block double-click retains its existing view action.
view.centerOn(node);app.processEvents()
point=view.mapFromScene(node.mapToScene(node.rect().center()))
QTest.mouseClick(view.viewport(),Qt.MouseButton.LeftButton,pos=point)
QTest.mouseDClick(view.viewport(),Qt.MouseButton.LeftButton,pos=point)
app.processEvents()
assert w.quick_insert is None
# Template reset dismisses a pending insertion without creating a block.
w.show_quick_insert(expected,view.viewport().mapToGlobal(position))
w.reset_template();app.processEvents()
assert w.quick_insert is None and len(w.scene.nodes())==7
w.undo_stack.setClean();w.close();app.processEvents()
''')
