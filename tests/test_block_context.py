"""Context actions and previews stay scoped to the supported block types."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class BlockContextTests(unittest.TestCase):
    def test_rectangle_preview_lifecycle_and_real_context_event(self):
        code = r'''
from PySide6.QtCore import QTimer
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtWidgets import QApplication, QMenu
from margam.ui.app import Window
from margam.ui.plots import RectanglePreview
app=QApplication([])
w=Window(state_directory=False);w.show()
for _ in range(3):app.processEvents()
assert not w.findChildren(RectanglePreview)
assert not w.rectangle_previews
assert w.messages_dock.isHidden()
owner=next(n for n in w.scene.nodes() if n.kind=='domain')
other=next(n for n in w.scene.nodes() if n.kind=='network')

def labels(node):
    return [a.text() for a in w.create_block_menu(node).actions() if not a.isSeparator()]
assert labels(owner)==['Properties','Preview geometry','Preview samples','Delete block','Copy blocks','Cut blocks','Paste blocks','Undo','Redo','Help']
assert labels(other)==['Properties','Delete block','Copy blocks','Cut blocks','Paste blocks','Undo','Redo','Help']

# Exercise Qt's actual viewport context-menu event rather than only the helper.
captured=[]
def dismiss():
    menu=app.activePopupWidget()
    assert isinstance(menu,QMenu)
    captured.extend(a.text() for a in menu.actions())
    menu.close()
w.fit_model();app.processEvents()
position=w.view.mapFromScene(owner.mapToScene(owner.rect().center()))
QTimer.singleShot(0,dismiss)
app.sendEvent(w.view.viewport(),QContextMenuEvent(QContextMenuEvent.Reason.Mouse,position,w.view.viewport().mapToGlobal(position)))
assert 'Preview samples' in captured
assert not w.rectangle_previews

# Geometry and sample views are separate and are created only when requested.
menu=w.create_block_menu(owner)
next(a for a in menu.actions() if a.text()=='Preview geometry').trigger()
app.processEvents()
geometry,geometry_view=w.rectangle_previews[(owner.identifier,False)]
assert geometry.isFloating() and not geometry_view.show_samples
assert not geometry_view.grab().isNull()
next(a for a in menu.actions() if a.text()=='Preview samples').trigger()
app.processEvents()
samples,samples_view=w.rectangle_previews[(owner.identifier,True)]
assert samples.isFloating() and samples_view.show_samples
assert len(w.rectangle_previews)==2
samples.close();app.processEvents()
assert samples.isHidden()
assert w.open_rectangle_preview(owner,True) is samples
w.change('x_max',2)
assert samples_view.config.x_max==2
assert not samples_view.grab().isNull()

# Output menus only show the action their kind supports.
for title,label in [('Training Monitor','Open monitor'),('Solution Plot','Open plot'),('Comparison','Open comparison')]:
    node=w.add_node(title)
    assert label in labels(node)
    assert 'Preview samples' not in labels(node)
assert not w.output_views.entries
monitor=next(n for n in w.scene.nodes() if n.kind=='monitor')
next(a for a in w.create_block_menu(monitor).actions() if a.text()=='Open monitor').trigger()
app.processEvents()
assert monitor.identifier in w.output_views.entries

# Changing shape removes the rectangle windows, not just their menu entries.
w.change('shape','Interval');app.processEvents()
assert not w.rectangle_previews
assert 'Preview samples' not in labels(owner)
assert w.open_rectangle_preview(owner,True) is None
w.change('shape','Rectangle')
assert not w.rectangle_previews
w.open_rectangle_preview(owner,True)
assert len(w.rectangle_previews)==1
w.properties_dock.close()
w.delete_block(owner);app.processEvents()
assert w.properties_dock.isHidden(), "Deleting a block must not open its inspector"
assert not w.rectangle_previews
assert other.scene() is w.scene
w.reset_template();app.processEvents()
owner=next(n for n in w.scene.nodes() if n.kind=='domain')
w.open_rectangle_preview(owner,False)
w.reset_template();app.processEvents()
assert not w.rectangle_previews
w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],
                              env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),
                              capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertNotIn('Traceback',result.stderr)
