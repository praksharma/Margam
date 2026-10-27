"""Check package isolation and desktop startup/teardown in a separate process."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class DesktopLifecycleTests(unittest.TestCase):
    def test_startup_selection_theme_and_clean_shutdown(self):
        code = """
import sys
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window, Node
from margam.ui.theme import configure_theme, colors
assert 'torch' not in sys.modules, 'UI import should not load the numerical engine'
app = QApplication([])
w = Window(state_directory=False)
w.show()
for _ in range(3):app.processEvents()
assert w.view.viewport().rect().contains(w.view.mapFromScene(w.scene.itemsBoundingRect()).boundingRect())
assert len(w.scene.wires) == 7
assert not w.output_views.entries
assert w.messages_dock.isHidden()
assert not w.rectangle_previews
assert [a.text() for a in w.menuBar().actions()] == ["File", "Edit", "View", "Run", "Help"]
node = next(i for i in w.scene.items() if isinstance(i, Node))
w.scene.clearSelection()
node.setSelected(True)
app.processEvents()
assert w.name.text() == node.title
for theme in ('dark', 'light'):
    configure_theme(app, theme)
    app.processEvents()
    assert w.scene.backgroundBrush().color().name() == colors()['canvas']
    assert not w.grab().isNull()
edit=next(a.menu() for a in w.menuBar().actions() if a.text()=='Edit')
next(a for a in edit.actions() if a.text()=='Select all blocks').trigger()
app.processEvents()
assert all(n.isSelected() for n in w.scene.nodes())
assert w.name.text()=='Multiple blocks selected'
action=w.library_dock.toggleViewAction()
action.trigger();app.processEvents()
assert w.library_dock.isHidden()
action.trigger();app.processEvents()
assert not w.library_dock.isHidden()
w.undo_stack.setClean();w.close()
app.processEvents()
"""
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-c", code],
                                cwd=Path(__file__).resolve().parents[1], env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == '__main__':
    unittest.main()
