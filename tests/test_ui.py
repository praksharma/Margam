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
w = Window()
w.show()
app.processEvents()
assert len(w.scene.wires) == 8
node = next(i for i in w.scene.items() if isinstance(i, Node))
node.setSelected(True)
app.processEvents()
assert w.name.text() == node.title
for theme in ('dark', 'light'):
    configure_theme(app, theme)
    app.processEvents()
    assert w.scene.backgroundBrush().color().name() == colors()['canvas']
    assert not w.grab().isNull()
w.close()
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
