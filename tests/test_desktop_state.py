"""Layout persistence and locked draft recovery never reopen numerical views."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class DesktopStateTests(unittest.TestCase):
    def test_layout_and_recovery(self):
        code=r'''
from tempfile import TemporaryDirectory
from unittest.mock import patch
from PySide6.QtWidgets import QApplication,QMessageBox
from margam.ui.app import Window
app=QApplication([])
with TemporaryDirectory() as directory:
    w=Window(state_directory=directory);w.show();app.processEvents()
    w.library.tree.topLevelItem(0).setExpanded(False)
    w.properties_dock.hide();w.change('points',81)
    owned=w.project_workspace.directory
    (owned/'domain.jorvik').write_bytes(b'owned fixture')
    state=w.desktop_state
    assert state.autosave() and state.path.exists()
    other=Window(state_directory=directory)
    with patch.object(QMessageBox,'question') as question:
        assert not other.desktop_state.offer_recovery();question.assert_not_called()
    other.undo_stack.setClean();other.close()
    # Simulate a process ending before its clean-close cleanup.
    state.timer.stop();state.lock.unlock();state.lock=None
    with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes):
        assert other.desktop_state.offer_recovery()
    assert (other.project_workspace.directory/'domain.jorvik').read_bytes()==b'owned fixture'
    assert other.config.points==81 and other.isWindowModified()
    assert not other.output_views.entries and other.model_path is None
    other.undo_stack.setClean();other.desktop_state.clear()
    w.undo_stack.setClean();w.close();app.processEvents()
    restored=Window(state_directory=directory);restored.show();app.processEvents()
    assert not restored.library.tree.topLevelItem(0).isExpanded()
    assert restored.properties_dock.isHidden() and not restored.output_views.entries
    restored.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
