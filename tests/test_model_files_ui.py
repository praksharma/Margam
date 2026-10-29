"""Save/open identity, dirty history and Save/Discard/Cancel lifecycle."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class ModelFileUITests(unittest.TestCase):
    def test_model_files_and_unsaved_choices(self):
        code=r'''
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from PySide6.QtWidgets import QApplication, QMessageBox
from margam.ui.app import Window
from margam.ui.model_document import capture_model
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
B=QMessageBox.StandardButton
with TemporaryDirectory() as directory:
    path=Path(directory)/'heat.margam.json'
    w.load_heat_template()
    w.change_boundary_sampling('left','method','random')
    w.change_boundary_sampling('left','points',31)
    monitor=next(n for n in w.scene.nodes() if n.kind=='monitor')
    w.rename_output(monitor,'My losses');w.change_output(monitor,'history_limit',150)
    monitor.setPos(1250,-100)
    temporary=w.project_workspace.directory
    (temporary/"domain.jorvik").write_bytes(b"fixture")
    expected=capture_model(w)
    assert w.isWindowModified() and w.save_model(path)
    assert not temporary.exists()
    assert (w.project_workspace.directory/"domain.jorvik").read_bytes()==b"fixture"
    assert not w.isWindowModified() and w.model_path==str(path)
    w.change('source',3)
    assert w.isWindowModified()
    w.undo_edit();assert not w.isWindowModified()
    w.redo_edit();assert w.isWindowModified()
    # Rejecting open/close/reset preserves the complete current edit state.
    current=capture_model(w);index=w.undo_stack.index()
    with patch.object(QMessageBox,'question',return_value=B.Cancel):
        assert not w.load_model(path)
        assert not w.request_template()
        assert not w.close() and w.isVisible()
    assert capture_model(w)==current and w.undo_stack.index()==index
    # Choosing Save but cancelling the file picker also cancels replacement.
    w.model_path=None
    with patch.object(QMessageBox,'question',return_value=B.Save),patch('margam.ui.app.QFileDialog.getSaveFileName',return_value=('','')):
        assert not w.request_template()
    assert capture_model(w)==current
    # Bad files are rejected before any current data/history/filename is discarded.
    bad=Path(directory)/'bad.json';bad.write_text('{"format":"unknown"}')
    with patch.object(QMessageBox,'warning') as warning,patch.object(QMessageBox,'question') as question:
        assert not w.load_model(bad)
        warning.assert_called_once();question.assert_not_called()
    assert capture_model(w)==current and w.undo_stack.index()==index
    # Failed save does not mark the model clean or change the chosen document.
    with patch('margam.ui.app.write_model',side_effect=OSError('disk failure')),patch.object(QMessageBox,'warning'):
        assert not w.save_model(path)
    assert w.isWindowModified() and w.model_path is None
    # Discard permits opening; IDs, names, recipes, positions, ports and settings round-trip.
    w.open_block(monitor);app.processEvents()
    with patch.object(QMessageBox,'question',return_value=B.Discard):assert w.load_model(path)
    assert capture_model(w)==expected
    assert not w.isWindowModified() and w.undo_stack.count()==0 and w.result is None
    assert not w.output_views.entries and not w.rectangle_previews
    assert w.validate_model()
    w.change('iterations',25)
    assert w.start_training()
    deadline=time.monotonic()+20
    while w.worker is not None and time.monotonic()<deadline:
        app.processEvents();time.sleep(.005)
    assert w.worker is None and w.result is not None and w.result['steps']==25
    assert len(w.result['outputs'])==2 and not w.output_views.entries
    assert len(w.monitor_records[monitor.identifier])==25
    # Reopening the current file with Save uses its newly saved content, not
    # the older document inspected before the confirmation.
    w.change('source',6)
    with patch.object(QMessageBox,'question',return_value=B.Save):assert w.load_model(path)
    assert w.config.source==6 and not w.isWindowModified()
    # Save confirmation actually writes edits before replacing the current model.
    w.change('source',4)
    with patch.object(QMessageBox,'question',return_value=B.Save):assert w.request_template()
    assert not w.isWindowModified() and w.model_path is None
    assert w.load_model(path) and w.config.source==4
    # Save As adds the standard suffix when omitted and updates the window name.
    with patch('margam.ui.app.QFileDialog.getSaveFileName',return_value=(str(Path(directory)/'copy'),'')):
        assert w.save_model_as()
    assert w.model_path.endswith('copy.margam.json') and 'copy.margam.json' in w.windowTitle()
    original=(Path(w.model_path)).read_bytes()
    w.change('source',5)
    with patch('margam.ui.app.QFileDialog.getSaveFileName',return_value=(str(Path(directory)/'copy'),'')),patch.object(QMessageBox,'question',return_value=B.No):
        assert not w.save_model_as()
    assert Path(w.model_path).read_bytes()==original and w.isWindowModified()
    w.set_busy(True)
    assert not w.open_model_action.isEnabled() and not w.save_model_action.isEnabled()
    assert not w.load_model(path) and not w.save_model(path)
    w.set_busy(False)
    # Incomplete graphs can be saved and reopened as drafts, then fail Run validation.
    network=next(n for n in w.scene.nodes() if n.kind=='network')
    w.scene.clearSelection();network.setSelected(True);w.delete_selected()
    draft_path=Path(directory)/'draft.margam.json'
    assert w.save_model(draft_path)
    assert w.load_model(draft_path) and not w.validate_model()
    w.undo_stack.setClean();assert w.close()
app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],
            env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),
            capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_save_commits_pending_numeric_editor(self):
        code=r'''
from pathlib import Path
from tempfile import TemporaryDirectory
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QSpinBox
from margam.ui.app import Window
from margam.ui.model_document import read_model
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
node=next(n for n in w.scene.nodes() if n.kind=='interior')
w.show_block_properties(node)
spin=w.property_scroll.widget().findChild(QSpinBox)
spin.setFocus();spin.lineEdit().selectAll();QTest.keyClicks(spin.lineEdit(),'121')
assert w.config.points==64
with TemporaryDirectory() as directory:
    path=Path(directory)/'pending.margam.json'
    assert w.save_model(path)
    assert w.config.points==121 and read_model(path)['config']['points']==121
    assert not w.isWindowModified()
w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],
            env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),
            capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
