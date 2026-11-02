"""Recent models remain local and preserve normal open/cancel/locking semantics."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class RecentModelsTests(unittest.TestCase):
    def test_local_history_and_safe_open(self):
        code=r'''
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from PySide6.QtWidgets import QApplication,QMessageBox
from margam.ui.app import Window
from margam.ui.model_document import capture_model
from margam.ui.recent_models import valid_paths
app=QApplication([])
assert valid_paths(None)==[] and valid_paths({'path':'bad'})==[]
assert valid_paths(['relative',42,'/tmp/a','/tmp/a','\x00'])==['/tmp/a']
with TemporaryDirectory() as directory:
 root=Path(directory);state=root/'preferences'
 w=Window(state_directory=state)
 try:
  w.refresh_recent_models();assert not w.recent_models_menu.actions()[0].isEnabled()
  a=root/'same.margam.json';b=root/'other'/'same.margam.json';b.parent.mkdir()
  assert w.save_model(a) and w.save_model(b)
  assert w.recent_model_paths()==[str(b),str(a)]
  assert 'recent_models' not in str(capture_model(w))
  w.refresh_recent_models();actions=w.recent_models_menu.actions()
  assert actions[0].toolTip()==str(b) and actions[1].toolTip()==str(a)
  # Dirty cancellation leaves graph/results/history/order unchanged.
  w.change('points',81);before=capture_model(w);index=w.undo_stack.index()
  sentinel={'keep':True};w.result=sentinel
  with patch.object(w,'confirm_discard',return_value=False):assert not w.open_recent_model(str(a))
  assert capture_model(w)==before and w.undo_stack.index()==index and w.result is sentinel
  assert w.recent_model_paths()==[str(b),str(a)]
  w.set_busy(True)
  assert not w.open_recent_model(str(a)) and not w.recent_models_menu.isEnabled()
  w.set_busy(False)
  with patch.object(w,'confirm_discard',return_value=True):assert w.open_recent_model(str(a))
  assert w.recent_model_paths()==[str(a),str(b)]
  # Missing paths require explicit removal; do not delete the companion folder.
  b.unlink();owned=w.project_workspace.companion(b);owned.mkdir(exist_ok=True);(owned/'keep.txt').write_text('fixture')
  with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.No):assert not w.open_recent_model(str(b))
  assert str(b) in w.recent_model_paths()
  with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes):assert not w.open_recent_model(str(b))
  assert str(b) not in w.recent_model_paths() and owned.exists()
  invalid=root/'invalid.json';invalid.write_text('{}')
  with patch.object(QMessageBox,'warning'):assert not w.load_model(invalid)
  assert str(invalid) not in w.recent_model_paths()
  with patch.object(w.project_workspace,'save',side_effect=OSError('fixture')),patch.object(QMessageBox,'warning'):
   assert not w.save_model(root/'failed.json')
  assert w.recent_model_paths()==[str(a)]
  w.desktop_state.settings.sync()
  other=Window(state_directory=state)
  try:
   assert other.recent_model_paths()==[str(a)]
   for i in range(15):other.remember_recent_model(root/f'{i}.json')
   assert len(other.recent_model_paths())==10
   other.refresh_recent_models();other.recent_models_menu.actions()[-1].trigger()
   assert other.recent_model_paths()==[] and a.exists()
  finally:other.undo_stack.setClean();other.close()
 finally:w.undo_stack.setClean();w.close()
 # State-disabled tests use only ephemeral memory, never personal preferences.
 isolated=Window(state_directory=False)
 try:
  assert isolated.recent_model_paths()==[]
  isolated.remember_recent_model(a);assert isolated.recent_model_paths()==[str(a)]
 finally:isolated.undo_stack.setClean();isolated.close()
app.clipboard().clear();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=60,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertNotIn('Traceback',result.stderr)
