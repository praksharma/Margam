"""Archived run data/Adam state round-trip and resume identically to continuous training."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class RunFileTests(unittest.TestCase):
    def test_run_archive_and_resume(self):
        code=r'''
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import torch
from PySide6.QtWidgets import QApplication,QMessageBox
from margam.ui.app import Window
from margam.templates.poisson import build_pinn
app=QApplication([]);w=Window(state_directory=False);w.show();w.change('shape','Interval');w.change('iterations',25)
method=next(n for n in w.scene.nodes() if n.kind=='method')
monitor=w.add_node('Training Monitor');solution=w.add_node('Solution Plot')
monitor_id=monitor.identifier;solution_id=solution.identifier
w.scene.connect_nodes(method,monitor);w.scene.connect_nodes(method,solution)
def finish():
    deadline=time.monotonic()+20
    while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
    assert w.worker is None and w.result is not None,w.log.toPlainText()
assert w.start_training();finish()
expected=w.result['outputs'][solution_id]['values']
assert w.save_run_action.isEnabled() and w.resume_action.isEnabled()
with TemporaryDirectory() as directory:
    path=Path(directory)/'baseline.margam-run.pt'
    assert w.save_current_run(path)
    w.undo_stack.setClean()
    assert w.open_saved_run(path)
    assert w.result['outputs'][solution_id]['values']==expected
    assert len(w.monitor_records[monitor_id])==25 and not w.output_views.entries
    assert w.loaded_checkpoint is not None and w.resume_action.isEnabled()
    node=next(n for n in w.scene.nodes() if n.identifier==solution_id)
    w.open_block(node);assert solution_id in w.output_views.entries
    assert w.resume_training(25);finish()
    assert w.result['steps']==50 and len(w.monitor_records[monitor_id])==50
    continuous=build_pinn(w.config);continuous.train(50,display_every=0)
    points=torch.linspace(0,1,21)[:,None]
    torch.testing.assert_close(w.last_run.model.predict(points),continuous.predict(points),rtol=0,atol=0)
    saved=w.result
    bad=Path(directory)/'bad.pt';torch.save({'bad':'data'},bad)
    with patch.object(QMessageBox,'warning'):assert not w.open_saved_run(bad)
    assert w.result is saved
    w.change('source',3)
    assert w.result is None and w.loaded_checkpoint is None and not w.resume_action.isEnabled()
w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=40)
        self.assertEqual(result.returncode,0,result.stderr)
