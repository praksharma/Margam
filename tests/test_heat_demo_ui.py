"""The heat template supplies an explicit monitor and 2D reference comparison."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class HeatDemoUITests(unittest.TestCase):
    def test_ready_wired_heat_demo_and_reference_views(self):
        code=r'''
import time
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
app=QApplication([]);w=Window();w.show()
for _ in range(3):app.processEvents()
assert len(w.scene.nodes())==7
w.heat_action.trigger();app.processEvents()
assert len(w.scene.nodes())==11 and len(w.scene.wires)==11
assert len(w.scene.output_requests(w.config))==3
assert w.validate_model()
assert not w.output_views.entries and not w.rectangle_previews
monitor=next(n for n in w.scene.nodes() if n.kind=='monitor')
comparison=next(n for n in w.scene.nodes() if n.kind=='comparison')
solution=next(n for n in w.scene.nodes() if n.kind=='solution')
w.open_block(monitor)
assert w.start_training()
assert not w.heat_action.isEnabled()
deadline=time.monotonic()+90
while w.worker is not None and time.monotonic()<deadline:
    app.processEvents();time.sleep(.005)
assert w.worker is None and w.result is not None,w.log.toPlainText()
assert w.result['steps']==w.config.iterations
assert len(w.monitor_records[monitor.identifier])==w.config.iterations
assert w.heat_action.isEnabled()
data=w.result['outputs'][comparison.identifier]
assert data['rmse']<.035,data['rmse']
assert 'truncated' in data['reference_name']
assert w.result['outputs'][solution.identifier]['reference'] is None
assert len(w.output_views.entries)==1, 'Training must not open comparison automatically'
w.open_block(comparison);app.processEvents()
view=w.output_views.entries[comparison.identifier][1]
assert view.rectangle_comparison is not None
assert 'RMSE' in view.rectangle_comparison.metrics.text()
for _ in range(3):app.processEvents()
assert view.rectangle_comparison.prediction.height()>=230
assert not view.grab().isNull()
w.reset_template();app.processEvents()
assert len(w.scene.nodes())==7 and not w.output_views.entries
w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],
                              env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),
                              capture_output=True,text=True,timeout=110)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertNotIn('Traceback',result.stderr)
