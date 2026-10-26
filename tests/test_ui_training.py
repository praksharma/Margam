"""Exercise real background execution, graph validation and cancellation via Qt."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class UITrainingTests(unittest.TestCase):
    def test_end_to_end_baseline_and_shutdown(self):
        code = r'''
import time
from unittest.mock import patch
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
app = QApplication([])
w = Window()
w.show()
app.processEvents()

def until_finished(timeout=45):
    deadline = time.monotonic()+timeout
    while w.worker is not None and time.monotonic()<deadline:
        app.processEvents()
        time.sleep(.005)
    assert w.worker is None, 'Worker failed to finish'
    app.processEvents()

# Removing a required edge must block execution, rather than train a hidden model.
wire=w.scene.wires.pop()
source,target=wire.source.parentItem(),wire.target.parentItem()
w.scene.removeItem(wire)
assert not w.start_training()
assert 'Missing connections' in w.validation.text()
start=w.view.mapFromScene(source.output.scenePos())
end=w.view.mapFromScene(target.input.scenePos())
QTest.mousePress(w.view.viewport(),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start)
QTest.mouseMove(w.view.viewport(),end,20)
QTest.mouseRelease(w.view.viewport(),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,end)
app.processEvents()
assert len(w.scene.wires)==8, 'Port drag did not restore the connection' 

# Match the original analytical interval example and prove event-loop responsiveness.
w.change('shape','Interval')
w.change('points',32)
beats=[]
heartbeat=QTimer()
heartbeat.setInterval(10)
heartbeat.timeout.connect(lambda:beats.append(1))
heartbeat.start()
assert w.start_training()
assert not w.train_button.isEnabled()
assert not w.property_scroll.isEnabled()
until_finished()
heartbeat.stop()
assert len(beats)>5, 'UI event loop did not stay responsive'
assert w.result is not None, w.log.toPlainText()
assert not w.result['cancelled']
assert w.result['steps']==1000
assert w.result['rmse']<.02, w.result['rmse']
assert len(w.history)==1000
assert w.progress_bar.value()==1000
assert w.train_button.isEnabled()
assert w.tabs.currentWidget() is w.solution

# Changing a property invalidates results, then the default rectangle trains too.
w.change('shape','Rectangle')
w.change('points',64)
assert w.result is None
assert w.start_training()
until_finished()
assert w.result['rmse'] is None
assert len(w.result['values'])==41*41
assert w.history[-1]['total'] < w.history[0]['total']*.1
assert not w.grab().isNull()

# Worker errors reach the UI and release controls; they do not kill the app.
with patch('margam.templates.poisson.build_pinn',side_effect=ValueError('Deliberate test failure')):
    assert w.start_training()
    until_finished()
assert w.result is None
assert 'Deliberate test failure' in w.log.toPlainText()
assert w.train_button.isEnabled()

# Stop is available without closing and leaves the app usable.
w.change('iterations',100000)
assert w.start_training()
QTest.mouseClick(w.cancel_button,Qt.MouseButton.LeftButton)
until_finished()
assert w.result['cancelled'] and w.result['steps']<100000
assert w.isVisible() and w.train_button.isEnabled()

# Closing during training requests cancellation and waits for safe thread teardown.
w.change('iterations',100000)
assert w.start_training()
w.close()
assert w._closing
until_finished()
assert w.result['cancelled']
assert w.result['steps']<100000
app.processEvents()
assert not w.isVisible()
print('Interval convergence, rectangle training, responsiveness, validation, failure and close/cancel passed.')
'''
        result = subprocess.run([sys.executable, '-c', code],
                                cwd=Path(__file__).resolve().parents[1],
                                env=dict(os.environ, QT_QPA_PLATFORM='offscreen', PYTHONDONTWRITEBYTECODE='1'),
                                capture_output=True, text=True, timeout=100)
        self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
        self.assertNotIn('Traceback', result.stderr)
