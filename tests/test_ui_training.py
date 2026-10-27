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
end=w.view.mapFromScene(wire.target.scenePos())
QTest.mousePress(w.view.viewport(),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start)
QTest.mouseMove(w.view.viewport(),end,20)
QTest.mouseRelease(w.view.viewport(),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,end)
app.processEvents()
assert len(w.scene.wires)==7, 'Port drag did not restore the connection'

# Training alone creates no predictions, reference comparison or plot windows.
w.change('iterations',25)
assert w.start_training()
until_finished()
assert w.result['outputs']=={}
assert w.last_run.model.iter==25
assert not w.output_views.entries

# Output requests are explicit and instances have their own retained history/settings.
w.change('shape','Interval')
w.change('points',32)
w.change('iterations',1000)
method=next(n for n in w.scene.nodes() if n.kind=='method')
monitor=w.add_node('Training Monitor');monitor.setPos(1000,-80)
solution=w.add_node('Solution Plot');solution.setPos(1000,90)
reference=w.add_node('Reference');reference.setPos(730,450)
comparison=w.add_node('Comparison');comparison.setPos(1000,350)
for node in [monitor,solution,comparison]:w.scene.connect_nodes(method,node)
assert not w.validate_model(), 'Comparison must require a reference'
w.scene.connect_nodes(reference,comparison)
w.change_output(monitor,'history_limit',100)
w.fit_model();app.processEvents()
assert w.validate_model()
assert not w.output_views.entries, 'Connections must not open views'

# Double-click opens a view; closing it hides the view but keeps recording requested.
point=w.view.mapFromScene(monitor.mapToScene(monitor.rect().center()))
QTest.mouseClick(w.view.viewport(),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,point)
app.processEvents()
point=w.view.mapFromScene(monitor.mapToScene(monitor.rect().center()))
QTest.mouseDClick(w.view.viewport(),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,point)
app.processEvents()
assert monitor.identifier in w.output_views.entries
monitor_dock,_=w.output_views.entries[monitor.identifier]
monitor_dock.close()
assert monitor_dock.isHidden()
beats=[]
heartbeat=QTimer()
heartbeat.setInterval(10)
heartbeat.timeout.connect(lambda:beats.append(1))
heartbeat.start()
opened_during=[]
def reopen_monitor_during_training():
    assert w.is_running()
    w.scene.clearSelection();monitor.setSelected(True)
    w.open_output_action.trigger()
    assert not monitor_dock.isHidden()
    monitor_dock.close()
    opened_during.append(True)
QTimer.singleShot(50,reopen_monitor_during_training)
assert w.start_training()
assert not w.train_button.isEnabled()
assert not w.property_scroll.isEnabled()
until_finished()
heartbeat.stop()
assert len(beats)>5, 'UI event loop did not stay responsive'
assert opened_during, 'Monitor could not be reopened during training'
assert w.result is not None, w.log.toPlainText()
assert not w.result['cancelled']
assert w.result['steps']==1000
assert w.result['outputs'][comparison.identifier]['rmse']<.02
assert w.result['outputs'][solution.identifier]['reference'] is None
assert w.result['outputs'][solution.identifier]['rmse'] is None
assert len(w.monitor_records[monitor.identifier])==100
assert w.monitor_records[monitor.identifier][-1][0]==1000
assert w.progress_bar.value()==1000
assert w.train_button.isEnabled()
assert monitor_dock.isHidden(), 'Training must not reopen hidden views'
assert len(w.output_views.entries)==1, 'Training must not create unopened views'

# Open the prediction and comparison independently; detach/hide/redock retains identity.
solution_dock=w.open_block(solution);app.processEvents()
_,view=w.output_views.entries[solution.identifier]
assert len(view.curve.series)==1
comparison_dock=w.open_block(comparison);app.processEvents()
assert len(w.output_views.entries[comparison.identifier][1].curve.series)==2
solution_dock.setFloating(True);app.processEvents()
assert solution_dock.isFloating()
solution_dock.close()
w.reset_panel_layout();app.processEvents()
assert solution_dock.isHidden() and not solution_dock.isFloating()
assert w.open_block(solution) is solution_dock

# Removing an output removes its request/view. Unused reference sources do nothing.
w.scene.clearSelection();comparison.setSelected(True)
w.delete_selected();app.processEvents()
assert comparison.identifier not in w.output_views.entries
w.change('shape','Rectangle')
w.change('points',64)
w.change_output(solution,'resolution',21)
assert w.result is None
with patch('margam.ui.app.make_poisson_reference',side_effect=AssertionError('Implicit reference')):
    assert w.start_training()
    until_finished()
assert w.result['outputs'][solution.identifier]['rmse'] is None
assert len(w.result['outputs'][solution.identifier]['values'])==21*21
assert w.monitor_records[monitor.identifier][-1][1]['total']<.1
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
