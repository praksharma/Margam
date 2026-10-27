"""Undo restores model identity/settings/wires, not numerical results or windows."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class EditHistoryTests(unittest.TestCase):
    def run_ui(self, code):
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],
            env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),
            capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_graph_and_setting_round_trips(self):
        self.run_ui(r'''
from PySide6.QtWidgets import QApplication, QComboBox, QSpinBox
from margam.ui.app import Window
app=QApplication([]);w=Window();w.show();app.processEvents()
history=w.undo_stack
assert history.count()==0 and not w.undo_action.isEnabled() and not w.redo_action.isEnabled()
method=next(n for n in w.scene.nodes() if n.kind=='method')
monitor=w.add_node('Training Monitor');identifier=monitor.identifier
w.rename_output(monitor,'My monitor')
w.change_output(monitor,'history_limit',80)
wire=w.scene.connect_nodes(method,monitor)
assert history.count()==4 and w.validate_model()
w.open_block(monitor);app.processEvents()
w.scene.clearSelection();monitor.setSelected(True);w.delete_selected()
assert monitor.scene() is None and identifier not in w.output_views.entries
w.undo_action.trigger()
assert monitor.scene() is w.scene and monitor.identifier==identifier
assert monitor.title=='My monitor' and monitor.settings['history_limit']==80
assert wire in w.scene.wires and wire.target.parentItem() is monitor
assert identifier not in w.output_views.entries and w.validate_model()
w.redo_action.trigger()
assert monitor.scene() is None and wire not in w.scene.wires
w.undo_edit();w.undo_edit()  # Restore deletion, undo connection.
assert monitor.scene() is w.scene and wire not in w.scene.wires
w.undo_edit();assert monitor.settings['history_limit']==10000
w.undo_edit();assert monitor.title=='Training Monitor 1'
w.undo_edit();assert monitor.scene() is None and len(w.scene.nodes())==7
assert not history.canUndo()
for _ in range(4):w.redo_edit()
assert monitor.scene() is w.scene and wire in w.scene.wires
assert monitor.settings['history_limit']==80 and monitor.title=='My monitor'
# Undoing a branch point discards its old redo path.
w.undo_edit();assert history.canRedo()
w.change('points',81)
assert not history.canRedo()
w.undo_edit();assert w.config.points==64
# Per-side recipes restore the absence of an override, not merely its values.
w.change_boundary_sampling('left','points',31)
assert w.config.boundary_recipe('left').points==31
w.undo_edit();assert 'left' not in w.config.boundary_sampling
w.redo_edit();assert w.config.boundary_recipe('left').points==31
w.change_bc('left','type','Neumann')
w.undo_edit();assert w.config.bc_types['left']=='Dirichlet'
w.redo_edit();assert w.config.bc_types['left']=='Neumann'
# Shape history updates visible inspector fields; previews remain optional.
owner=next(n for n in w.scene.nodes() if n.kind=='domain')
w.show_block_properties(owner);w.open_rectangle_preview(owner,True)
w.change('shape','Interval');assert not w.rectangle_previews
w.undo_edit();assert w.config.shape=='Rectangle' and not w.rectangle_previews
assert w.property_scroll.widget().findChild(QComboBox).currentText()=='Rectangle'
# Consecutive values from one property control form a single undo step.
interior=next(n for n in w.scene.nodes() if n.kind=='interior')
w.show_block_properties(interior)
spin=w.property_scroll.widget().findChild(QSpinBox)
before=history.index();old=w.config.points
spin.setValue(100);spin.setValue(121)
assert history.index()==before+1 and w.config.points==121
w.undo_edit();assert w.config.points==old
w.redo_edit();assert w.config.points==121
# Training lock blocks actions and public editing wrappers, preserving history.
w.set_busy(True);index=history.index()
assert not w.undo_action.isEnabled() and not w.redo_action.isEnabled()
w.undo_edit();w.redo_edit();w.change('points',144)
assert history.index()==index and w.config.points==121
w.set_busy(False);assert w.undo_action.isEnabled()
# No-op/rejected operations do not pollute history.
index=history.index();w.change('points',121);w.scene.clearSelection();w.delete_selected()
try:w.scene.connect_nodes(method,owner)
except ValueError:pass
assert history.index()==index
w.reset_template();assert history.count()==0 and len(w.scene.nodes())==7
w.load_heat_template();assert history.count()==0 and len(w.scene.nodes())==11
w.undo_edit();assert len(w.scene.nodes())==11
w.close();app.processEvents()
''')

    def test_real_drag_grouping_shortcuts_and_result_rules(self):
        self.run_ui(r'''
from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit
from margam.ui.app import Window
app=QApplication([]);w=Window();w.show();app.processEvents()
w.actual_size_action.trigger();w.view.centerOn(450,250);app.processEvents()
network=next(n for n in w.scene.nodes() if n.kind=='network')
pde=next(n for n in w.scene.nodes() if n.kind=='pde')
w.scene.clearSelection();network.setSelected(True);pde.setSelected(True)
original={n.identifier:n.pos() for n in (network,pde)}
w.result={'outputs':{},'preserved':True}
start=w.view.mapFromScene(network.mapToScene(network.rect().center()));end=start+QPoint(70,-40)
QTest.mousePress(w.view.viewport(),Qt.MouseButton.LeftButton,pos=start)
QTest.mouseMove(w.view.viewport(),end,20)
QTest.mouseRelease(w.view.viewport(),Qt.MouseButton.LeftButton,pos=end)
app.processEvents()
assert network.pos()!=original[network.identifier] and pde.pos()!=original[pde.identifier]
assert w.undo_stack.count()==1 and w.result['preserved']
moved={n.identifier:n.pos() for n in (network,pde)}
zoom=w.view.transform().m11()
w.view.setFocus();QTest.keyClick(w.view,Qt.Key.Key_Z,Qt.KeyboardModifier.ControlModifier)
assert all(n.pos()==original[n.identifier] for n in (network,pde))
assert w.result['preserved'] and w.view.transform().m11()==zoom
QTest.keyClick(w.view,Qt.Key.Key_Y,Qt.KeyboardModifier.ControlModifier)
assert all(n.pos()==moved[n.identifier] for n in (network,pde))
# Local text undo must not invoke model undo.
w.show_block_properties(network)
edit=w.property_scroll.widget().findChild(QLineEdit)
edit.setFocus();edit.selectAll();QTest.keyClicks(edit,'24, 12')
assert w.config.layers==(24,12)
index=w.undo_stack.index()
QTest.keyClick(edit,Qt.Key.Key_Z,Qt.KeyboardModifier.ControlModifier)
assert w.undo_stack.index()>=index  # The editor's text undo may update the live setting.
assert network.pos()==moved[network.identifier]
# Menu undo replays model state even with inspector focus.
w.undo_action.trigger()
assert network.pos()==moved[network.identifier]
# Semantic changes and their undo never resurrect discarded run data.
w.result={'outputs':{},'preserved':True}
w.change('source',3)
assert w.result is None
w.undo_edit();assert w.config.source==2 and w.result is None
# Group deletion restores nodes and all incident edges in one step.
w.scene.clearSelection();network.setSelected(True);pde.setSelected(True)
count=len(w.scene.wires);w.delete_selected()
assert network.scene() is None and pde.scene() is None
w.undo_edit()
assert network.scene() is w.scene and pde.scene() is w.scene
assert len(w.scene.wires)==count and w.validate_model()
w.close();app.processEvents()
''')

    def test_training_boundary_and_history_limit(self):
        self.run_ui(r'''
import time
from PySide6.QtWidgets import QApplication, QSpinBox
from margam.ui.app import Window
app=QApplication([]);w=Window();w.show();app.processEvents()
w.change('iterations',25)
node=next(n for n in w.scene.nodes() if n.kind=='interior')
w.show_block_properties(node)
spin=w.property_scroll.widget().findChild(QSpinBox)
spin.setValue(81)
index=w.undo_stack.index()
assert w.start_training()
deadline=time.monotonic()+20
while w.worker is not None and time.monotonic()<deadline:
    app.processEvents();time.sleep(.005)
assert w.worker is None and w.result is not None
spin.setValue(100)
assert w.undo_stack.index()==index+1
w.undo_edit();assert w.config.points==81
w.reset_template()
for value in range(3,108):w.change('source',value)
assert w.undo_stack.count()==100
for _ in range(100):w.undo_edit()
assert w.config.source==7 and not w.undo_stack.canUndo()
for _ in range(100):w.redo_edit()
assert w.config.source==107
w.close();app.processEvents()
''')
