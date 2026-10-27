"""Port direction/type/cardinality checks apply before a wire enters the model."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class TypedPortTests(unittest.TestCase):
    def test_wiring_contracts_and_rejected_gestures(self):
        code=r'''
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window, Wire
from margam.ui.ports import INPUTS, OUTPUTS
from margam.ui.block_catalogue import KINDS
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
scene=w.scene
nodes={n.kind:n for n in scene.nodes()}
method=nodes['method'];network=nodes['network'];pde=nodes['pde']
assert set(INPUTS)==set(OUTPUTS)==set(KINDS.values())
assert set(method.inputs)=={'pde','boundary','network'}
assert len({port.pos().y() for port in method.inputs.values()})==3
assert not nodes['domain'].inputs and not network.inputs
assert all(wire.source.data_type==wire.target.data_type for wire in scene.wires)
assert {wire.target.name for wire in scene.wires if wire.target.parentItem() is method}=={'pde','boundary','network'}
assert w.validate_model()
# A rejected native drag must not change the graph or clear trained results.
w.result={'preserved':True}
count=len(scene.wires)
w.fit_model();app.processEvents()
start=w.view.mapFromScene(network.output.scenePos())
end=w.view.mapFromScene(method.inputs['pde'].scenePos())
QTest.mousePress(w.view.viewport(),Qt.MouseButton.LeftButton,pos=start)
QTest.mouseMove(w.view.viewport(),end,20)
QTest.mouseRelease(w.view.viewport(),Qt.MouseButton.LeftButton,pos=end)
app.processEvents()
assert len(scene.wires)==count and scene.pending is None
assert w.result=={'preserved':True}
assert 'cannot feed' in w.validation.text() and 'Connection:' in w.log.toPlainText()
# Same endpoint checks for programmatic creation.
def rejected(call, fragment):
    before=len(scene.wires)
    try:call()
    except ValueError as error:assert fragment in str(error),str(error)
    else:raise AssertionError('Unsupported connection accepted')
    assert len(scene.wires)==before
rejected(lambda:scene.connect_nodes(pde,method),'already has a source')
rejected(lambda:scene.connect_nodes(network,method,'pde'),'cannot feed')
rejected(lambda:scene.connect_ports(method.inputs['pde'],pde.output),'output to an input')
rejected(lambda:scene.connect_ports(method.output,method.inputs['pde']),'itself')
rejected(lambda:scene.connect_nodes(method,nodes['domain']),'Choose an output')
# One source branches to multiple explicitly requested outputs; each input gets one source.
a=w.add_node('Training Monitor');b=w.add_node('Training Monitor')
assert a.output is None and b.output is None
scene.connect_nodes(method,a);scene.connect_nodes(method,b)
assert w.validate_model() and len(scene.output_requests(w.config))==2
assert not w.output_views.entries
rejected(lambda:scene.connect_nodes(method,a),'already has a source')
reference=w.add_node('Reference');comparison=w.add_node('Comparison')
assert not reference.inputs and comparison.output is None
scene.connect_nodes(method,comparison)
rejected(lambda:scene.connect_nodes(reference,comparison,'model'),'cannot feed')
scene.connect_nodes(reference,comparison)
assert w.validate_model()
# Full validation also checks endpoints if an invalid wire bypasses editor creation.
bad=Wire(network.output,method.inputs['pde']);scene.addItem(bad);scene.wires.append(bad)
assert not w.validate_model()
assert any(text in w.validation.text() for text in ('cannot feed','already has a source'))
scene.wires.remove(bad);scene.removeItem(bad)
assert w.validate_model()
# Cancelling a dangling gesture is an edit-free operation; training clears it safely.
w.result={'preserved':True}
scene.begin_wire(method.output)
scene.finish_wire(method.output.scenePos()+method.output.pos()*10)
assert scene.pending is None and w.result=={'preserved':True}
scene.begin_wire(method.output)
assert scene.pending is not None
w.set_busy(True)
assert scene.pending is None
rejected(lambda:scene.connect_nodes(method,w.add_node('Training Monitor') or a),'locked')
scene.begin_wire(method.output);assert scene.pending is None
w.set_busy(False)
# Removing an owner cancels an incomplete gesture rather than leaving stale endpoints.
scene.begin_wire(method.output)
scene.clearSelection();method.setSelected(True)
w.delete_selected()
assert scene.pending is None
w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],
            env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),
            capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_contextual_double_click_uses_existing_state(self):
        code=r'''
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit
from margam.ui.app import Window
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
def double_click(node):
    w.view.centerOn(node);app.processEvents()
    point=w.view.mapFromScene(node.mapToScene(node.rect().center()))
    QTest.mouseClick(w.view.viewport(),Qt.MouseButton.LeftButton,pos=point)
    app.processEvents()
    point=w.view.mapFromScene(node.mapToScene(node.rect().center()))
    QTest.mouseDClick(w.view.viewport(),Qt.MouseButton.LeftButton,pos=point)
    QTest.mouseRelease(w.view.viewport(),Qt.MouseButton.LeftButton,pos=point)
    app.processEvents()
network=next(n for n in w.scene.nodes() if n.kind=='network')
w.properties_dock.close()
double_click(network)
assert not w.properties_dock.isHidden() and network.isSelected()
edit=w.property_scroll.widget().findChild(QLineEdit)
edit.setText('20, 12')
assert w.config.layers==(20,12)
w.show_block_properties(network)
assert w.property_scroll.widget().findChild(QLineEdit).text()=='20, 12'
reference=w.add_node('Reference');reference.setPos(150,-150)
double_click(reference)
assert reference.isSelected() and not w.output_views.entries
monitor=w.add_node('Training Monitor');monitor.setPos(-200,-200)
double_click(monitor)
assert monitor.identifier in w.output_views.entries
view=w.output_views.entries[monitor.identifier][0]
view.close();double_click(monitor)
assert w.output_views.entries[monitor.identifier][0] is view and not view.isHidden()
assert w.quick_insert is None
w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],
            env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),
            capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
