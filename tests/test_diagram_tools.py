"""Diagram-only edits, deterministic layout and safe instance navigation."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

class DiagramToolsTests(unittest.TestCase):
    def test_editor_layout_and_search(self):
        code=r"""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF
from margam.ui.app import Window
from margam.ui.model_document import capture_model
from margam.ui.diagram_layout import positions
app=QApplication([]);w=Window(state_directory=False)
try:
 nodes=sorted(w.scene.nodes(),key=lambda n:n.identifier)
 for i,n in enumerate(nodes):n.setPos(i*250,i*120)
 originals={n.identifier:QPointF(n.pos()) for n in nodes}
 definitions=capture_model(w)
 sentinel={'unchanged':True};w.result=sentinel
 edges=[(x.source,x.target) for x in w.scene.wires]
 for n in nodes[:3]:n.setSelected(True)
 w.update_layout_actions();assert w.layout_actions['left'].isEnabled()
 before=w.undo_stack.index()
 assert w.arrange_blocks('left')
 assert len({n.pos().x() for n in nodes[:3]})==1
 assert w.undo_stack.index()==before+1
 assert w.result is sentinel and edges==[(x.source,x.target) for x in w.scene.wires]
 w.undo_edit();assert all(n.pos()==originals[n.identifier] for n in nodes)
 w.redo_edit();w.undo_edit()
 for operation in ['center','right','top','middle','bottom','horizontal','vertical']:
  proposals=positions(nodes[:3],w.scene.wires,operation)
  assert len(proposals)==3
 assert w.arrange_blocks('auto')
 assert all(x.source.parentItem().pos().x()<x.target.parentItem().pos().x() for x in w.scene.wires)
 first={n.identifier:QPointF(n.pos()) for n in nodes}
 assert not w.arrange_blocks('auto') # stable second application, no undo entry
 assert all(n.pos()==first[n.identifier] for n in nodes)
 assert w.result is sentinel
 w.undo_edit();assert all(n.pos()==originals[n.identifier] for n in nodes)
 assert capture_model(w)==definitions
 w.set_busy(True);w.update_layout_actions()
 assert all(not a.isEnabled() for a in w.layout_actions.values())
 assert not w.arrange_blocks('left')
 w.set_busy(False)
 dialog=w.show_model_search();dialog.search.setText(nodes[0].identifier)
 assert dialog.results.count()==1
 index=w.undo_stack.index();dialog.locate()
 assert nodes[0].isSelected() and w.undo_stack.index()==index and w.result is sentinel
 # A stale search result must not retain or access a deleted graphics object.
 w.scene.removeItem(nodes[0]);dialog.locate()
 assert dialog.results.count()==0 and 'no longer' in dialog.status.text()
 w.scene.addItem(nodes[0])
 dialog.search.setText('not-a-real-block');assert dialog.results.count()==0
 dialog.close()
 # No overlap is introduced by distribute when the span cannot accommodate blocks.
 for i,n in enumerate(nodes[:3]):n.setPos(i,0)
 assert positions(nodes[:3],[], 'horizontal')=={}
 # Kahn leftovers from cyclic wiring terminate and remain separately visible.
 class Port:
  def __init__(self,node):self.node=node
  def parentItem(self):return self.node
 class Edge:
  def __init__(self,a,b):self.source=Port(a);self.target=Port(b)
 cycle=positions(nodes[:3],[Edge(nodes[0],nodes[1]),Edge(nodes[1],nodes[0])],'auto')
 assert len(set(cycle.values()))==3
finally:
 w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=90,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertNotIn('Traceback',result.stderr)
