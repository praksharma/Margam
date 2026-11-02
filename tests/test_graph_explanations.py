"""Explain graph dependencies and actual execution without changing the model."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class GraphExplanationTests(unittest.TestCase):
    def run_ui(self,code):
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=120,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertNotIn('Traceback',result.stderr)

    def test_wire_settings_and_summary_are_read_only(self):
        self.run_ui(r'''
from copy import deepcopy
from unittest.mock import patch
from PySide6.QtWidgets import QApplication,QWidget
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from margam.ui.app import Window
from margam.ui.model_document import capture_model
from margam.ui.run_summary import execution_plan
from margam.ui.settings_search import entries
app=QApplication([]);w=Window(state_directory=False)
try:
 w.set_application_device('cpu');w.show();app.processEvents()
 before=capture_model(w);history=w.undo_stack.index();sentinel={'preserved':True};w.result=sentinel
 boundary=next(n for n in w.scene.nodes() if n.kind=='boundary')
 pde=next(n for n in w.scene.nodes() if n.kind=='pde')
 bc=next(n for n in w.scene.nodes() if n.kind=='bc')
 pde_wire=next(wire for wire in w.scene.wires if wire.target is pde.inputs['boundary_points'])
 bc_wire=next(wire for wire in w.scene.wires if wire.target.parentItem() is bc)
 assert 'PDE loss' in pde_wire.toolTip() and 'does not include' in bc_wire.toolTip()
 assert 'Collocation' in pde.inputs['input'].toolTip() and 'Boundary' in boundary.output.toolTip()
 w.scene.clearSelection()
 QTest.mouseClick(w.view.viewport(),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,w.view.mapFromScene(pde_wire.path().pointAtPercent(.6)));app.processEvents()
 assert pde_wire.isSelected()
 assert w.name.text()=='Connection' and 'Additional boundary' in w.detail.text()
 assert not any(n.isSelected() for n in w.scene.nodes())
 assert 'points' in boundary.toolTip() and boundary.detail in boundary.toolTip()
 search=w.show_settings_search();search.search.setText('learning rate')
 assert search.results.count()>0
 search.locate()
 focus=next(child for child in w.property_scroll.widget().findChildren(QWidget) if child.property('help_key')=='lr')
 assert focus.hasFocus() or 'Located' in search.status.text()
 search.search.setText('boundary coefficient left');assert search.results.count()==1;search.locate()
 assert 'Located' in search.status.text() and w.name.text()==bc.title
 search.search.setText('boundary sampling points left');assert search.results.count()>=1;assert search.results.currentItem().data(256)[1]=='boundary_sampling.left.points';search.locate();assert 'Located' in search.status.text()
 search.search.setText('no-matching-setting');assert search.results.count()==0
 with patch('margam.templates.poisson.build_pinn',side_effect=AssertionError('must not build a network')):
  config,counts,text=execution_plan(w)
 assert counts==dict(interior=64,boundary=64,pde=128,sets=dict(left=16,right=16,bottom=16,top=16))
 assert config.device=='cpu' and 'Adam' in text and 'Requested outputs: none' in text
 dialog=w.show_run_summary();assert dialog.train.isEnabled()
 assert capture_model(w)==before and w.undo_stack.index()==history and w.result is sentinel
 # Normal Train UI reviews the model first, without launching a worker.
 dialog.close();w.train_button.click();assert not w.is_running() and dialog.isVisible()
 with patch.object(w,'start_training',return_value=True) as start:
  dialog.train.click();start.assert_called_once();assert not dialog.isVisible()
 # Refresh is safe with stale instance IDs and conditional/compact settings.
 search.search.setText('learning rate');record=search.results.currentItem().data(256);node=next(n for n in w.scene.nodes() if n.identifier==record[0])
 w.scene.removeItem(node);search.locate();assert 'no longer' in search.status.text();w.scene.addItem(node)
 w.scene.clearSelection()
 w.load_flow_template();search.refresh();search.search.setText('flow boundary top u target');assert search.results.count()==1
 search.locate();assert w.flow_bc_side=='top' and 'Located' in search.status.text()
 _,counts,text=execution_plan(w);assert counts['pde']>counts['interior'] and 'u, v, p' in text
 search.close();dialog.close()
finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
''')

    def test_freshness_training_edits_evaluation_and_archive(self):
        self.run_ui(r'''
from pathlib import Path
from tempfile import TemporaryDirectory
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.model_document import capture_model
app=QApplication([]);w=Window(state_directory=False)
def wait():
 while w.worker is not None:app.processEvents()
 app.processEvents()
try:
 w.set_application_device('cpu')
 # A newly connected output must stop saying Not connected before its first run.
 extra=w.add_node('Training Monitor');assert extra.result_status=='Not connected'
 trainer=next(n for n in w.scene.nodes() if n.kind=='method')
 w.scene.connect_nodes(trainer,extra);assert extra.result_status=='Not run'
 w.load_heat_template();w.change('iterations',2)
 outputs=[n for n in w.scene.nodes() if n.kind in ('monitor','solution','comparison')]
 assert outputs and all(n.result_status=='Not run' for n in outputs)
 assert w.start_training();assert all(n.result_status=='Running' for n in outputs)
 wait();assert w.result is not None and all(n.result_status=='Current' for n in outputs)
 node=next(n for n in outputs if n.kind=='monitor');w.output_views.open(node)
 view=w.output_views.entries[node.identifier][1];assert view.freshness_label.text()=='Result: Current'
 saved=capture_model(w);assert all('result_status' not in record for record in saved['nodes'])
 w.rename_output(node,'Losses');node.setPos(node.pos().x()+10,node.pos().y());assert node.result_status=='Current'
 with TemporaryDirectory() as directory:
  path=Path(directory)/'run.pt';assert w.save_current_run(path)
  w.change('source',3);assert all(n.result_status=='Settings changed' for n in outputs)
  assert not view.series and view.freshness_label.text()=='Result: Settings changed'
  w.undo_stack.setClean();assert w.open_saved_run(path)
  outputs=[n for n in w.scene.nodes() if n.kind in ('monitor','solution','comparison')]
  assert all(n.result_status=='Current' for n in outputs)
  assert w.evaluate_trained_outputs();assert all(n.result_status=='Evaluating' for n in outputs)
  wait();assert all(n.result_status=='Current' for n in outputs)
  # A stopped run is labelled partial, even though retained outputs are usable.
  w.result['cancelled']=True;w.refresh_result_status('complete')
  assert all(n.result_status=='Stopped / partial' for n in outputs)
  w.active_output_ids={n.identifier for n in outputs};w.training_failed('fixture')
  assert all(n.result_status=='Failed' for n in outputs)
  w.refresh_result_status();assert all(n.result_status=='Failed' for n in outputs)
finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
''')

    def test_summary_concrete_sampling_contracts(self):
        self.run_ui(r'''
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.run_summary import execution_plan
from margam.components.collocation import boundary_rows
app=QApplication([]);w=Window(state_directory=False)
try:
 w.set_application_device('cpu');assert w.load_training_data_example()
 _,counts,_=execution_plan(w);assert counts['pde']==91 and counts['interior']==55 and counts['boundary']==36
 # Remove the boundary-PDE arrow: no automatic append in the review.
 pde=next(n for n in w.scene.nodes() if n.kind=='pde')
 wire=next(wire for wire in w.scene.wires if wire.target is pde.inputs['boundary_points'])
 w.scene.wires.remove(wire);w.scene.removeItem(wire)
 _,counts,_=execution_plan(w);assert counts['pde']==55 and counts['boundary']==0
 # Bypass the inspector cache: only block-held parameters determine the review.
 w.config.points=99999
 _,counts,_=execution_plan(w);assert counts['pde']==55
 w.scene.addItem(wire);w.scene.wires.append(wire)
 # Concrete paper periodic rows are paired and gauge anchors are excluded.
 from margam.templates.examples import EXAMPLE_PROBLEMS
 for key in ('expert_advection','expert_cavity','scale_kdv'):
  assert key in EXAMPLE_PROBLEMS,key
  w.load_example(key)
  c,counts,text=execution_plan(w)
  assert counts['pde']==counts['interior']+counts['boundary']
  assert counts['boundary']>0 and 'Initial PDE rows' in text
 w.load_example('expert_advection');w.change_expert_option('windows',2)
 _,_,text=execution_plan(w);assert 'first window' in text and '2 windows' in text
 # Verify the KdV executor and refresh hook use exactly the plan's connected rows.
 from margam.templates.examples import example_config
 from margam.templates.poisson import build_pinn
 from margam.templates.kdv import samples
 from margam.components.collocation import collocation_rows
 c=example_config('scale_kdv');c.layers=(4,);c.network_type='fcnn';c.network_options={};c.points=8;c.boundary_points=3;c.method='baseline';c.device='cpu'
 for mode,expected in [('interior',8),('boundary',9),('both',17)]:
  c.pde_points=mode;model=build_pinn(c)
  assert len(model.problem.collocation_points)==expected
  rows,_=model.problem.refresh_samples(1);initial,sets=samples(c,1)
  assert rows==collocation_rows(c,initial,sets) and len(rows)==expected
 w.set_busy(True);dialog=w.show_run_summary();assert not dialog.train.isEnabled();w.set_busy(False);dialog.close()
finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
''')
