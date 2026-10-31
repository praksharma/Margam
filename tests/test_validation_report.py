"""Independent issues, navigation and unchanged authoritative execution checks."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class ValidationReportTests(unittest.TestCase):
    def run_ui(self,code):
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=90,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertNotIn('Traceback',result.stderr)

    def test_collect_many_issues_navigate_and_fix(self):
        self.run_ui(r'''
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.model_document import capture_model
app=QApplication([]);w=Window(state_directory=False)
try:
 w.change_layers('0, -1');w.change('pde_expression','not_allowed(u)')
 w.change_bc('left','type','Custom');w.change_bc_expression('left','bc_operators','bad(u)')
 w.change_bc('right','type','Custom');w.change_bc_expression('right','bc_operators','also_bad(u)')
 # Remove two independent required connections without deleting their blocks.
 for wire in list(w.scene.wires):
  if wire.target.parentItem().kind=='method' and wire.target.name in ('pde','network'):
   w.scene.wires.remove(wire);w.scene.removeItem(wire)
 before=capture_model(w);index=w.undo_stack.index();w.result={'sentinel':True}
 dialog=w.show_validation_report();app.processEvents()
 issues=w.last_validation_issues
 assert sum(i.category=='Connections' for i in issues)>=2
 assert any(i.category=='Network' for i in issues)
 assert any(i.category=='Physics' for i in issues)
 assert {i.boundary for i in issues if i.category=='Boundary condition'}=={'left','right'}
 assert capture_model(w)==before and w.undo_stack.index()==index and w.result=={'sentinel':True}
 assert not w.start_training() and w.worker is None
 row=next(i for i,issue in enumerate(dialog.issues) if issue.boundary=='right');dialog.table.selectRow(row);dialog.select_issue()
 assert w.boundary_side=='right' and next(n for n in w.scene.nodes() if n.kind=='bc').isSelected()
 w.result=None;w.reset_template();assert 'changed' in dialog.summary.text()
 assert w.validate_model() and not any(i.severity=='Error' for i in dialog.issues)
 assert 'passed' in dialog.summary.text()
 # Programmatic checks never open a report in a fresh window.
 other=Window(state_directory=False)
 assert other.validate_model() and not hasattr(other,'validation_report_dialog')
 other.undo_stack.setClean();other.close()
finally:
 app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
''')

    def test_multiple_missing_files_and_unused_source_warning(self):
        self.run_ui(r'''
from pathlib import Path
from tempfile import TemporaryDirectory
from copy import deepcopy
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
app=QApplication([]);w=Window(state_directory=False)
try:
 assert w.load_training_data_example()
 with TemporaryDirectory() as directory:
  boundaries=deepcopy(w.config.boundary_data)
  boundaries['base']['path']=str(Path(directory)/'base.csv');boundaries['slope']['path']=str(Path(directory)/'slope.csv')
  w.change('boundary_data',boundaries)
  assert not w.validate_model()
  data=[i for i in w.last_validation_issues if i.category=='Input data' and i.severity=='Error']
  assert {'base','slope'}<={i.boundary for i in data}
 w.reset_template();unused=w.add_node('Reference data')
 assert w.validate_model() # Unused optional source doesn't block the executor.
 assert any(i.severity=='Warning' and i.node_id==unused.identifier for i in w.last_validation_issues)
 # Legacy mapped references used to defer file errors until prediction.
 w.change('shape','Interval');reference=w.add_node('Reference');comparison=w.add_node('Comparison')
 w.scene.connect_nodes(next(n for n in w.scene.nodes() if n.kind=='method'),comparison)
 w.scene.connect_nodes(reference,comparison)
 with TemporaryDirectory() as directory:
  w.change_output(reference,'reference_data',dict(path=str(Path(directory)/'missing.csv'),coordinates='x',normals='',values='u'))
  assert not w.validate_model() and any(i.severity=='Error' and i.node_id==reference.identifier for i in w.last_validation_issues)
  assert not w.start_training() and w.worker is None
  path=Path(directory)/'missing.csv';path.write_text('x,u\n0,0\n0.25,0.25\n1,0\n')
  w.change_output(comparison,'resolution',3)
  assert not w.validate_model() and any('Reference coordinates' in i.message for i in w.last_validation_issues)
 # Checking a missing project copy must not recreate its companion directory.
 w.load_training_data_example()
 from margam.components.prepared_data import read_points
 import shutil
 with TemporaryDirectory() as directory:
  owned=w.project_workspace.store_dataset(w.config.interior_data,copy=True,validator=lambda s:read_points(s,2))
  w.change('interior_data',owned);assert w.save_model(Path(directory)/'copy.margam.json')
  folder=Path(directory)/'copy_files';shutil.rmtree(folder)
  assert not w.validate_model() and not folder.exists()
  dialog=w.show_validation_report()
  row=next(i for i,issue in enumerate(dialog.issues) if issue.category=='Input data')
  dialog.table.selectRow(row);dialog.select_issue();assert not folder.exists()

finally:
 app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
''')

    def test_compatibility_and_authoritative_cross_checks(self):
        self.run_ui(r'''
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.validation_report import check_execution
app=QApplication([]);w=Window(state_directory=False)
try:
 w.change('method','hard_dirichlet');w.change_bc('left','type','Neumann')
 assert not w.validate_model()
 assert any('Hard Dirichlet' in i.message and i.severity=='Error' for i in w.last_validation_issues)
 w.reset_template();w.change_bc('left','value',2.)
 assert not w.validate_model() # The existing corner check remains a blocking error.
 assert any('corner' in i.message for i in w.last_validation_issues)
 w.reset_template();w.change('loss_formulation','integral')
 assert not w.validate_model() and any('quadrature' in i.message for i in w.last_validation_issues)
 w.load_flow_template()
 w.change_flow_boundary('left','u','target','bad(u)')
 w.change_flow_boundary('right','v','type','Custom');w.change_flow_boundary('right','v','operator','bad(v)')
 assert not w.validate_model()
 assert {'left','right'}<={i.boundary for i in w.last_validation_issues if i.category=='Boundary condition'}
 w.load_training_data_example()
 from copy import deepcopy
 mappings=deepcopy(w.config.boundary_data)
 for side in ('upright','slope'):mappings[side]['normals']=''
 w.change('boundary_data',mappings);assert not w.validate_model()
 assert {'upright','slope'}<={i.boundary for i in w.last_validation_issues if i.category=='Input data'}
 plot=w.add_node('Solution Plot');w.scene.connect_nodes(next(n for n in w.scene.nodes() if n.kind=='method'),plot)
 assert not w.validate_model() and any('Evaluation points' in i.message and i.node_id==plot.identifier for i in w.last_validation_issues)
 w.reset_template();config,count=check_execution(w)
 assert w.validate_model() and w.config==config and count==0
 w.validate_button.click();app.processEvents();assert w.validation_report_dialog.isVisible()
 # Removed blocks are never retained as Qt objects by old issues.
 w.change_layers('0');assert not w.validate_model()
 dialog=w.validation_report_dialog;index=next(i for i,x in enumerate(dialog.issues) if x.category=='Network');dialog.table.selectRow(index)
 network=next(n for n in w.scene.nodes() if n.kind=='network');w.delete_block(network)
 dialog.select_issue();assert 'removed' in dialog.details.toPlainText()
finally:
 app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
''')
