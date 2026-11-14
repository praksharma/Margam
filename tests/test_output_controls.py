"""Output-only disabling keeps graph semantics, stored projects and workers honest."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
import torch

CASE=r'''
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch,Mock
import time,os,torch
from PySide6.QtWidgets import QApplication,QCheckBox
from margam.ui.app import Window
from margam.ui.output_views import OUTPUT_KINDS,output_enabled
from margam.ui.model_document import capture_model,decode_document
from margam.ui.validation_report import collect_issues
from margam.ui.settings_search import entries
from margam.components.outputs import evaluate_outputs
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
def finish():
 deadline=time.monotonic()+60
 while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.003)
 assert w.worker is None and w.result is not None,w.log.toPlainText()
try:
 w.set_application_device(os.environ['MARGAM_TEST_DEVICE'])
 w.load_example('interval_poisson');w.change('points',8);w.change('layers',(5,));w.change('iterations',5)
 outputs=[n for n in w.scene.nodes() if n.kind in OUTPUT_KINDS]
 export=w.add_node('ParaView Export');root=next(n for n in w.scene.nodes() if n.kind=='method')
 w.scene.connect_nodes(root,export);outputs.append(export)
 # Disable every optional output without removing wires or changing physics.
 before=w.scene.compile_config();wires=len(w.scene.wires)
 for node in outputs:
  assert w.set_output_enabled(node,False)
  assert node.result_status=='Disabled' and not output_enabled(node)
 assert w.scene.compile_config()==before and len(w.scene.wires)==wires
 assert w.scene.output_requests(before)==[] and w.validate_model(),w.validation.text()
 fake=Mock();assert evaluate_outputs(fake,before,w.scene.output_requests(before))=={};fake.predict.assert_not_called()
 for node in outputs:
  assert not w.open_block(node)
  if node.kind in ('export','paraview'):
   with TemporaryDirectory() as directory:
    path=Path(directory)/'disabled.csv';assert not w.export_block_data(node,path);assert not path.exists()
 # No physics/Reference/source flag or non-boolean flag is accepted.
 for node in w.scene.nodes():
  if node.kind not in OUTPUT_KINDS:assert not w.set_output_enabled(node,False)
 assert not w.set_output_enabled(outputs[0],0)
 core=next(n for n in w.scene.nodes() if n.kind in ('pde','predefined_pde'));core.settings['enabled']=False
 try:w.scene.compile_config()
 except ValueError as error:assert 'required physics cannot be disabled' in str(error)
 else:raise AssertionError('Disabled required physics was accepted')
 core.settings.pop('enabled')
 doc=capture_model(w);assert doc['version']==33;decode_document(doc)
 for flag in (False,True,0,'false'):
  bad=deepcopy(doc);bad['nodes'][0]['settings']['enabled']=flag
  if bad['nodes'][0]['kind'] not in OUTPUT_KINDS:
   try:decode_document(bad)
   except ValueError:pass
   else:raise AssertionError('Core output flag was accepted')
 for flag in (0,1,'false',None):
  bad=deepcopy(doc);next(n for n in bad['nodes'] if n['kind']=='paraview')['settings']['enabled']=flag
  try:decode_document(bad)
  except ValueError:pass
  else:raise AssertionError('Malformed output flag was accepted')
 legacy=deepcopy(doc);legacy['version']=32
 for n in legacy['nodes']:n['settings'].pop('enabled',None)
 _,nodes,_=decode_document(legacy);assert all(n['settings'].get('enabled',True) for n in nodes)
 # Inspector and context actions are undoable; selection/search and clipboard survive.
 node=next(n for n in outputs if n.kind=='solution');w.scene.clearSelection();node.setSelected(True);app.processEvents()
 control=w.property_scroll.findChild(QCheckBox,'output_enabled');assert control and not control.isChecked()
 index=w.undo_stack.index();control.setChecked(True);app.processEvents();assert output_enabled(node) and w.undo_stack.index()==index+1
 w.undo_edit();assert not output_enabled(node);w.redo_edit();assert output_enabled(node)
 action=next(a for a in w.create_block_menu(node).actions() if a.text()=='Output enabled');action.trigger();assert not output_enabled(node)
 assert any(row[0]==node.identifier and row[1]=='output_enabled' for row in entries(w))
 assert w.copy_blocks() and w.paste_blocks()
 duplicate=next(n for n in w.scene.nodes() if n.kind=='solution' and n is not node)
 assert not output_enabled(duplicate);w.undo_edit();assert not w.set_output_enabled(duplicate,True)
 w.set_busy(True);assert not w.set_output_enabled(node,True)
 assert not next(a for a in w.create_block_menu(node).actions() if a.text()=='Output enabled').isEnabled()
 w.set_busy(False)
 # Disabled Comparison data is not evaluated; enabled dependent exports fail clearly.
 comparison=next(n for n in outputs if n.kind=='comparison')
 dependent=w.add_node('CSV Export');w.scene.connect_nodes(comparison,dependent,'comparison')
 try:w.scene.output_requests(w.scene.compile_config())
 except ValueError as error:assert 'disabled Comparison' in str(error)
 else:raise AssertionError('Disabled dependency was silently bypassed')
 issues=collect_issues(w);assert any(i.node_id==dependent.identifier and i.severity=='Error' for i in issues)
 assert w.set_output_enabled(dependent,False);outputs.append(dependent)
 reference=next(n for n in w.scene.nodes() if n.kind=='reference');original=reference.settings.get('reference_expression')
 reference.settings['reference_expression']='bad expression!'
 assert w.scene.output_requests(w.scene.compile_config())==[]
 reference.settings.pop('reference_expression') if original is None else reference.settings.update(reference_expression=original)
 # Native CPU/CUDA workers retain no disabled output data or monitor requests.
 assert w.start_training();finish()
 assert w.result['outputs']=={} and not w.monitor_records and not w.active_monitors
 saved=deepcopy(w.last_run.model.training_state());assert saved['iter']==5
 with TemporaryDirectory() as directory:
  directory=Path(directory);model=directory/'model.margam.json';archive=directory/'run.pt'
  assert w.save_model(model) and w.save_current_run(archive)
  assert w.load_model(model)
  assert all(not output_enabled(n) for n in w.scene.nodes() if n.kind in OUTPUT_KINDS)
  assert w.open_saved_run(archive) and w.resume_training(2);finish()
  assert w.result['steps']==7 and not w.result['outputs'] and not w.monitor_records
  old=deepcopy(w.last_run.model.training_state())
  solution=next(n for n in w.scene.nodes() if n.kind=='solution')
  assert w.set_output_enabled(solution,True)
  assert w._evaluation_config is not None and w.evaluate_trained_outputs();finish()
  assert w.result['steps']==7 and set(w.result['outputs'])=={solution.identifier}
  current=w.last_run.model.training_state()
  assert current['iter']==old['iter'] and current['loss_history']==old['loss_history']
  for key,value in old['network'].items():assert torch.equal(value.cpu(),current['network'][key].cpu())
  for n in w.scene.nodes():
   if n.kind in OUTPUT_KINDS and n is not solution:assert n.result_status=='Disabled'
  from margam.ui.run_document import capture_run,validate_run
  good=capture_run(w);bad=deepcopy(good)
  next(n for n in bad['model']['nodes'] if n['id']==solution.identifier)['settings']['enabled']=False
  try:validate_run(bad)
  except ValueError as error:assert 'Disabled outputs' in str(error)
  else:raise AssertionError('Disabled output archive data was accepted')
  bad=deepcopy(good);monitor=next(n for n in w.scene.nodes() if n.kind=='monitor');bad['monitors'][monitor.identifier]=[]
  try:validate_run(bad)
  except ValueError as error:assert 'Disabled monitors' in str(error)
  else:raise AssertionError('Disabled monitor archive data was accepted')
finally:
 app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
'''


class OutputControlTests(unittest.TestCase):
    def run_case(self,device):
        result=subprocess.run([sys.executable,'-c',CASE],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=120,
            env=dict(os.environ,QT_QPA_PLATFORM='offscreen',MARGAM_TEST_DEVICE=device,PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertNotIn('Traceback',result.stderr)

    def test_cpu_disabled_outputs_persistence_workers_and_evaluation(self):self.run_case('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_disabled_outputs_persistence_workers_and_evaluation(self):self.run_case('cuda')


if __name__=='__main__':unittest.main()
