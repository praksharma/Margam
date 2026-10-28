"""Individual data blocks and evaluation without additional optimizer steps."""
import os
from pathlib import Path
import subprocess,sys,unittest


class EvaluationUITests(unittest.TestCase):
    def test_individual_blocks_export_archive_and_reuse_weights(self):
        code=r"""import time,csv
from pathlib import Path
from tempfile import TemporaryDirectory
from copy import deepcopy
import torch
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.model_document import capture_model,decode_document
from margam.ui.evaluation_mapping import EvaluationMappingDialog
from margam.ui.output_views import VectorResultView
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
def finish():
 deadline=time.monotonic()+30
 while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
 assert w.worker is None and w.result is not None,w.log.toPlainText()
try:
 w.load_evaluation_example();assert w.validate_model(),w.validation.text()
 assert w.config.interior_data=={} and w.config.boundary_data=={}
 nodes={n.kind:n for n in w.scene.nodes()};points=nodes['evaluation_points'];ref=nodes['reference_data'];comparison=nodes['comparison'];export=nodes['export'];pv=nodes['paraview']
 assert w.scene.connection_error(points.output,nodes['interior'].inputs['input'])
 dialog=EvaluationMappingDialog(w.config,points.settings['source_data'],w);assert dialog.check();assert 'node-B' in dialog.preview.toPlainText();dialog.deleteLater()
 doc=capture_model(w);assert doc['version']==20;decode_document(doc)
 w.change('iterations',3);w.change('device','cpu');assert w.start_training();finish()
 model=w.last_run.model;before=model.training_state();data=w.result['outputs'][comparison.identifier]
 assert data['point_ids']==['node-B','node-A','node-D','node-C'] and data['points']==[[.75],[0.],[.25],[1.]]
 w.open_block(comparison);app.processEvents();view=w.output_views.entries[comparison.identifier][1];assert isinstance(view,VectorResultView)
 assert view.table.item(0,0).text()=='node-B' and view.table.rowCount()==4
 # Output-only edits also reuse a live model, without checkpoint reconstruction.
 w.change_output(comparison,'resolution',9);assert w.evaluate_trained_outputs();finish()
 assert w.last_run.model is model and model.iter==3
 for name,value in before['network'].items():torch.testing.assert_close(value,model.training_state()['network'][name],rtol=0,atol=0)
 with TemporaryDirectory() as directory:
  directory=Path(directory);assert w.export_block_data(export,directory/'out.csv');assert w.export_block_data(pv,directory/'out.vtu')
  with (directory/'out.csv').open() as stream:rows=list(csv.DictReader(stream))
  assert rows[0]['point_id']=='node-B' and 'error_u' in rows[0]
  archive=directory/'run.pt';assert w.save_current_run(archive);w.undo_stack.setClean();assert w.open_saved_run(archive)
  points=next(n for n in w.scene.nodes() if n.kind=='evaluation_points');comparison=next(n for n in w.scene.nodes() if n.kind=='comparison')
  # A new supplied point set must preserve weights and optimizer phase.
  mapping=directory/'new.csv';mapping.write_text('id,x\nnew-2,0.6\nnew-1,0.1\n')
  reference=directory/'ref.csv';reference.write_text('id,x,u\nnew-1,0.1,0.09\nnew-2,0.6,0.24\n')
  w.change_output(points,'source_data',dict(path=str(mapping),coordinates='x',ids='id',values=''))
  ref=next(n for n in w.scene.nodes() if n.kind=='reference_data');w.change_output(ref,'source_data',dict(path=str(reference),coordinates='x',ids='id',values='u'))
  reference.write_text('id,x,u\nnew-1,0.1,0.09\nnew-2,0.7,0.21\n')
  assert not w.validate_model() and w.worker is None # Coordinate mismatch fails before any work.
  reference.write_text('id,x,u\nnew-1,0.1,0.09\nnew-2,0.6,0.24\n')
  assert w.result is None and w.evaluate_trained_outputs();finish();assert w.result['steps']==3
  after=w.last_run.model.training_state();assert after['loss_history']==before['loss_history']
  for name,value in before['network'].items():torch.testing.assert_close(value,after['network'][name],rtol=0,atol=0)
  assert w.result['outputs'][comparison.identifier]['point_ids']==['new-2','new-1']
  assert 'no optimizer updates' in w.log.toPlainText()
  archive=directory/'new.pt';assert w.save_current_run(archive);mapping.unlink();reference.unlink();w.undo_stack.setClean();assert w.open_saved_run(archive)
  comparison=next(n for n in w.scene.nodes() if n.kind=='comparison');w.open_block(comparison);app.processEvents();assert w.output_views.entries[comparison.identifier][1].table.rowCount()==2
  assert w.export_block_data(next(n for n in w.scene.nodes() if n.kind=='export'),directory/'offline.csv')
  assert not w.validate_model() # Missing external files should be a validation message.
  # Changing core physics after reopening cannot use the old cached weights.
  w.change('source',3.);assert getattr(w,'_evaluation_model',None) is None and getattr(w,'_evaluation_checkpoint',None) is None
finally:
 app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stderr+'\n'+result.stdout)
