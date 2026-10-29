"""Imported training sources, named-set inspectors, archives and export lifecycle."""
import os,subprocess,sys,unittest
from pathlib import Path


class TrainingDataUITests(unittest.TestCase):
    def test_triangle_blocks_training_archive_and_preflight(self):
        code=r"""import time,shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from copy import deepcopy
from PySide6.QtWidgets import QApplication,QPushButton,QComboBox
from margam.ui.app import Window
from margam.ui.model_document import capture_model,decode_document
from margam.ui.data_mapping import DataMappingDialog
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
def finish():
 deadline=time.monotonic()+30
 while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
 assert w.worker is None and w.result is not None,w.log.toPlainText()
try:
 assert w.load_training_data_example() and w.validate_model(),w.validation.text()
 nodes={n.kind:n for n in w.scene.nodes()}
 assert not w.output_views.entries and w.prepared_example_action in w.examples_menu.actions()
 assert w.scene.connection_error(nodes['boundary_data'].output,nodes['pde'].inputs['input'])
 c=w.scene.compile_config();assert c.shape=='Prepared2D' and c.sides==('base','upright','slope')
 w.show_block_properties(nodes['interior_data']);assert any('interior coordinates' in b.text() for b in w.property_scroll.findChildren(QPushButton))
 w.show_block_properties(nodes['boundary_data']);choice=w.property_scroll.findChild(QComboBox,'imported_boundary_set');assert choice.count()==3
 w.select_boundary_side('slope');assert w.boundary_side=='slope'
 w.show_block_properties(nodes['bc']);assert 'slope' in str(w.config.bc_types)
 dialog=DataMappingDialog(2,c.boundary_data['slope'],w);assert dialog.check();assert 'normals: True' in dialog.preview.toPlainText();dialog.deleteLater()
 w.clear_point_data('slope');assert 'slope' not in w.scene.compile_config().sides
 w.undo_stack.undo();assert 'slope' in w.scene.compile_config().sides
 w.undo_stack.redo();assert 'slope' not in w.scene.compile_config().sides
 w.undo_stack.undo();assert 'slope' in w.scene.compile_config().sides
 document=capture_model(w);assert document['version']==26;decode_document(document)
 with TemporaryDirectory() as directory:
  directory=Path(directory);path=directory/'triangle.margam.json';assert w.save_model(path);assert w.load_model(path)
  assert capture_model(w)==document
  local=directory/'inputs';shutil.copytree('assets/training/triangle',local)
  mappings=deepcopy(w.config.boundary_data)
  for source in mappings.values():source['path']=str(local/Path(source['path']).name)
  source=deepcopy(w.config.interior_data);source['path']=str(local/'interior.csv')
  w.change('interior_data',source);w.change('boundary_data',mappings)
  w.change('iterations',3);assert w.start_training();finish();assert w.result['steps']==3
  comparison=next(n for n in w.scene.nodes() if n.kind=='comparison');data=w.result['outputs'][comparison.identifier]
  assert len(data['points'])==58 and data['point_ids'][0]=='node-0'
  assert w.export_block_data(next(n for n in w.scene.nodes() if n.kind=='paraview'),directory/'triangle.vtu')
  archive=directory/'triangle.pt';assert w.save_current_run(archive);w.undo_stack.setClean();assert w.open_saved_run(archive)
  assert w.resume_training(additional=2);finish();assert w.result['steps']==5
  offline=directory/'offline.pt';assert w.save_current_run(offline)
  # A missing source and a missing required normal mapping fail before a worker starts.
  boundary=next(n for n in w.scene.nodes() if n.kind=='boundary_data');w.show_block_properties(boundary)
  mappings=deepcopy(w.config.boundary_data);mappings['slope']['normals']='';w.change('boundary_data',mappings)
  assert not w.validate_model() and w.worker is None
  w.undo_stack.undo();assert w.validate_model()
  mappings=deepcopy(w.config.boundary_data);mappings['base']['path']=str(directory/'missing.csv');w.change('boundary_data',mappings)
  assert not w.validate_model() and not w.start_training() and w.worker is None
  w.undo_stack.undo();assert w.validate_model()
  shutil.rmtree(local);w.undo_stack.setClean();assert w.open_saved_run(offline)
  assert w.export_block_data(next(n for n in w.scene.nodes() if n.kind=='export'),directory/'offline.csv')
  assert not w.validate_model()
finally:app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stderr+'\n'+result.stdout)
