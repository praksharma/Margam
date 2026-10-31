"""Accepted data fingerprints persist; inspection never re-accepts changed bytes."""
from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
import unittest
from margam.templates.poisson import PoissonConfig
from margam.ui.model_document import decode_document
from margam.ui.source_fingerprints import validate_fingerprints


class SourceFingerprintTests(unittest.TestCase):
    def run_ui(self,code):
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=90,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertNotIn('Traceback',result.stderr)

    def test_strict_bindings_and_legacy_model(self):
        source=dict(path='linked.csv',coordinates='x',normals='',values='')
        valid={'interior_data':dict(source=source,sha256='a'*64)}
        validate_fingerprints('interior',valid)
        for kind,records in [('network',valid),('interior',{'bad':next(iter(valid.values()))}),
                            ('interior',{'interior_data':dict(source=source,sha256='not a hash')}),
                            ('interior',{'interior_data':dict(source=dict(source,unexpected='x'),sha256='a'*64)})]:
            with self.subTest(kind=kind,records=records),self.assertRaises(ValueError):validate_fingerprints(kind,records)
        # Loading legacy declarations must not invent a checksum for older bytes.
        config,nodes,wires=decode_document(dict(format='margam-model',version=27,config=asdict(PoissonConfig()),nodes=[],wires=[]))
        self.assertEqual(nodes,[])

    def test_link_change_across_reopen_and_undo_reaccept(self):
        self.run_ui(r'''
from pathlib import Path
from tempfile import TemporaryDirectory
from copy import deepcopy
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from margam.ui.app import Window
from margam.ui.model_document import capture_model
app=QApplication([]);w=Window(state_directory=False);w.change('shape','Interval')
try:
 with TemporaryDirectory() as directory:
  directory=Path(directory);source=directory/'points.csv';source.write_text('x\n0.2\n0.8\n')
  def accept_link():
   d=app.activeModalWidget();d.fields['path'].setText(str(source));d.fields['coordinates'].setText('x');d.storage.choice.setCurrentIndex(1)
   assert d.check();d.accept_checked()
  QTimer.singleShot(0,accept_link);w.edit_point_data()
  owner=w.parameter_owner('interior_data');saved=deepcopy(owner.settings['data_fingerprints']['interior_data'])
  assert saved['source']['path']==str(source) and len(saved['sha256'])==64
  model=directory/'model.margam.json';assert w.save_model(model)
  source.write_text('x\n0.3\n0.7\n')
  assert w.load_model(model)
  view=w.show_project_data();assert view.records[0]['baseline_kind']=='Import'
  assert view.results[0]['status']=='Changed since import'
  before=capture_model(w);view.refresh();view.refresh();assert capture_model(w)==before
  assert w.validate_model() and any(i.severity=='Warning' and 'Changed since import' in i.message for i in w.last_validation_issues)
  # The same explicit mapping acceptance records current bytes and is one undo.
  index=w.undo_stack.index();QTimer.singleShot(0,accept_link);w.edit_point_data()
  assert w.undo_stack.index()==index+1
  view.refresh();assert view.results[0]['status']=='Available'
  owner=w.parameter_owner('interior_data');assert owner.settings['data_fingerprints']['interior_data']['sha256']!=saved['sha256']
  w.undo_edit();view.refresh();assert view.results[0]['status']=='Changed since import'
  w.redo_edit();view.refresh();assert view.results[0]['status']=='Available'
  w.clear_point_data();assert not owner.settings['data_fingerprints']
  w.undo_edit();view.refresh();assert view.results[0]['status']=='Available'
  # A metadata record for a different mapping must never be used for this one.
  alternate=directory/'alternate.csv';alternate.write_text('x\n0.1\n0.9\n')
  mapping=dict(w.config.interior_data,path=str(alternate));w.change('interior_data',mapping)
  view.refresh();assert view.records[0]['baseline_kind']=='Session'
finally:
 app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
''')

    def test_owned_sources_archive_and_saved_as_bindings(self):
        self.run_ui(r'''
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from margam.ui.app import Window
from margam.ui.model_document import capture_model
app=QApplication([]);w=Window(state_directory=False);w.load_evaluation_example();w.set_application_device('cpu')
def finish():
 deadline=time.monotonic()+30
 while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
 assert w.worker is None and w.result is not None,w.log.toPlainText()
try:
 with TemporaryDirectory() as directory:
  directory=Path(directory);points=next(n for n in w.scene.nodes() if n.kind=='evaluation_points')
  def copy_points():
   d=app.activeModalWidget();d.storage.choice.setCurrentIndex(0);assert d.check();d.accept_checked()
  QTimer.singleShot(0,copy_points);w.edit_evaluation_data(points)
  fingerprint=points.settings['data_fingerprints']['source_data']
  assert fingerprint['source']['path'].startswith('margam-owned:')
  w.change('iterations',2);w.change('layers',(4,4));assert w.start_training();finish()
  first=directory/'first.margam.json';second=directory/'second.margam.json'
  assert w.save_model(first) and w.save_model(second) and w.validate_model()
  view=w.show_project_data();index=next(i for i,r in enumerate(view.records) if r['reader']=='evaluation_points')
  assert view.records[index]['baseline_kind']=='Import' and view.results[index]['status']=='Available'
  archive=directory/'run.pt';assert w.save_current_run(archive)
  w.undo_stack.setClean();assert w.open_saved_run(archive)
  view.refresh();index=next(i for i,r in enumerate(view.records) if r['reader']=='evaluation_points')
  assert view.records[index]['baseline_kind']=='Import'
  points=next(n for n in w.scene.nodes() if n.kind=='evaluation_points');path=Path(w.project_workspace.resolve(points.settings['source_data']['path']))
  path.write_bytes(path.read_bytes()+b'\n');view.refresh();assert view.results[index]['status']=='Changed since import'
  assert w.save_model(directory/'restored.margam.json')
  assert w.load_model(directory/'restored.margam.json');view.refresh()
  index=next(i for i,r in enumerate(view.records) if r['reader']=='evaluation_points')
  assert view.results[index]['status']=='Changed since import' # Save/reload doesn't accept it.
finally:
 app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
''')
