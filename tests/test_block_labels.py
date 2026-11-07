"""Presentation-only block labels retain type, identity and trained state."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class BlockLabelTests(unittest.TestCase):
    def run_script(self,code):
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=120,
            env=dict(os.environ,QT_QPA_PLATFORM='offscreen',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertNotIn('Traceback',result.stderr)

    def test_labels_preserve_results_and_survive_files_archives_and_copy(self):
        self.run_script(r'''
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication,QLineEdit,QInputDialog,QLabel
from margam.ui.app import Window
from margam.ui.model_document import capture_model,decode_document
from margam.ui.model_diff import compare_models
app=QApplication([]);w=Window(state_directory=False);w.set_application_device('cpu')
try:
 w.load_heat_template();w.change('iterations',2);w.change('layers',(4,));assert w.start_training()
 while w.worker is not None:app.processEvents()
 assert w.result is not None
 network=next(n for n in w.scene.nodes() if n.kind=='network');monitor=next(n for n in w.scene.nodes() if n.kind=='monitor')
 w.output_views.open(monitor);before=capture_model(w);config=w.scene.compile_config();result=w.result;run=w.last_run;state=run.model.training_state()
 index=w.undo_stack.index();identifier=network.identifier;kind=network.kind;parameters=deepcopy(network.settings['parameters'])
 w.scene.clearSelection();network.setSelected(True)
 assert w.rename_block(network,'<b>Research NN</b>')
 assert network.title=='<b>Research NN</b>' and network.kind==kind and network.identifier==identifier and network.type_title=='Neural network'
 assert '&lt;b&gt;' in network.toolTip() and identifier in network.toolTip() and '[network]' in network.toolTip()
 assert '&lt;b&gt;' in next(wire for wire in w.scene.wires if wire.source.parentItem() is network).toolTip()
 assert w.undo_stack.index()==index+1 and network.settings['parameters']==parameters
 assert w.result is result and w.last_run is run and state['loss_history']==run.model.training_state()['loss_history']
 assert w.scene.compile_config()==config
 after=capture_model(w);assert after['version']==31;decode_document(after)
 assert compare_models(before,after)==[]
 changes=compare_models(before,after,include_presentation=True);assert len(changes)==1 and changes[0].field=='/settings/label'
 headings=[x for x in w.property_scroll.widget().findChildren(QLabel) if x.objectName()=='block_heading']
 assert headings[0].text()=='<b>Research NN</b>' and headings[0].textFormat()==Qt.TextFormat.PlainText
 w.undo_edit();assert network.title=='Neural network' and 'label' not in network.settings and w.result is result
 w.redo_edit();assert network.title=='<b>Research NN</b>' and w.result is result
 assert w.rename_output(monitor,'Convergence');assert w.output_views.entries[monitor.identifier][0].windowTitle()=='Convergence'
 assert monitor.result_status=='Current' and w.result is result
 search=w.show_model_search();search.search.setText('Research NN');assert search.results.count()==1;search.close()
 settings=w.show_settings_search();settings.search.setText('block label');assert settings.results.count()>=len(w.scene.nodes());settings.close()
 assert not w.rename_block(network,'x'*257) and not w.rename_block(network,'bad\nline')
 w.set_busy(True);assert not w.rename_block(network,'Blocked');w.set_busy(False)
 with patch.object(QInputDialog,'getText',return_value=('Discard me',False)):assert not w.request_block_label(network)
 # Mapping-looking label text must not be treated as a project-owned file.
 assert w.rename_block(network,'margam-owned:Research NN')
 w.scene.clearSelection();network.setSelected(True);assert w.copy_blocks();assert w.paste_blocks()
 pasted=next(n for n in w.scene.nodes() if n.isSelected());assert pasted.identifier!=identifier and pasted.title==network.title and pasted.settings['label']==network.settings['label']
 w.undo_edit() # remove disconnected copy before persistence checks
 # Labels alone preserve a new run's data and checkpoint identity.
 w.undo_stack.setClean()
 with TemporaryDirectory() as directory:
  path=Path(directory)/'labels.margam.json';assert w.save_model(path)
  assert w.load_model(path);restored=next(n for n in w.scene.nodes() if n.identifier==identifier)
  assert restored.title=='margam-owned:Research NN' and restored.settings['parameters']==parameters
  assert w.start_training()
  while w.worker is not None:app.processEvents()
  archive=Path(directory)/'labels.pt';assert w.save_current_run(archive);assert w.open_saved_run(archive)
  assert next(n for n in w.scene.nodes() if n.identifier==identifier).title=='margam-owned:Research NN'
  assert next(n for n in w.scene.nodes() if n.identifier==monitor.identifier).title=='Convergence'
finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
''')

    def test_automatic_type_names_reset_recipes_notes_and_schema(self):
        self.run_script(r'''
from copy import deepcopy
from PySide6.QtWidgets import QApplication,QLineEdit
from margam.ui.app import Window
from margam.ui.model_document import capture_model,decode_document
from margam.ui.block_labels import valid_label
app=QApplication([]);w=Window(state_directory=False)
try:
 domain=next(n for n in w.scene.nodes() if n.kind=='domain')
 assert w.rename_block(domain,'Rectangle') # explicit label equal to current automatic name
 w.change('shape','Interval');assert domain.title=='Rectangle' and domain.type_title=='Interval'
 assert w.rename_block(domain,'');assert domain.title=='Interval' and 'label' not in domain.settings
 w.undo_edit();assert domain.title=='Rectangle' and domain.settings['label']=='Rectangle'
 w.redo_edit();assert domain.title=='Interval'
 # Editing the inspector changes a label in one Undo step.
 w.scene.clearSelection();domain.setSelected(True)
 edit=next(e for e in w.property_scroll.widget().findChildren(QLineEdit) if e.property('search_key')=='block_label')
 index=w.undo_stack.index();edit.setText('Study domain');edit.editingFinished.emit()
 assert domain.title=='Study domain' and w.undo_stack.index()==index+1
 method=next(n for n in w.scene.nodes() if n.kind=='method');network=next(n for n in w.scene.nodes() if n.kind=='network')
 w.rename_block(method,'Experiment A');w.rename_block(network,'Trial network')
 compact=w.collapse_recipe(method);assert compact.title=='Experiment A' and compact.settings['network_label']=='Trial network'
 document=capture_model(w);decode_document(document)
 w.replace_model(document);compact=next(n for n in w.scene.nodes() if n.kind=='recipe')
 expanded=w.expand_recipe(compact);assert expanded.title=='Experiment A'
 assert next(n for n in w.scene.nodes() if n.kind=='network').title=='Trial network'
 note=w.add_node('Note');w.rename_block(note,'Assumptions');w.change_note(note,'Use consistent units.')
 assert w.name.text()=='Assumptions' and 'Assumptions' in note.toolTip();decode_document(capture_model(w))
 doc=capture_model(w)
 for key,value in (('label',''),('label','x'*257),('label','two\nlines'),('network_label','bad')):
  bad=deepcopy(doc);record=next(n for n in bad['nodes'] if n['kind']=='domain');record['settings'][key]=value
  try:decode_document(bad)
  except ValueError:pass
  else:raise AssertionError('Invalid metadata accepted')
 legacy=deepcopy(doc);legacy['version']=30
 for record in legacy['nodes']:
  record['settings'].pop('label',None);record['settings'].pop('network_label',None)
 decode_document(legacy)
 assert valid_label('Research α') and not valid_label('bad\nline')
 # A stale Python reference must not edit a deleted native window during teardown.
 from PySide6.QtCore import QCoreApplication,QEvent
 deleted=Window(state_directory=False);old_node=deleted.scene.nodes()[0]
 deleted.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
 assert not deleted.rename_block(old_node,'Too late')
finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
''')
