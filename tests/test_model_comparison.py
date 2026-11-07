"""Read-only saved-definition differences and desktop comparison lifecycle."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from margam.templates.poisson import PoissonConfig
from margam.ui.graph_model import parameters_for
from margam.ui.model_diff import compare_models,ABSENT,value_text


def document():
    def node(identifier,kind):
        return dict(id=identifier,kind=kind,title=kind,position=[0,0],settings=dict(
            resolution=41,history_limit=10000,parameters=parameters_for(kind)))
    return dict(format='margam-model',version=30,config=asdict(PoissonConfig()),
        nodes=[node('domain','domain'),node('sampler','interior'),node('sampler2','interior'),node('pde','pde')],
        wires=[dict(source='domain',target='sampler',input='input'),dict(source='sampler',target='pde',input='input')])


class ModelDifferenceTests(unittest.TestCase):
    def test_owned_nested_settings_and_connection_changes(self):
        a=document();b=deepcopy(a)
        b['nodes'][1]['settings']['parameters']['points']=81
        b['nodes'][3]['settings']['parameters']['source']=3
        b['wires'][1]['source']='sampler2'
        changes=compare_models(a,b)
        self.assertEqual({(x.category,x.node_id,x.field) for x in changes},{
            ('Setting','sampler','/settings/parameters/points'),('Setting','pde','/settings/parameters/source'),('Connection','pde','/inputs/input')})
        wire=next(x for x in changes if x.category=='Connection')
        self.assertEqual(wire.before['id'],'sampler');self.assertEqual(wire.after['id'],'sampler2')
        self.assertEqual(compare_models(a,a),[])

    def test_stable_ids_add_remove_and_literal_values(self):
        a=document();b=deepcopy(a)
        b['nodes'][1]['id']='replacement';b['wires'][0]['target']='replacement';b['wires'][1]['source']='replacement'
        changes=compare_models(a,b)
        self.assertEqual(sum(x.category=='Added block' for x in changes),1)
        self.assertEqual(sum(x.category=='Removed block' for x in changes),1)
        removed=next(x for x in changes if x.category=='Removed block');self.assertIs(removed.after,ABSENT)
        self.assertEqual(value_text(ABSENT),'(absent)')
        self.assertIn('<b>',value_text('<b>literal</b>'))

    def test_presentation_optional_and_inspector_cache_ignored(self):
        a=document();b=deepcopy(a)
        b['nodes'][1]['title']='Research sampling';b['nodes'][1]['position']=[45,90]
        b['config']['points']=900
        self.assertEqual(compare_models(a,b),[])
        changes=compare_models(a,b,include_presentation=True)
        self.assertEqual({x.field for x in changes},{'/title','/position'})
        note=dict(id='note',kind='annotation',title='Note',position=[0,0],settings=dict(resolution=41,history_limit=10000,annotation_text='<b>text</b>'))
        b['nodes'].append(note)
        self.assertEqual(compare_models(a,b),[])
        self.assertEqual(len(compare_models(a,b,include_presentation=True)),3)
        a=deepcopy(b);b['nodes'][-1]['settings']['annotation_text']='New note'
        self.assertEqual(compare_models(a,b),[])
        self.assertEqual(compare_models(a,b,include_presentation=True)[0].category,'Presentation')

    def test_migration_and_input_immutability(self):
        a=document();b=deepcopy(a);b['version']=29;b['config'].pop('inverse_settings')
        original=deepcopy(b)
        self.assertEqual(compare_models(a,b),[]);self.assertEqual(b,original)
        old=document();old['version']=1
        for node in old['nodes']:node['settings'].pop('parameters')
        self.assertEqual(compare_models(document(),old),[])

    def test_file_references_compared_without_reading_data(self):
        a=document();b=deepcopy(a)
        b['nodes'][1]['settings']['parameters']['interior_data']=dict(path='missing.csv',coordinates='x,y',values='',normals='')
        changes=compare_models(a,b)
        self.assertTrue(any(x.field.endswith('/interior_data/path') for x in changes))
        self.assertFalse(Path('missing.csv').exists())
        invalid=deepcopy(a);invalid['version']=99
        with self.assertRaises(ValueError):compare_models(a,invalid)


class ModelComparisonUITests(unittest.TestCase):
    def test_compare_two_files_current_navigation_and_no_mutation(self):
        code=r'''
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from PySide6.QtWidgets import QApplication,QFileDialog
from margam.ui.app import Window
from margam.ui.model_document import capture_model,write_model
app=QApplication([]);w=Window(state_directory=False);w.set_application_device('cpu')
try:
 with TemporaryDirectory() as directory:
  root=Path(directory);a=root/'baseline.json';b=root/'candidate.json';bad=root/'bad.json'
  baseline=capture_model(w);write_model(a,baseline)
  w.change('iterations',2);w.change('layers',(4,));assert w.start_training()
  while w.worker is not None:app.processEvents()
  assert w.result is not None
  current=capture_model(w);write_model(b,current);bad.write_text('{}')
  before=deepcopy(current);index=w.undo_stack.index();result=w.result;run=w.last_run;model=run.model
  state=model.training_state();clean=w.undo_stack.isClean();files={p.name:p.read_bytes() for p in root.iterdir()}
  dialog=w.show_model_comparison();assert dialog.set_file(a)
  assert dialog.changes and dialog.comparison_path is None
  dialog.search.setText('iterations');assert len(dialog.visible_changes)==1
  assert chr(92)+'nBlock ID:' not in dialog.details.toPlainText()
  assert chr(10)+'Block ID:' in dialog.details.toPlainText()
  assert dialog.locate();assert w.name.text()
  assert capture_model(w)==before and w.undo_stack.index()==index and w.undo_stack.isClean()==clean
  assert w.result is result and w.last_run is run and w.last_run.model is model
  assert state['loss_history']==model.training_state()['loss_history']
  assert w.project_workspace._temporary is None
  assert dialog.set_file(b,comparison=True);assert not dialog.locate_button.isEnabled() and not dialog.locate()
  saved_path=dialog.baseline_path;assert not dialog.set_file(bad);assert dialog.baseline_path==saved_path
  w.set_busy(True);dialog.refresh();assert dialog.changes;w.set_busy(False)
  with patch.object(QFileDialog,'getOpenFileName',return_value=('','')):dialog.choose_file()
  assert dialog.baseline_path==saved_path
  # Refresh rereads saved bytes; missing files report an inline error and clear rows.
  a.unlink();dialog.refresh();assert 'unavailable' in dialog.status.text() and not dialog.changes
  write_model(a,baseline);dialog.use_current();dialog.search.clear();assert dialog.changes
  # Stale IDs are resolved at navigation time, not retained as Qt objects.
  dialog.search.setText('iterations');target=dialog.visible_changes[0].node_id
  node=next(n for n in w.scene.nodes() if n.identifier==target)
  w.scene.clearSelection();node.setSelected(True);w.delete_selected()
  assert not dialog.locate() and 'no longer exists' in dialog.details.toPlainText()
  dialog.refresh()
  assert {p.name:p.read_bytes() for p in root.iterdir()}==files
  dialog.close()
finally:
 w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=90,
            env=dict(os.environ,QT_QPA_PLATFORM='offscreen',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
