"""Managed snapshot cleanup protects referenced/undo data and skips foreign files."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import os
import subprocess
import sys
import unittest
from margam.ui.project_workspace import ProjectWorkspace
from margam.ui.dataset_cleanup import scan_snapshots,remove_snapshots,owned_references


def source(path):return dict(path=str(path),coordinates='x',values='',normals='')


def store(workspace,root,name='points.csv'):
    path=root/name;path.write_text('x\n0.2\n0.8\n')
    return workspace.store_dataset(source(path),copy=True,validator=lambda mapping:None)


@unittest.skipUnless(hasattr(os,'O_NOFOLLOW'),'Descriptor-based cleanup requires POSIX')
class DatasetCleanupTests(unittest.TestCase):
    def test_unused_and_protected_snapshots_only(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);w=ProjectWorkspace()
            try:
                self.assertEqual(scan_snapshots(w),[]);self.assertIsNone(w._temporary)
                a=store(w,root);b=store(w,root,'second.npz')
                folder=w.directory
                (folder/'domain.jorvik').write_text('keep geometry')
                other=folder/'datasets'/'manual';other.mkdir();(other/'notes.txt').write_text('keep manual data')
                selected=scan_snapshots(w);self.assertEqual(len(selected),2)
                reasons={b['path']:'Current model'}
                removed,kept=remove_snapshots(w,selected,reasons)
                self.assertEqual(len(removed),1);self.assertEqual(len(kept),1)
                self.assertFalse(Path(w.resolve(a['path'],create=False)).exists())
                self.assertTrue(Path(w.resolve(b['path'],create=False)).exists())
                self.assertTrue((folder/'domain.jorvik').exists());self.assertTrue((other/'notes.txt').exists())
                self.assertTrue((root/'points.csv').exists())
            finally:w.close()

    def test_changed_files_symlinks_hardlinks_and_unknown_layouts_kept(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);w=ProjectWorkspace()
            try:
                a=store(w,root);row=scan_snapshots(w)[0];path=Path(w.resolve(a['path']))
                path.write_text('x\n0.5\n');removed,kept=remove_snapshots(w,[row],{})
                self.assertFalse(removed);self.assertEqual(kept,[row.relative]);self.assertTrue(path.exists())
                row=scan_snapshots(w)[0];external=root/'external.csv';external.write_text('do not delete')
                path.unlink();path.symlink_to(external)
                self.assertEqual(scan_snapshots(w),[]);self.assertFalse(remove_snapshots(w,[row],{})[0]);self.assertEqual(external.read_text(),'do not delete')
                path.unlink();os.link(external,path);self.assertEqual(scan_snapshots(w),[])
                path.unlink();path.write_text('x\n0.5\n');(path.parent/'extra.txt').write_text('foreign file')
                self.assertEqual(scan_snapshots(w),[])
                # Replacing the import directory with a symlink never follows it.
                (path.parent/'extra.txt').unlink();path.unlink();parent=path.parent;parent.rmdir();parent.symlink_to(root,target_is_directory=True)
                self.assertEqual(scan_snapshots(w),[]);self.assertEqual(external.read_text(),'do not delete')
            finally:w.close()

    def test_references_are_paths_and_ambiguous_paths_block_cleanup(self):
        w=ProjectWorkspace()
        try:
            uri='margam-owned:datasets/'+'a'*32+'/points.csv'
            self.assertEqual(owned_references(dict(label=uri,annotation_text=uri),w),set())
            self.assertEqual(owned_references(dict(path=uri),w),{uri})
            self.assertEqual(owned_references(dict(data_fingerprints={'old':dict(source=dict(path=uri))}),w),set())
            with self.assertRaises(ValueError):owned_references(dict(path='margam-owned:a/../datasets/file.csv'),w)
            self.assertIsNone(w._temporary)
        finally:w.close()


class DatasetCleanupUITests(unittest.TestCase):
    def test_dropped_history_and_redo_branch_release_old_imports(self):
        code=r'''
from pathlib import Path
from tempfile import TemporaryDirectory
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.dataset_cleanup import protected_references,history_references
app=QApplication([]);w=Window(state_directory=False);w.change('shape','Interval');w.undo_stack.clear();w.undo_stack.setUndoLimit(2)
try:
 with TemporaryDirectory() as directory:
  root=Path(directory);sources=[]
  for number in range(3):
   path=root/f'{number}.csv';path.write_text('x\\n0.2\\n0.8\\n')
   sources.append(w.project_workspace.store_dataset(dict(path=str(path),coordinates='x',values='',normals=''),copy=True,validator=lambda source:None))
  a,b,c=sources
  w.change('interior_data',a);w.change('interior_data',b);w.change('interior_data',c)
  assert protected_references(w)[a['path']]=='Undo / Redo'
  w.change('source',3);assert a['path'] not in protected_references(w) and protected_references(w)[b['path']]=='Undo / Redo'
  w.undo_edit();w.undo_edit();assert protected_references(w)[c['path']]=='Undo / Redo'
  w.change('source',4);assert c['path'] not in protected_references(w)
  assert protected_references(w)[b['path']]=='Current model'
  w.undo_stack.clear();token=object()
  w.change('interior_data',a,token);w.change('interior_data',c,token)
  assert w.undo_stack.count()==1
  refs=history_references(w.undo_stack);assert {a['path'],b['path'],c['path']}<=refs
  w.undo_edit();assert protected_references(w)[c['path']]=='Undo / Redo'
finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=45,
            env=dict(os.environ,QT_QPA_PLATFORM='offscreen',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+'\\n'+result.stderr)

    def test_undo_redo_macro_saved_files_cached_state_and_stale_review(self):
        code=r'''
from pathlib import Path
from tempfile import TemporaryDirectory
from copy import deepcopy
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QUndoCommand
from margam.ui.app import Window
from margam.ui.dataset_cleanup import protected_references,scan_snapshots,remove_snapshots,history_references
from margam.ui.model_document import capture_model
app=QApplication([]);w=Window(state_directory=False);w.set_application_device('cpu');w.change('shape','Interval')
def mapping(path):return dict(path=str(path),coordinates='x',values='',normals='')
def store(root,number):
 path=root/f'points{number}.csv';path.write_text('x\n0.2\n0.8\n')
 return w.project_workspace.store_dataset(mapping(path),copy=True,validator=lambda source:None)
try:
 with TemporaryDirectory() as directory:
  root=Path(directory);a=store(root,1);b=store(root,2)
  w.change('interior_data',a);w.undo_stack.beginMacro('Replace data');w.change('interior_data',b);w.undo_stack.endMacro()
  refs=history_references(w.undo_stack);assert a['path'] in refs and b['path'] in refs
  reasons=protected_references(w);assert reasons[a['path']]=='Undo / Redo' and reasons[b['path']]=='Current model'
  w.undo_edit();assert protected_references(w)[b['path']]=='Undo / Redo'
  w.redo_edit();w.undo_stack.clear()
  # A cached trained/evaluation configuration keeps its actual input file.
  w._evaluation_config=deepcopy(w.config);w._evaluation_config.interior_data=a
  assert protected_references(w)[a['path']]=='Cached run';w._evaluation_config=None
  # Save baseline A, then change to B and discard history: saved A is still protected.
  w.change('interior_data',a);path=root/'model.margam.json';assert w.save_model(path)
  w.change('interior_data',b);w.undo_stack.clear()
  assert protected_references(w)[a['path']]=='Saved model'
  assert w.save_model(path);w.undo_stack.clear()
  reasons=protected_references(w);assert a['path'] not in reasons
  dialog=w.show_dataset_cleanup();assert len(dialog.rows)==2
  original=capture_model(w);index=w.undo_stack.index();result={'outputs':{},'keep':True};w.result=result
  dialog.select_unused();assert dialog.remove_button.isEnabled()
  # A new mapping after review must be protected at removal time.
  w.change('interior_data',a);assert not dialog.remove_selected()
  assert Path(w.project_workspace.resolve(a['path'],create=False)).exists()
  w.undo_edit();w.undo_stack.clear();dialog.refresh();dialog.select_unused()
  w.set_busy(True);assert not dialog.remove_selected();w.set_busy(False)
  # Remove only the now-unused snapshot; current model and training data stay intact.
  before=capture_model(w);index=w.undo_stack.index();w.result=result
  assert dialog.remove_selected()
  assert capture_model(w)==before and w.undo_stack.index()==index and w.result is result
  assert not Path(w.project_workspace.resolve(a['path'],create=False)).exists()
  assert Path(w.project_workspace.resolve(b['path'],create=False)).exists()
  assert (root/'points1.csv').exists() and (root/'points2.csv').exists()
  # Unknown third-party commands refuse cleanup rather than guessing their references.
  w.undo_stack.push(QUndoCommand('Unknown external edit'));dialog.refresh();assert 'no file-reference record' in dialog.status.text()
  w.undo_stack.clear()
  # Project replacement makes an old review stale, without touching its files.
  c=store(root,3);dialog.refresh();dialog.select_unused();owned=Path(w.project_workspace.resolve(c['path'],create=False))
  w.replace_model(capture_model(w),path);assert not dialog.remove_selected() and owned.exists()
  # Empty workspaces are inspected without making an owned directory.
  empty=Window(state_directory=False)
  assert not empty.show_dataset_cleanup().rows and empty.project_workspace._temporary is None
  empty.undo_stack.setClean();empty.close()
finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=90,
            env=dict(os.environ,QT_QPA_PLATFORM='offscreen',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertNotIn('Traceback',result.stderr)
