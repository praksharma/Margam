"""Owned clipboard files are bounded, independent and safe across project closure."""
import json,os
from pathlib import Path
import subprocess,sys,tempfile,unittest
from margam.ui.clipboard_files import capture_files,encode_files,decode_files,rebase,install,read_owned,PREFIX,FILE_LIMIT
from margam.ui.project_workspace import ProjectWorkspace


class ClipboardFileTests(unittest.TestCase):
    def test_snapshot_rebase_and_literal_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            w=ProjectWorkspace(Path(folder)/'model.margam.json');root=w.directory
            path=root/'datasets'/'original.csv';path.parent.mkdir();path.write_text('x\n0.2\n')
            uri=PREFIX+'datasets/original.csv'
            node=dict(id='source',kind='interior_data',settings=dict(label=uri,parameters=dict(interior_data=dict(path=uri)),data_fingerprints=dict(interior_data=dict(source=dict(path=uri),sha256='a'*64))))
            doc=dict(nodes=[node]);files=capture_files(doc,w);content=json.dumps(doc).encode();bundle=encode_files(content,files)
            captured=decode_files(doc,content,bundle);path.write_text('changed')
            destination=ProjectWorkspace();rebased,prefix=rebase(doc,captured);install(destination,captured,prefix)
            settings=rebased['nodes'][0]['settings'];mapping=settings['parameters']['interior_data']['path']
            self.assertEqual(settings['label'],uri);self.assertNotEqual(mapping,uri)
            self.assertEqual(settings['data_fingerprints']['interior_data']['source']['path'],mapping)
            self.assertEqual(Path(destination.resolve(mapping)).read_text(),'x\n0.2\n')
            other,other_prefix=rebase(doc,captured);self.assertNotEqual(prefix,other_prefix)
            destination.close()

    def test_rejects_missing_mismatched_extra_or_traversal_payloads(self):
        doc=dict(nodes=[dict(kind='interior_data',settings=dict(parameters=dict(interior_data=dict(path=PREFIX+'data.csv'))))]);content=json.dumps(doc).encode()
        for bundle in (b'',encode_files(b'other',{'data.csv':b'hi'}),encode_files(content,{}),encode_files(content,{'data.csv':b'hi','extra.csv':b'no'})):
            with self.assertRaises(ValueError):decode_files(doc,content,bundle)
        doc['nodes'][0]['settings']['parameters']['interior_data']['path']=PREFIX+'../escape'
        with self.assertRaises(ValueError):decode_files(doc,b'',b'')
        good=dict(nodes=[dict(kind='annotation',settings=dict(annotation_text=PREFIX+'../literal'))])
        self.assertEqual(decode_files(good,b'',b''),{})

    def test_missing_links_budget_and_locked_copy_do_not_create_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'data').mkdir();(root/'outside').write_text('outside')
            (root/'data'/'link').symlink_to(root/'outside')
            with self.assertRaises(OSError):read_owned(root,'data/link')
            (root/'alias').symlink_to(root/'data',target_is_directory=True)
            with self.assertRaises(OSError):read_owned(root,'alias/link')
            with (root/'big').open('wb') as f:f.truncate(FILE_LIMIT+1)
            with self.assertRaises(ValueError):read_owned(root,'big')
            w=ProjectWorkspace(root/'missing.margam.json')
            doc=dict(nodes=[dict(kind='interior_data',settings=dict(parameters=dict(interior_data=dict(path=PREFIX+'missing.csv'))))])
            with self.assertRaises(ValueError):capture_files(doc,w,locked=True)
            with self.assertRaises(FileNotFoundError):capture_files(doc,w)
            self.assertFalse(w.companion(w.model_path).exists())

    def test_blank_geometry_does_not_create_workspace_but_missing_applied_document_fails(self):
        w=ProjectWorkspace();doc=dict(nodes=[dict(id='blank',kind='jorvik_geometry',settings={})])
        self.assertEqual(capture_files(doc,w),{});self.assertIsNone(w._temporary)
        doc['nodes'][0]['settings']['jorvik']={'sha256':'a'*64}
        with self.assertRaisesRegex(ValueError,'missing'):capture_files(doc,w)
        self.assertIsNone(w._temporary)

    def test_failed_install_rolls_back_and_symbolic_destination_is_rejected(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            w=ProjectWorkspace(Path(folder)/'model.margam.json');root=w.directory
            with patch('margam.ui.clipboard_files.os.fdopen',side_effect=OSError('injected disk failure')):
                with self.assertRaises(OSError):install(w,{'data.csv':b'rows'},'clipboard/failed/')
            self.assertFalse((root/'clipboard'/'failed').exists())
            outside=Path(folder)/'outside';outside.mkdir()
            (root/'clipboard').rmdir();(root/'clipboard').symlink_to(outside,target_is_directory=True)
            with self.assertRaises(OSError):install(w,{'data.csv':b'rows'},'clipboard/blocked/')
            self.assertEqual(list(outside.iterdir()),[])

    def test_real_jorvik_cross_project_paste_undo_save_and_closed_source(self):
        code=r'''
import tempfile
from pathlib import Path
from unittest.mock import patch
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.components.prepared_data import read_points
from margam.ui.model_document import capture_model
from margam.ui.clipboard_files import MIME
from PySide6.QtCore import QMimeData
from jorvik.project import load_project
from test_jorvik import fixture,SELECTION
app=QApplication([]);a=Window(state_directory=False);b=Window(state_directory=False)
a.show();b.show();app.processEvents()
g=a.add_node('Jorvik geometry');original=a.jorvik_document(g,create=True);fixture(original.with_suffix('.nc'))
with patch('margam.ui.jorvik_geometry.export_saved_project',return_value=3):assert a.update_jorvik(g,SELECTION,confirm=False)
a.scene.clearSelection()
for node in a.scene.nodes():
    if node.kind in ('jorvik_geometry','interior_data','boundary_data'):node.setSelected(True)
assert a.copy_blocks()
# Paste uses a clipboard byte snapshot even after the source workspace closes.
a.undo_stack.setClean();a.close();app.processEvents();assert not original.exists()
assert b.paste_blocks()
nodes=[n for n in b.scene.nodes() if n.isSelected()];geometry=next(n for n in nodes if n.kind=='jorvik_geometry')
interior=next(n for n in nodes if n.kind=='interior_data');boundary=next(n for n in nodes if n.kind=='boundary_data')
assert len(nodes)==3 and len([w for w in b.scene.wires if w.source.parentItem() in nodes and w.target.parentItem() in nodes])==2
path=b.jorvik_document(geometry);load_project(path)
assert b.jorvik_export(geometry)==path.with_suffix('.nc') and b.jorvik_export(geometry).exists()
assert 'export' not in geometry.settings['jorvik_files'] # Do not turn adjacent export into an override.
source=interior.settings['parameters']['interior_data'];points=read_points(b.project_workspace.resolve_source(source),2)[0];assert len(points)==3
assert set(boundary.settings['parameters']['boundary_data'])=={'base','wall'}
snapshot=Path(b.project_workspace.resolve(source['path']));before=snapshot.read_bytes()
b.undo_edit();assert all(n.scene() is None for n in nodes) and snapshot.exists()
b.redo_edit();assert all(n.scene() is b.scene for n in nodes) and snapshot.read_bytes()==before
# Bad/missing snapshots must not change the graph, history or owned files.
old=capture_model(b);index=b.undo_stack.index();before_paths=set(b.project_workspace.directory.rglob('*'))
mime=QMimeData();mime.setData('application/x-margam-model-fragment',app.clipboard().mimeData().data('application/x-margam-model-fragment'))
app.clipboard().setMimeData(mime);assert not b.paste_blocks()
assert capture_model(b)==old and b.undo_stack.index()==index and set(b.project_workspace.directory.rglob('*'))==before_paths
b.scene.clearSelection();geometry.setSelected(True);assert b.cut_blocks();assert geometry.scene() is None and path.exists()
b.undo_edit();assert geometry.scene() is b.scene
assert b.paste_blocks()
second=next(n for n in b.scene.nodes() if n.isSelected());second_path=b.jorvik_document(second)
assert second_path!=path and second_path.read_bytes()==path.read_bytes()
second_path.write_bytes(b'independent');assert path.read_bytes()!=second_path.read_bytes()
# A new paste is independent from subsequent edits to a previous paste too.
assert b.paste_blocks();third=next(n for n in b.scene.nodes() if n.isSelected());load_project(b.jorvik_document(third))
with tempfile.TemporaryDirectory() as folder:
    saved=Path(folder)/'copy.margam.json';assert b.save_model(saved)
    assert b.load_model(saved)
    restored=next(n for n in b.scene.nodes() if n.identifier==geometry.identifier)
    load_project(b.jorvik_document(restored))
app.clipboard().clear();b.undo_stack.setClean();b.close();app.processEvents()
'''
        root=Path(__file__).resolve().parents[1]
        result=subprocess.run([sys.executable,'-c',code],cwd=root,env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONPATH=str(root/'tests'),PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
