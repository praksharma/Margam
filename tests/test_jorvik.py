"""Unequal-size NetCDF datasets, planar validation and owned-document UI lifecycle."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from margam.components.jorvik_data import inspect_selection
from margam.components.prepared_data import read_points


def fixture(path, *, nonplanar=False, normal_kind='outward_conormal'):
    from jorvik.export import export_samples
    batches=[]
    for label,points,dimension,normal in [
        ('interior',[[.2,.2,0],[.3,.4,0],[.6,.2,0]],2,None),
        ('base',[[.2,0,0],[.5,0,0]],1,[0,-1,0]),
        ('wall',[[0,.1,0],[0,.3,0],[0,.6,0],[0,.9,0]],1,[-1,0,0])]:
        coordinates=np.array(points,dtype=float)
        if nonplanar and label=='wall':coordinates[-1,2]=.2
        batch=dict(document='fixture',source_hash='a'*64,label=label,coordinates=coordinates,
                   entity_tags=np.ones(len(points),dtype=np.int32),entity_dimension=dimension,
                   entities=[[dimension,1]],count=len(points),method='sobol',seed=42)
        if normal:
            batch.update(normals=np.tile(normal,(len(points),1)),normal_status='available',normal_kind=normal_kind,
                         normal_validation='fixture',boundary_region={'id':label+'-stable'})
        batches.append(batch)
    export_samples(batches,path)
    return batches


SELECTION={'interior':('interior','interior'),'base':('boundary','base'),'wall':('boundary','wall')}


class JorvikDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:import jorvik.export
        except ImportError:raise unittest.SkipTest('Install optional Jorvik integration to run exchange fixtures.')

    def test_unequal_sizes_and_normals(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'domain.nc';fixture(path)
            report=inspect_selection(path,SELECTION)
            self.assertEqual(read_points(report['interior'],2)[0].shape,(3,2))
            self.assertEqual(read_points(report['boundaries']['base']['mapping'],2)[0].shape,(2,2))
            points,normals,_=read_points(report['boundaries']['wall']['mapping'],2)
            self.assertEqual(points.shape,(4,2));np.testing.assert_array_equal(normals,np.tile([-1,0],(4,1)))
            self.assertEqual(report['boundaries']['base']['identity'],'base-stable')

    def test_export_saved_project_and_missing_samples(self):
        from unittest.mock import patch
        from margam.components.jorvik_data import export_saved_project
        from jorvik.project import create_empty_project
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'domain.nc';batches=fixture(path)
            path.unlink()
            manifest={'areas':[dict(status='Generated',area=batch) for batch in batches]}
            with patch('jorvik.project.load_project',return_value=(manifest,None)):
                self.assertEqual(export_saved_project('saved.jorvik',path),3)
            report=inspect_selection(path,SELECTION)
            self.assertEqual(len(report['boundaries']),2)
            original=path.read_bytes()
            empty=Path(directory)/'empty.jorvik';create_empty_project(empty)
            with self.assertRaisesRegex(ValueError,'No generated samples'):export_saved_project(empty,path)
            self.assertEqual(path.read_bytes(),original)

    def test_nonplanar_and_invalid_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'domain.nc';fixture(path,nonplanar=True)
            with self.assertRaisesRegex(ValueError,'axis-aligned'):inspect_selection(path,SELECTION)
            fixture(path)
            with self.assertRaisesRegex(ValueError,'exactly one'):inspect_selection(path,{'base':('boundary','base')})
            bad=dict(SELECTION,wall=('boundary','base'))
            with self.assertRaisesRegex(ValueError,'unique'):inspect_selection(path,bad)
            with self.assertRaises(ValueError):inspect_selection(path,SELECTION,(0,0))

    def test_ui_update_save_move_undo_and_failure(self):
        code=r'''
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from unittest.mock import patch
from PySide6.QtWidgets import QApplication,QMessageBox
from margam.ui.app import Window
from margam.ui.model_document import capture_model,read_model,decode_document
from margam.components.training_data import read_training_sets
from tests.test_jorvik import fixture,SELECTION
app=QApplication([]);w=Window(state_directory=False)
with TemporaryDirectory() as directory:
    w.reset_template()
    for node in list(w.scene.nodes()):
        if node.kind in ('domain','interior','boundary'):w.delete_block(node)
    geometry=w.add_node('Jorvik geometry')
    initial=w.jorvik_document(geometry)
    from jorvik.project import create_empty_project
    external=Path(directory)/'existing.gsproj';create_empty_project(external)
    original=external.read_bytes()
    with patch('margam.ui.jorvik_geometry.QFileDialog.getOpenFileName',return_value=('','')):
        assert not w.browse_jorvik(geometry,'document')
    assert w.jorvik_document(geometry)==initial
    assert w.browse_jorvik(geometry,'document',external)
    assert external.read_bytes()==original and w.jorvik_document(geometry)!=external
    w.undo_edit();assert w.jorvik_document(geometry)==initial
    w.redo_edit()
    project=w.jorvik_document(geometry)
    assert project.suffix=='.jorvik' and project.exists()
    fixture(project.with_suffix('.nc'))
    external_nc=Path(directory)/'different-name.nc';shutil.copyfile(project.with_suffix('.nc'),external_nc)
    assert w.browse_jorvik(geometry,'export',external_nc)
    chosen=w.jorvik_export(geometry);assert chosen!=external_nc and chosen.read_bytes()==external_nc.read_bytes()
    assert w.use_adjacent_jorvik_export(geometry) and w.jorvik_export(geometry)==project.with_suffix('.nc')
    w.undo_edit();assert w.jorvik_export(geometry)==chosen
    before_bad=capture_model(w);broken=Path(directory)/'bad.jorvik';broken.write_bytes(b'bad')
    with patch.object(QMessageBox,'warning'):
        assert not w.browse_jorvik(geometry,'document',broken)
    assert capture_model(w)==before_bad
    assert w.update_jorvik(geometry,SELECTION,confirm=False)
    interior=next(n for n in w.scene.nodes() if n.kind=='interior_data')
    boundary=next(n for n in w.scene.nodes() if n.kind=='boundary_data')
    w.undo_edit();assert not geometry.settings.get('jorvik') and interior.scene() is None
    w.redo_edit();assert interior.scene() is w.scene
    pde=next(n for n in w.scene.nodes() if n.kind=='pde')
    bc=next(n for n in w.scene.nodes() if n.kind=='bc')
    w.scene.connect_nodes(interior,pde);w.scene.connect_nodes(boundary,bc)
    from margam.ui.jorvik_geometry import DatasetDialog
    dialog=DatasetDialog(w.jorvik_export(geometry),{},w)
    assert dialog.selection()==SELECTION
    assert dialog.table.rowCount()==3
    assert [alias.isEnabled() for label,choice,alias in dialog.rows]==[False,True,True]
    dialog.rows[1][2].setText('renamed');assert dialog.selection()['base']==('boundary','renamed')
    dialog.close()
    from PySide6.QtCore import QProcess
    with patch.object(QProcess,'start'):
        assert w.edit_jorvik(geometry)
        process=w.jorvik_process
        assert process.program()==__import__('sys').executable
        assert process.arguments()==['-m','jorvik',str(w.jorvik_document(geometry))]
        assert w.scene.locked and w.project_workspace.busy and not w.confirm_discard()
        process.finished.emit(0,QProcess.ExitStatus.NormalExit)
        assert not w.scene.locked and not w.project_workspace.busy
    assert w.validate_model()
    bc.settings['parameters']['bc_values']['base']=5.
    w.result={'sentinel':True}
    renamed=dict(SELECTION,base=('boundary','floor'))
    assert w.update_jorvik(geometry,renamed,confirm=False)
    assert 'base' not in bc.settings['parameters']['bc_values'] and w.result is None
    w.undo_edit();assert bc.settings['parameters']['bc_values']['base']==5.
    assert geometry.settings['jorvik']['datasets']['base-stable']['name']=='base'
    config=w.scene.compile_config();assert config.shape=='Prepared2D'
    assert len(read_training_sets(config)[0][0])==3
    doc=capture_model(w);decode_document(doc)
    from copy import deepcopy
    invalid=deepcopy(doc)
    record=next(n for n in invalid['nodes'] if n['kind']=='jorvik_geometry')
    record['settings']['jorvik_files']['document']='margam-owned:../outside.jorvik'
    try:decode_document(invalid)
    except ValueError:pass
    else:raise AssertionError('Escaping project file reference accepted')
    assert 'margam-owned:' in str(doc) and str(project.parent) not in str(doc)
    original=Path(directory)/'first.margam.json'
    assert w.save_model(original) and not project.exists()
    assert w.save_model()
    assert str(project.parent) not in str(capture_model(w))
    assert 'margam-owned:' in original.read_text()
    assert w.validate_model()
    # Changing the editable export cannot mutate the previously applied snapshot.
    export=w.jorvik_export(geometry);export.write_bytes(b'incomplete')
    before=capture_model(w)
    with patch.object(QMessageBox,'warning'):
        assert not w.update_jorvik(geometry,SELECTION,confirm=False)
    assert capture_model(w)==before and w.validate_model()
    copy=Path(directory)/'copy.margam.json';assert w.save_model(copy)
    relocated=Path(directory)/'relocated';relocated.mkdir()
    shutil.copy2(copy,relocated/copy.name)
    shutil.copytree(Path(directory)/'copy_files',relocated/'copy_files')
    assert w.load_model(relocated/copy.name) and w.validate_model()
    geometry=next(n for n in w.scene.nodes() if n.kind=='jorvik_geometry')
    assert w.jorvik_document(geometry).is_file()
    source=w.scene.compile_config().interior_data['path']
    Path(source).write_bytes(b'changed snapshot')
    assert not w.validate_model()
    w.project_workspace.busy=True
    assert not w.confirm_discard()
    with patch.object(QMessageBox,'warning'):assert not w.save_model()
    w.project_workspace.busy=False
    w.undo_stack.setClean();w.close()
app.clipboard().clear();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],
                              env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=40)
        self.assertEqual(result.returncode,0,result.stderr)
