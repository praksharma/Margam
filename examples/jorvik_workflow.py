"""Reproduce the desktop Jorvik → checkpoint → supplied-point VTU workflow."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET


def run(directory, device='cpu', screenshot=None):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    import numpy as np
    import torch
    from PySide6.QtWidgets import QApplication, QMessageBox
    from margam.ui.app import Window
    from margam.components.training_data import read_training_sets
    from examples.jorvik_exchange import sample_annulus
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([])
    window=Window(state_directory=False)
    def fail(parent,title,message,*args):
        raise AssertionError(f'{title}: {message}')
    from unittest.mock import patch
    def finish():
        deadline=time.monotonic()+60
        while window.worker is not None and time.monotonic()<deadline:
            app.processEvents();time.sleep(.005)
        assert window.worker is None and window.result is not None,window.log.toPlainText()
    def node(kind):return next(n for n in window.scene.nodes() if n.kind==kind)
    try:
        with patch.object(QMessageBox,'warning',side_effect=fail):
            assert window.set_application_device(device)
            export,selections=sample_annulus(directory/'source')
            for item in list(window.scene.nodes()):
                if item.kind in ('domain','interior','boundary'):window.delete_block(item)
            geometry=window.add_node('Jorvik geometry')
            assert window.browse_jorvik(geometry,'document',export.with_suffix('.jorvik'))
            assert window.update_jorvik(geometry,selections,confirm=False)
            window.scene.connect_nodes(node('interior_data'),node('pde'))
            window.scene.connect_nodes(node('boundary_data'),node('bc'))
            window.scene.connect_nodes(node('boundary_data'),node('pde'),'boundary_points')
            window.change('source',0.);window.change('layers',(8,8));window.change('iterations',100000)
            # u=log(2r): inner Dirichlet 0; outer outward derivative 1.
            window.change_bc('inner','type','Dirichlet');window.change_bc('inner','value',0.)
            window.change_bc('outer','type','Neumann');window.change_bc('outer','value',1.)
            window.change('boundary_weight',3.)
            points=window.add_node('Evaluation points')
            supplied=directory/'evaluation.csv'
            supplied.write_text('id,x,y\nprobe-C,0.75,0\nprobe-A,0,0.6\nprobe-B,-0.8,0\n')
            window.change_output(points,'source_data',dict(path=str(supplied),coordinates='x,y',ids='id',values=''))
            output=window.add_node('ParaView Export')
            window.scene.connect_nodes(node('method'),output)
            window.scene.connect_nodes(points,output,'points')
            assert window.validate_model(),window.validation.text()
            assert window.scene.compile_config().pde_points=='both'
            definition=directory/'annulus.margam.json'
            assert window.save_model(definition)
            # New window: genuine close/reopen rather than relying on inspector state.
            window.undo_stack.setClean();window.close();app.processEvents()
            window=Window(state_directory=False);assert window.set_application_device(device)
            assert window.load_model(definition) and window.validate_model()
            config=window.scene.compile_config()
            assert config.bc_types['outer']=='Neumann' and config.bc_values['outer']==1.
            interior,boundaries=read_training_sets(config)
            assert [len(interior[0]),len(boundaries['inner'][0]),len(boundaries['outer'][0])]==[97,17,29]
            assert window.start_training()
            window.worker.progress.connect(lambda *args:window.cancel_training())
            finish()
            stopped=window.result['steps']
            assert window.result['cancelled'] and 0<stopped<100000
            model=window.last_run.model
            assert str(model.device)==device
            # Independently check the imported physical operators against u=log(2r).
            from margam.templates.poisson import build_pinn
            operator_model=build_pinn(config)
            for condition,coordinates,target in operator_model._constraints:
                x=coordinates.detach().clone().requires_grad_(True)
                u=.5*torch.log(x.square().sum(1,keepdim=True))+np.log(2.)
                actual=u if condition.operator is None else condition.operator(x,u)
                torch.testing.assert_close(actual,target.expand_as(actual),atol=2e-6,rtol=2e-6)
            del operator_model
            assert len(window.last_run.model.collocation_point_sample)==143
            archive=directory/'annulus.margam-run.pt'
            assert window.save_current_run(archive)
            # Hide the original companion folder: continuation must use archive contents.
            original_files=directory/'annulus_files';hidden=directory/'original-files-hidden'
            original_files.rename(hidden)
            assert window.open_saved_run(archive)
            assert window.validate_model(),window.log.toPlainText()
            assert window.evaluate_trained_outputs();finish()
            hidden.rename(original_files)
            assert window.resume_training(2),window.log.toPlainText();finish()
            assert window.result['steps']==stopped+2
            from margam.templates.poisson import build_pinn
            continuous=build_pinn(window.config);continuous.train(stopped+2,0)
            assert continuous.loss_history==window.last_run.model.loss_history
            for key,value in continuous.network.state_dict().items():
                torch.testing.assert_close(value.cpu(),window.last_run.model.network.state_dict()[key],rtol=0,atol=0)
            before={key:value.detach().cpu().clone() for key,value in window.last_run.model.network.state_dict().items()}
            assert window.evaluate_trained_outputs();finish()
            for key,value in before.items():torch.testing.assert_close(value,window.last_run.model.network.state_dict()[key].cpu(),rtol=0,atol=0)
            data=window.result['outputs'][node('paraview').identifier]
            assert data['point_ids']==['probe-C','probe-A','probe-B']
            assert data['points']==[[.75,0.],[0.,.6],[-.8,0.]]
            assert not window.output_views.entries
            destination=directory/'prediction.vtu';assert not destination.exists()
            assert window.export_block_data(node('paraview'),destination)
            mesh=ET.parse(destination).find('UnstructuredGrid');piece=mesh.find('Piece')
            assert piece.get('NumberOfPoints')=='3' and piece.get('NumberOfCells')=='3'
            coordinates=np.fromstring(piece.find('Points/DataArray').text,sep=' ').reshape(-1,3)
            np.testing.assert_array_equal(coordinates[:,:2],data['points'])
            values=np.fromstring(piece.find("PointData/DataArray[@Name='u']").text,sep=' ')
            np.testing.assert_array_equal(values,data['values'])
            encoded=mesh.find("FieldData/DataArray[@Name='point_ids_json_utf8']")
            assert json.loads(bytes(int(v) for v in encoded.text.split()).decode())==data['point_ids']
            # Verify actual VTK reader interoperability when the optional reader exists.
            reader_checked=False
            try:from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
            except ImportError:pass
            else:
                reader=vtkXMLUnstructuredGridReader();reader.SetFileName(str(destination));reader.Update()
                grid=reader.GetOutput();assert grid.GetNumberOfPoints()==3 and grid.GetPointData().GetArray('u') is not None
                reader_checked=True
            if screenshot is not None:
                from PySide6.QtWidgets import QLabel,QLineEdit,QPlainTextEdit
                from examples.documentation import validate_public_text
                positions={'jorvik_geometry':(0,170),'interior_data':(250,40),'boundary_data':(250,280),'pde':(500,40),'bc':(500,280),'network':(500,450),'method':(750,170),'evaluation_points':(750,450),'paraview':(1000,170)}
                for item in window.scene.nodes():item.setPos(*positions[item.kind])
                window.resize(1550,950);window.show_block_properties(node('bc'));window.show();window.fit_model()
                for _ in range(5):app.processEvents()
                for cls in (QLabel,QLineEdit,QPlainTextEdit):
                    for child in window.findChildren(cls):
                        if child.isVisible():validate_public_text(child.toPlainText() if isinstance(child,QPlainTextEdit) else child.text(),'Jorvik workflow capture')
                screenshot=Path(screenshot);screenshot.parent.mkdir(parents=True,exist_ok=True)
                assert window.grab().save(str(screenshot))
            assert window.save_model(directory/'resumed.margam.json')
            moved=directory/'relocated';moved.mkdir(exist_ok=True)
            shutil.copy2(definition,moved/definition.name)
            shutil.copytree(directory/'annulus_files',moved/'annulus_files',dirs_exist_ok=True)
            assert window.load_model(moved/definition.name) and window.validate_model()
            assert window.jorvik_document(node('jorvik_geometry')).is_relative_to(moved)
            report=dict(case='Jorvik annulus desktop workflow',device=device,counts=dict(interior=97,inner=17,outer=29,pde=143),
                        boundary_types=dict(inner='Dirichlet',outer='Neumann'),boundary_weight=3.,updates=stopped+2,stopped_after_updates=stopped,stop_verified=True,mixed_operators_verified=True,
                        project_reopened=True,owned_geometry_relocated=True,archive_resumed=True,exact_same_device_resume=True,
                        inference_preserved_weights=True,export_points=3,export_ids_preserved=True,
                        vtk_reader_checked=reader_checked,
                        scope='Correctness check; no accuracy claim. Evaluation CSV remains an external input.')
            (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            return report
    finally:
        window.undo_stack.setClean();window.close();app.clipboard().clear();app.processEvents()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('build/jorvik-workflow'))
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--screenshot',type=Path)
    args=parser.parse_args()
    from tempfile import mkdtemp
    args.directory.mkdir(parents=True,exist_ok=True)
    destination=Path(mkdtemp(prefix='run-',dir=args.directory))
    print(json.dumps(run(destination,args.device,args.screenshot),indent=2))
    print('Artifacts:',destination)


if __name__=='__main__':main()
