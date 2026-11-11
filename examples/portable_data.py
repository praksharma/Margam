"""Exercise copy/link dialogs, portable model files, recovery and run archives."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import time


def run(directory,device='cpu',capture=False):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    import numpy as np
    import torch
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication,QMessageBox,QLabel,QLineEdit,QPlainTextEdit
    from margam.ui.app import Window
    from margam.ui.model_document import capture_model,write_model
    from margam.ui.project_workspace import ProjectWorkspace
    from margam.ui.evaluation_mapping import EvaluationMappingDialog
    from margam.components.evaluation_data import read_point_data
    from margam.templates.poisson import build_pinn
    from unittest.mock import patch
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    external=directory/'external';shutil.copytree('assets/training/triangle',external)
    rows=np.genfromtxt(external/'interior.csv',delimiter=',',names=True)
    np.savez(external/'interior.npz',points=np.column_stack((rows['x'],rows['y'])))
    with (external/'evaluation.csv').open() as stream:evaluation=list(csv.DictReader(stream))
    with (external/'reference.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['id','x','y','u'])
        for row in reversed(evaluation):writer.writerow([row['point_id'],row['x'],row['y'],float(row['x'])**2+float(row['y'])**2])
    app=QApplication.instance() or QApplication([]);window=Window(state_directory=False)
    captures={}
    def node(kind):return next(n for n in window.scene.nodes() if n.kind==kind)
    def finish():
        deadline=time.monotonic()+40
        while window.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
        assert window.worker is None and window.result is not None,window.log.toPlainText()
    def fail(parent,title,message,*args):raise AssertionError(f'{title}: {message}')
    def import_data(action,mapping,*,new=False):
        errors=[]
        def fill():
            try:
                dialog=app.activeModalWidget();assert dialog is not None
                if new:assert dialog.storage.choice.currentData()=='copy'
                for key,value in mapping.items():dialog.fields[key].setText(value)
                dialog.storage.choice.setCurrentIndex(0)
                assert dialog.check(),dialog.preview.toPlainText()
                dialog.accept_checked()
            except Exception as error:
                errors.append(error);app.activeModalWidget().reject()
        QTimer.singleShot(0,fill);action()
        if errors:raise errors[0]
    def save_capture(widget,name):
        from examples.documentation import validate_public_text
        for _ in range(5):app.processEvents()
        for cls in (QLabel,QLineEdit,QPlainTextEdit):
            for child in widget.findChildren(cls):
                if child.isVisible():validate_public_text(child.toPlainText() if isinstance(child,QPlainTextEdit) else child.text(),name)
        target=Path('assets/screenshots')/('manual-data-'+name+'.png');assert widget.grab().save(str(target));captures[name]=target
    try:
        with patch.object(QMessageBox,'warning',side_effect=fail):
            assert window.load_training_data_example();assert window.set_application_device(device)
            import_data(window.edit_point_data,dict(path=str(external/'interior.npz'),coordinates='points',normals='',values=''))
            for side,mapping in list(window.config.boundary_data.items()):
                source=dict(mapping,path=str(external/Path(mapping['path']).name))
                import_data(lambda side=side:window.edit_point_data(side),source)
            points=node('evaluation_points')
            import_data(lambda:window.edit_evaluation_data(points),dict(path=str(external/'evaluation.csv'),coordinates='x,y',ids='point_id',values=''))
            window.delete_block(node('reference'))
            reference=window.add_node('Reference data');reference.setPos(730,530)
            window.scene.connect_nodes(reference,node('comparison'),'reference')
            import_data(lambda:window.edit_evaluation_data(reference),dict(path=str(external/'reference.csv'),coordinates='x,y',ids='id',values='u'),new=True)
            expected=read_point_data(window.project_workspace.resolve_source(points.settings['source_data']),window.config)
            if capture:
                dialog=EvaluationMappingDialog(window.config,points.settings['source_data'],window)
                assert dialog.check();dialog.show();save_capture(dialog,'copy');dialog.close()
                linked=EvaluationMappingDialog(window.config,dict(path='assets/training/triangle/evaluation.csv',coordinates='x,y',values='',ids='point_id'),window)
                assert linked.check();linked.show();save_capture(linked,'link');linked.close()
            document=capture_model(window)
            assert str(external) not in json.dumps(document)
            assert len(list(window.project_workspace.directory.glob('datasets/*/*')))==6
            # All training/evaluation/reference rows must now be independent of originals.
            shutil.rmtree(external);assert window.validate_model(),window.log.toPlainText()
            window.change('iterations',2);window.change('layers',(4,4))
            recovery=directory/'recovery.margam.json'
            window.project_workspace.snapshot_to(recovery,lambda path:write_model(path,capture_model(window)))
            other=Window(state_directory=False)
            try:assert other.load_model(recovery) and other.validate_model()
            finally:other.undo_stack.setClean();other.close();app.processEvents()
            assert window.start_training();finish()
            weights={name:value.detach().cpu().clone() for name,value in window.last_run.model.network.state_dict().items()}
            for name in ('first','second'):
                assert window.save_model(directory/(name+'.margam.json'))
                assert window.evaluate_trained_outputs(),window.log.toPlainText()
                finish();assert window.result['steps']==2
                for key,value in weights.items():torch.testing.assert_close(value,window.last_run.model.network.state_dict()[key].cpu(),rtol=0,atol=0)
            archive=directory/'portable.margam-run.pt';assert window.save_current_run(archive)
            moved=directory/'relocated';moved.mkdir()
            shutil.copy2(directory/'second.margam.json',moved/'second.margam.json')
            shutil.copytree(directory/'second_files',moved/'second_files')
            for name in ('first','second','recovery'):shutil.rmtree(directory/(name+'_files'))
            window.undo_stack.setClean();window.close();app.processEvents();window=Window(state_directory=False)
            assert window.set_application_device(device)
            assert window.load_model(moved/'second.margam.json') and window.validate_model()
            assert window.start_training();finish();assert window.result['steps']==2
            # An archive must also be independent of the relocated model's files.
            shutil.rmtree(moved/'second_files')
            assert window.open_saved_run(archive) and window.validate_model()
            assert window.resume_training(2);finish();assert window.result['steps']==4
            continuous=build_pinn(window.config);continuous.train(4,0)
            assert continuous.loss_history==window.last_run.model.loss_history
            for key,value in continuous.network.state_dict().items():torch.testing.assert_close(value.cpu(),window.last_run.model.network.state_dict()[key],rtol=0,atol=0)
            result=window.result['outputs'][node('comparison').identifier]
            assert result['points']==expected.points and result['point_ids']==expected.point_ids
            np.testing.assert_allclose(result['reference'],[x*x+y*y for x,y in expected.points],rtol=0,atol=1e-12)
            assert window.export_block_data(node('export'),directory/'prediction.csv')
            assert window.export_block_data(node('paraview'),directory/'prediction.vtu')
            if capture:
                window.resize(1550,950);window.show();window.show_block_properties(node('interior_data'));window.fit_model();save_capture(window,'project')
            assert window.save_model(directory/'portable.margam.json')
            report=dict(case='Prepared triangle / copied training, evaluation and reference data',device=device,owned_sources=6,
                        csv_and_npz=True,originals_removed=True,recovery_validated=True,save_as_preserved_weights=True,
                        relocated_model_trained=True,archive_resumed_without_model_files=True,exact_same_device_resume=True,
                        evaluation_order_and_ids_preserved=True,reference_alignment_verified=True,updates=4,
                        exports=['CSV','VTU'],scope='Portability and lifecycle correctness; not solution accuracy.')
            (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            if captures:
                import hashlib
                Path('assets/screenshots/manual-data-captures.json').write_text(json.dumps(dict(generator='examples.portable_data --capture',files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in captures.values()}),indent=2)+'\n')
            return report
    finally:
        window.undo_stack.setClean();window.close();app.clipboard().clear();app.processEvents()


def main():
    from tempfile import mkdtemp
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('build/portable-data'))
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu');parser.add_argument('--capture',action='store_true')
    args=parser.parse_args();args.directory.mkdir(parents=True,exist_ok=True)
    directory=Path(mkdtemp(prefix='run-',dir=args.directory))
    print(json.dumps(run(directory,args.device,args.capture),indent=2));print('Artifacts:',directory)


if __name__=='__main__':main()
