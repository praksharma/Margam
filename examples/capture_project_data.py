"""Capture the actual read-only data overview with copied, changed and missing inputs."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit,QTextBrowser
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from margam.ui.model_document import capture_model
    from margam.components.prepared_data import read_points
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    image=Path('assets/screenshots/manual-project-data.png');manifest=image.with_name('manual-project-data-capture.json')
    if not args.overwrite and (image.exists() or manifest.exists()):parser.error('Capture exists; use --overwrite.')
    work=Path('build/project-data-overview');work.mkdir(parents=True,exist_ok=True)
    app=QApplication([]);configure_theme(app,'light');window=Window(state_directory=False)
    try:
        with TemporaryDirectory(dir=work) as folder:
            directory=Path(folder).relative_to(root)
            for name in ('base.csv','upright.csv','interior.csv'):shutil.copyfile(Path('assets/training/triangle')/name,directory/name)
            assert window.load_training_data_example()
            copied=window.project_workspace.store_dataset(dict(window.config.interior_data,path=str(directory/'interior.csv')),copy=True,validator=lambda source:read_points(source,2))
            window.change('interior_data',copied)
            mappings=deepcopy(window.config.boundary_data)
            for name in ('base','upright'):mappings[name]['path']=str(directory/(name+'.csv'))
            window.change('boundary_data',mappings)
            window.record_dataset_import(window.parameter_owner('interior_data'),'interior_data',copied,lambda:None)
            for name in ('base','upright'):
                window.record_dataset_import(window.parameter_owner('boundary_data'),'boundary:'+name,mappings[name],lambda:None)
            model=directory/'fingerprinted.margam.json'
            assert window.save_model(model) and window.load_model(model)
            (directory/'base.csv').unlink()
            path=directory/'upright.csv';path.write_bytes(path.read_bytes()+b'\n')
            before=capture_model(window);index=window.undo_stack.index()
            dialog=window.show_project_data()
            assert len(dialog.records)==5 and sum(result['changed'] for result in dialog.results)==1
            assert sum(result['status']=='Missing' for result in dialog.results)==1
            assert any(result['status']=='Changed since import' for result in dialog.results)
            selected=next(i for i,record in enumerate(dialog.records) if record['boundary']=='upright');dialog.table.selectRow(selected)
            assert capture_model(window)==before and window.undo_stack.index()==index
            for _ in range(5):app.processEvents()
            for cls in (QLabel,QLineEdit,QPlainTextEdit,QTextBrowser):
                for child in dialog.findChildren(cls):
                    if child.isVisible():validate_public_text(child.toPlainText() if hasattr(child,'toPlainText') else child.text(),'project data capture')
            assert dialog.grab().save(str(image))
            data=dict(generator='examples.capture_project_data',scope='Actual read-only inventory; file checks are separate from numerical validation.',
                      rows=[dict(block=record['block'],role=record['role'],storage=result['storage'],rows=result['rows'],status=result['status'],baseline=record['baseline_kind']) for record,result in zip(dialog.records,dialog.results)],
                      read_only_verified=True,files={image.name:dict(bytes=image.stat().st_size,sha256=hashlib.sha256(image.read_bytes()).hexdigest())})
            manifest.write_text(json.dumps(data,indent=2)+'\n');print('Captured project data overview and verified read-only behaviour.')
    finally:
        app.clipboard().clear();window.undo_stack.setClean();window.close();app.processEvents()


if __name__=='__main__':main()
