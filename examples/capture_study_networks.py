"""Capture real architecture/width preflight and a version-2 frozen-default study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit,QTableWidget
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from margam.ui.model_document import capture_model
    from margam.studies import catalogue
    from margam.study_document import read,trial_model
    from margam.execution.study_runner import source_identity
    from examples.documentation import validate_public_text
    import torch
    root=Path(__file__).resolve().parents[1];image=root/'assets/screenshots/manual-study-networks.png';manifest=image.with_suffix('.json')
    if not args.overwrite and (image.exists() or manifest.exists()):parser.error('Capture exists; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');window=Window(state_directory=False)
    try:
        window.set_application_device('cpu')
        for key,value in [('points',8),('boundary_points',4),('layers',(8,8)),('iterations',2)]:window.change(key,value)
        before=capture_model(window);undo=window.undo_stack.index();rng=torch.get_rng_state().clone();dialog=window.show_design_points()
        for key,text in [('network_type','["fcnn","dgm"]'),('layers','[[8,8],[8,16]]')]:
            dialog.add_axis();row=dialog.axes.rowCount()-1;axis=next(item for item in catalogue(before) if item['parameter']==key)
            combo=dialog.axes.cellWidget(row,0);combo.setCurrentIndex(combo.findData(json.dumps([axis['node_id'],key])))
            mode=dialog.axes.cellWidget(row,1);mode.setCurrentIndex(mode.findData('candidates'));dialog.axes.cellWidget(row,2).setText(text)
        for _ in range(4):app.processEvents()
        assert dialog.refresh();assert (dialog.result['total'],dialog.result['valid'],dialog.result['invalid'])==(4,3,1)
        with TemporaryDirectory(prefix='margam-network-study-capture-') as directory:
            path=Path(directory)/'network-grid.margam-study';assert dialog.save_study(path),dialog.status.text()
            study,_=read(path);assert study['version']==2 and set(study['network_defaults'])=={'fcnn','dgm'}
            for index in range(4):assert trial_model(study,index)[1]['nodes']
            dialog.table.setCurrentCell(3,0)
            for _ in range(4):app.processEvents()
            assert 'equal hidden widths' in dialog.details.toPlainText();assert 'resolved options' in dialog.details.toPlainText()
            assert before==capture_model(window) and undo==window.undo_stack.index() and torch.equal(rng,torch.get_rng_state())
            for cls in (QLabel,QLineEdit,QPlainTextEdit):
                for widget in dialog.findChildren(cls):validate_public_text(widget.toPlainText() if isinstance(widget,QPlainTextEdit) else widget.text(),'Network study capture')
            for table in dialog.findChildren(QTableWidget):
                for r in range(table.rowCount()):
                    for c in range(table.columnCount()):
                        if table.item(r,c):validate_public_text(table.item(r,c).text(),'Network study table')
            assert dialog.grab().save(str(image))
            record=dict(generator='examples.capture_study_networks',scope='Actual four-trial architecture/width preview: three structurally valid and one invalid. Version-2 frozen-default study saved; no training or accuracy claim.',
                source_sha256=source_identity()['sha256'],capture_tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                study_version=2,total=4,valid=3,invalid=1,current_model_preserved=True,parent_rng_preserved=True,
                file=image.name,sha256=hashlib.sha256(image.read_bytes()).hexdigest())
            text=json.dumps(record,indent=2)+'\n';validate_public_text(text,'Network study evidence');manifest.write_text(text)
        print('Captured architecture study preview; frozen defaults and model/RNG preservation verified.')
    finally:window.undo_stack.setClean();window.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
