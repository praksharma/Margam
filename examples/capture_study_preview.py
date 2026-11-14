"""Capture a state-isolated parameter preview, without training or publishing."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from margam.ui.model_document import capture_model
    from margam.studies import catalogue
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];image=root/'assets/screenshots/manual-study-preview.png';manifest=image.with_suffix('.json')
    if not args.overwrite and (image.exists() or manifest.exists()):parser.error('Capture exists; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');window=Window(state_directory=False)
    try:
        before=capture_model(window);index=window.undo_stack.index();dialog=window.show_design_points()
        for key,mode,text in [('iterations','range','[1000,3000,1000]'),('lr','candidates','[0.001,0.003]'),('layers','candidates','[[16,16],[32,32]]')]:
            dialog.add_axis();row=dialog.axes.rowCount()-1;record=next(item for item in catalogue(before) if item['parameter']==key)
            combo=dialog.axes.cellWidget(row,0);combo.setCurrentIndex(combo.findData(json.dumps([record['node_id'],key])))
            widget=dialog.axes.cellWidget(row,1);widget.setCurrentIndex(widget.findData(mode));dialog.axes.cellWidget(row,2).setText(text)
        for _ in range(4):app.processEvents()
        assert dialog.refresh();assert dialog.result['total']==12 and dialog.result['valid']==12
        assert capture_model(window)==before and index==window.undo_stack.index() and window.worker is None
        for cls in (QLabel,QLineEdit,QPlainTextEdit):
            for child in dialog.findChildren(cls):validate_public_text(child.toPlainText() if isinstance(child,QPlainTextEdit) else child.text(),'Study preview capture')
        assert dialog.grab().save(str(image))
        metadata=dict(generator='examples.capture_study_preview',scope='Actual read-only 12-design-point preview; no training, export or persistent study.',
            design_points=12,total_updates=dialog.result['total_updates'],largest_network_parameters=dialog.result['maximum_parameters'],file=image.name,sha256=hashlib.sha256(image.read_bytes()).hexdigest())
        manifest.write_text(json.dumps(metadata,indent=2)+'\n');print('Captured parameter-study preview.')
    finally:window.undo_stack.setClean();window.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
