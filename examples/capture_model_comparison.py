"""Capture actual model comparison with isolated, repository-relative fixtures."""
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
    from margam.ui.model_document import capture_model,write_model
    from margam.ui.theme import configure_theme
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    path=Path('assets/screenshots/manual-model-comparison.png');manifest=path.with_suffix('.json')
    if not args.overwrite and (path.exists() or manifest.exists()):parser.error('Capture exists; use --overwrite.')
    fixture=Path('build/comparison-capture/baseline.margam.json');fixture.parent.mkdir(parents=True,exist_ok=True)
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False)
    try:
        write_model(fixture,capture_model(w))
        w.change('iterations',2000);w.change('lr',.0005);w.change('points',256);w.change('layers',(32,32))
        wire=next(wire for wire in w.scene.wires if wire.target.name=='boundary_points')
        w.scene.clearSelection();wire.setSelected(True);w.delete_selected()
        dialog=w.show_model_comparison();assert dialog.set_file(fixture)
        dialog.resize(1200,740)
        for _ in range(5):app.processEvents()
        row=next(i for i,x in enumerate(dialog.visible_changes) if x.field.endswith('/iterations'))
        dialog.table.setCurrentCell(row,0)
        for cls in (QLabel,QLineEdit,QPlainTextEdit):
            for child in dialog.findChildren(cls):
                if child.isVisible():validate_public_text(child.toPlainText() if hasattr(child,'toPlainText') else child.text(),'model comparison capture')
        assert dialog.grab().save(str(path))
        manifest.write_text(json.dumps(dict(generator='examples.capture_model_comparison',scope='Actual read-only comparison of settings and an explicit boundary-PDE wire removal. No training or accuracy claims.',
            differences=len(dialog.changes),file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+'\n')
        print('Captured model comparison.')
    finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
