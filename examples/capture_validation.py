"""Capture actual actionable validation issues without training or auto-repair."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QTextBrowser
    from margam.ui.app import Window
    from margam.ui.model_document import capture_model
    from margam.ui.theme import configure_theme
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    image=Path('assets/screenshots/manual-validation.png');manifest=image.with_name('manual-validation-capture.json')
    if not args.overwrite and (image.exists() or manifest.exists()):parser.error('Capture exists; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');window=Window(state_directory=False)
    try:
        window.change_layers('0, -1');window.change('pde_expression','unsupported(u)')
        window.change_bc('left','type','Custom');window.change_bc_expression('left','bc_operators','bad_operator(u)')
        for wire in list(window.scene.wires):
            if wire.target.parentItem().kind=='method' and wire.target.name=='pde':
                window.scene.wires.remove(wire);window.scene.removeItem(wire)
        before=capture_model(window);index=window.undo_stack.index()
        dialog=window.show_validation_report()
        assert capture_model(window)==before and window.undo_stack.index()==index and window.worker is None
        selected=next(i for i,issue in enumerate(dialog.issues) if issue.boundary=='left');dialog.table.selectRow(selected)
        for _ in range(5):app.processEvents()
        for cls in (QLabel,QLineEdit,QTextBrowser):
            for child in dialog.findChildren(cls):
                if child.isVisible():validate_public_text(child.toPlainText() if isinstance(child,QTextBrowser) else child.text(),'validation capture')
        assert dialog.grab().save(str(image))
        manifest.write_text(json.dumps(dict(generator='examples.capture_validation',scope='Actual validation report; no training or automatic repair.',model_and_history_preserved=True,
            issues=[dict(severity=i.severity,category=i.category,message=i.message,boundary=i.boundary) for i in dialog.issues],
            files={image.name:dict(bytes=image.stat().st_size,sha256=hashlib.sha256(image.read_bytes()).hexdigest())}),indent=2)+'\n')
        print('Captured validation report and verified model/history preservation.')
    finally:
        app.clipboard().clear();window.undo_stack.setClean();window.close();app.processEvents()


if __name__=='__main__':main()
