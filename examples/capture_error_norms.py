"""Capture actual Comparison norms after a short isolated Couette lifecycle run."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication,QLabel
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    path=Path('assets/screenshots/manual-error-norms.png');manifest=path.with_suffix('.json')
    if not args.overwrite and (path.exists() or manifest.exists()):parser.error('Capture exists; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False)
    try:
        w.set_application_device('cpu');w.load_flow_template();w.change('iterations',2);w.change('layers',(4,))
        assert w.start_training()
        while w.worker is not None:app.processEvents()
        assert w.result is not None;result=w.result
        node=next(n for n in w.scene.nodes() if n.kind=='comparison');dialog=w.show_error_norms(node)
        dialog.fields.item(1).setCheckState(Qt.CheckState.Checked);dialog.resize(940,610)
        for _ in range(5):app.processEvents()
        for child in dialog.findChildren(QLabel):
            if child.isVisible():validate_public_text(child.text(),'error norm capture')
        assert w.result is result and len(dialog.reports)==3
        assert dialog.reports[0]['fields']==['u','v'] and dialog.reports[-1]['status']=='undefined_zero_reference'
        assert dialog.grab().save(str(path))
        manifest.write_text(json.dumps(dict(generator='examples.capture_error_norms',scope='Actual stored Couette comparison after two CPU updates; vector selection and a zero-reference component. Lifecycle fixture, not an accuracy benchmark.',
            reports=dialog.reports,file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2,allow_nan=False)+'\n')
        print('Captured selected scalar/vector error norms.')
    finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
