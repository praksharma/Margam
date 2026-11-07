"""Capture saved block labels and visible type/identity using isolated CPU state."""
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
    from margam.ui.model_document import capture_model
    from margam.ui.theme import configure_theme
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    path=Path('assets/screenshots/manual-block-labels.png');manifest=path.with_suffix('.json')
    if not args.overwrite and (path.exists() or manifest.exists()):parser.error('Capture exists; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False)
    try:
        w.set_application_device('cpu');w.load_heat_template();w.change('iterations',2);w.change('layers',(4,))
        assert w.start_training()
        while w.worker is not None:app.processEvents()
        assert w.result is not None;result=w.result
        labels=dict(domain='Plate domain',interior='PDE samples',boundary='Wall points',network='Candidate NN',method='Baseline run',monitor='Loss history')
        for node in w.scene.nodes():
            if node.kind in labels:assert w.rename_block(node,labels[node.kind])
        assert w.result is result
        monitor=next(n for n in w.scene.nodes() if n.kind=='monitor');w.output_views.open(monitor)
        network=next(n for n in w.scene.nodes() if n.kind=='network');w.scene.clearSelection();network.setSelected(True)
        w.resize(1600,1000);w.show()
        for _ in range(5):app.processEvents()
        w.fit_model()
        for _ in range(5):app.processEvents()
        for cls in (QLabel,QLineEdit,QPlainTextEdit):
            for child in w.findChildren(cls):
                if child.isVisible():validate_public_text(child.toPlainText() if hasattr(child,'toPlainText') else child.text(),'block labels capture')
        assert w.grab().save(str(path))
        manifest.write_text(json.dumps(dict(generator='examples.capture_block_labels',scope='Actual labels, unchanged types/IDs and retained monitor data after two CPU updates. Lifecycle fixture, not an accuracy benchmark.',
            model_version=capture_model(w)['version'],labels=labels,file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+'\n')
        print('Captured independent block labels and retained results.')
    finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
