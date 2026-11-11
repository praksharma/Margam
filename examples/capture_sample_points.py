"""Capture retained adaptive coordinates, including fixed PDE boundary anchors."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def prepare(window,kind='rar'):
    from margam.components.adaptive_sampling import sampling_defaults
    window.load_example('interval_poisson')
    window.delete_block(next(n for n in window.scene.nodes() if n.kind=='interior'))
    node=window.add_node('Residual refinement (RAR)' if kind=='rar' else 'Residual importance sampling')
    domain=next(n for n in window.scene.nodes() if n.kind=='domain');pde=next(n for n in window.scene.nodes() if n.kind=='predefined_pde')
    window.scene.connect_nodes(domain,node);window.scene.connect_nodes(node,pde)
    window.change('points',206);window.change('iterations',5);window.change('layers',(4,))
    settings=sampling_defaults(kind);settings.update(interval=2,candidates=19,chunk=8)
    if kind=='rar':settings.update(batch=3,maximum=218)
    window.change('adaptive_sampling',settings)
    return node


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit
    from margam.ui.app import Window
    from margam.ui.model_document import capture_model
    from margam.ui.theme import configure_theme
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    path=Path('assets/screenshots/manual-retained-points.png');manifest=path.with_suffix('.json')
    if not args.overwrite and any(p.exists() for p in (path,manifest)):parser.error('Capture exists; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False)
    try:
        w.set_application_device('cpu');node=prepare(w)
        assert w.start_training()
        while w.worker is not None:app.processEvents()
        before=capture_model(w);index=w.undo_stack.index();result=w.result
        dialog=w.show_sample_points(node);dialog.page.setValue(2);dialog.resize(1000,740)
        for _ in range(5):app.processEvents()
        assert dialog.snapshot and dialog.snapshot['count']==214 and dialog.snapshot['boundary_count']==2
        assert capture_model(w)==before and w.undo_stack.index()==index and w.result is result
        for cls in (QLabel,QLineEdit,QPlainTextEdit):
            for child in dialog.findChildren(cls):
                if child.isVisible():validate_public_text(child.toPlainText() if hasattr(child,'toPlainText') else child.text(),'retained point capture')
        assert dialog.grab().save(str(path))
        manifest.write_text(json.dumps(dict(generator='examples.capture_sample_points',scope='Actual RAR points after five CPU updates; page two includes original rows, fixed PDE anchors and additions. Lifecycle evidence, not accuracy.',
            snapshot=dialog.snapshot,file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2,allow_nan=False)+'\n')
        print('Captured retained RAR point page without changing the run.')
    finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
