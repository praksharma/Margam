"""Capture an exact stored-time frame after a short real Burgers run."""
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
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    path=Path('assets/screenshots/manual-result-times.png');manifest=path.with_suffix('.json')
    if not args.overwrite and any(p.exists() for p in (path,manifest)):parser.error('Capture exists; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False)
    try:
        w.set_application_device('cpu');w.load_example('burgers_1d');w.change('iterations',2);w.change('points',16);w.change('layers',(4,))
        for node in w.scene.nodes():
            if node.kind in ('solution','export','paraview'):w.change_output(node,'resolution',9)
        assert w.start_training()
        while w.worker is not None:app.processEvents()
        solution=next(n for n in w.scene.nodes() if n.kind=='solution');w.open_block(solution)
        dock,view=w.output_views.entries[solution.identifier];before=capture_model(w);index=w.undo_stack.index();result=w.result
        view.time.setCurrentIndex(view.time.findData(.75));dock.setFloating(True);dock.resize(1100,620);dock.show()
        for _ in range(5):app.processEvents()
        assert view.table.rowCount()==9 and capture_model(w)==before and w.undo_stack.index()==index and w.result is result
        for cls in (QLabel,QLineEdit,QPlainTextEdit):
            for child in dock.findChildren(cls):
                if child.isVisible():validate_public_text(child.toPlainText() if hasattr(child,'toPlainText') else child.text(),'physical-time capture')
        assert dock.grab().save(str(path))
        manifest.write_text(json.dumps(dict(generator='examples.capture_result_times',scope='Actual stored Burgers predictions after two CPU updates; display filtering only, not an accuracy benchmark.',
            selected_time=view.time.currentData(),stored_frames=view.time.count()-1,displayed_rows=view.table.rowCount(),stored_rows=81,
            file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+'\n')
        print('Captured a stored physical-time frame without changing the model or results.')
    finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
