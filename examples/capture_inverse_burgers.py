"""Capture the inverse graph and requested diagnostics using isolated CPU state."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit,QTextBrowser
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    paths={key:Path('assets/screenshots/manual-'+key+'.png') for key in ('inverse-burgers','inverse-coefficients','inverse-monitor')}
    manifest=Path('assets/screenshots/manual-inverse-burgers-capture.json')
    if not args.overwrite and any(p.exists() for p in [*paths.values(),manifest]):parser.error('Capture exists; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False)
    def settle():
        for _ in range(5):app.processEvents()
    def capture(widget,key):
        for cls in (QLabel,QLineEdit,QPlainTextEdit,QTextBrowser):
            for child in widget.findChildren(cls):
                if child.isVisible():validate_public_text(child.toPlainText() if hasattr(child,'toPlainText') else child.text(),'inverse capture')
        assert widget.grab().save(str(paths[key]))
    try:
        w.resize(1400,900);w.set_application_device('cpu');w.show();assert w.load_inverse_example();settle();capture(w,'inverse-burgers')
        w.change('iterations',4);assert w.start_training()
        while w.worker is not None:app.processEvents()
        assert w.result is not None
        dialog=w.show_inverse_parameters();settle();capture(dialog,'inverse-coefficients');dialog.close()
        monitor=next(n for n in w.scene.nodes() if n.kind=='monitor');w.output_views.open(monitor)
        trainer=next(n for n in w.scene.nodes() if n.kind=='inverse_method');w.show_block_properties(trainer);w.fit_model();settle();capture(w,'inverse-monitor')
        values,history=w.inverse_parameters()
        data=dict(generator='examples.capture_inverse_burgers',scope='Actual four-update CPU lifecycle fixture, not a coefficient accuracy benchmark.',observations=2000,estimates=values,history_updates=len(history),files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths.values()})
        manifest.write_text(json.dumps(data,indent=2)+'\n');print('Captured inverse graph, learned coefficients and observation-loss monitor.')
    finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()

if __name__=='__main__':main()
