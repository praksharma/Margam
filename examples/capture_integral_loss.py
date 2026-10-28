"""Capture the actual integral-loss control without personal desktop state."""
import argparse
import hashlib,json,os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QComboBox,QLabel,QLineEdit,QPlainTextEdit
    from PySide6.QtTest import QTest
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    path=root/'assets/screenshots/manual-loss-integral.png';manifest=path.with_suffix('.json')
    if not args.overwrite and (path.exists() or manifest.exists()):parser.error('Capture exists; use --overwrite to replace it.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False);w.resize(1560,1050);w.show()
    try:
        w.load_heat_template();w.change('x_max',3.);w.change('y_max',2.)
        trainer=next(n for n in w.scene.nodes() if n.kind=='method')
        w.scene.clearSelection();trainer.setSelected(True);app.processEvents()
        choice=w.property_scroll.findChild(QComboBox,'loss_formulation');choice.setCurrentIndex(choice.findData('integral'))
        assert w.validate_model();assert w.scene.compile_config().loss_formulation=='integral'
        w.fit_model();w.property_scroll.widget().repaint();w.repaint();QTest.qWait(100)
        for cls in (QLabel,QLineEdit,QPlainTextEdit):
            for child in w.findChildren(cls):validate_public_text(child.toPlainText() if isinstance(child,QPlainTextEdit) else child.text(),'integral loss screenshot')
        if not w.grab().save(str(path)):raise RuntimeError('Screenshot failed.')
        manifest.write_text(json.dumps(dict(scope='Actual Qt integral-loss inspector on a 3 by 2 rectangle; no training run.',
            file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+'\n')
        print('Captured the integral loss inspector.')
    finally:
        app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()


if __name__=='__main__':main()
