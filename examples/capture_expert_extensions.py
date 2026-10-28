"""Real Qt NTK and time-window controls, isolated from personal desktop state."""
import argparse
import hashlib,json,os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit
    from PySide6.QtTest import QTest
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root);out=root/'assets/screenshots'
    path=out/'manual-expert-extensions.png';manifest=out/'manual-expert-extensions-captures.json'
    if not args.overwrite and (path.exists() or manifest.exists()):parser.error('Captures exist; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False);w.resize(1560,1500);w.show()
    try:
        w.load_example('expert_burgers');trainer=next(n for n in w.scene.nodes() if n.kind=='expert_method');w.scene.clearSelection();trainer.setSelected(True);app.processEvents()
        w.change_expert_option('balance_mode','ntk');w.change_expert_option('windows',4);w.change_expert_option('window_steps',1000);w.selection_changed();w.fit_model()
        for _ in range(4):app.processEvents()
        w.property_scroll.verticalScrollBar().setValue(w.property_scroll.verticalScrollBar().maximum());QTest.qWait(100)
        for cls in (QLabel,QLineEdit,QPlainTextEdit):
            for child in w.findChildren(cls):validate_public_text(child.toPlainText() if isinstance(child,QPlainTextEdit) else child.text(),'Expert controls')
        if not w.grab().save(str(path)):raise RuntimeError('Capture failed.')
        manifest.write_text(json.dumps(dict(scope='Actual Expert trainer NTK/time-window controls; no training or accuracy claim.',file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+'\n')
        print('Captured NTK and time-window controls.')
    finally:
        app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()


if __name__=='__main__':main()
