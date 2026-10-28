"""Real Expert recipe/trainer/network inspector captures, with isolated UI state."""
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
    paths={name:out/('manual-expert-'+name+'.png') for name in ('trainer','network','compact')};manifest=out/'manual-expert-captures.json'
    if not args.overwrite and any(p.exists() for p in (*paths.values(),manifest)):parser.error('Captures exist; use --overwrite to replace them.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False);w.resize(1560,1250);w.show()
    def capture(name,node):
        w.scene.clearSelection();node.setSelected(True)
        for _ in range(4):app.processEvents()
        w.fit_model();w.property_scroll.widget().repaint();w.repaint();QTest.qWait(100)
        for cls in (QLabel,QLineEdit,QPlainTextEdit):
            for child in w.findChildren(cls):validate_public_text(child.toPlainText() if isinstance(child,QPlainTextEdit) else child.text(),name)
        if not w.grab().save(str(paths[name])):raise RuntimeError('Capture failed.')
    try:
        w.load_example('expert_burgers');assert w.validate_model(),w.validation.text()
        trainer=next(n for n in w.scene.nodes() if n.kind=='expert_method');capture('trainer',trainer)
        capture('network',next(n for n in w.scene.nodes() if n.kind=='network'))
        compact=w.collapse_recipe(trainer);assert compact;capture('compact',compact)
        manifest.write_text(json.dumps(dict(scope='Actual Expert PINN inspectors; no training or accuracy claim.',
            captures={name:dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for name,p in paths.items()}),indent=2)+'\n')
        print('Captured Expert trainer, RWF network and compact recipe.')
    finally:
        app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()


if __name__=='__main__':main()
