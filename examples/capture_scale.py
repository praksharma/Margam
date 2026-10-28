"""Real Scale trainer/recipe captures with isolated desktop state."""
import argparse
import hashlib,json,os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args();os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root);out=root/'assets/screenshots';paths={name:out/('manual-scale-'+name+'.png') for name in ('trainer','compact','examples')};manifest=out/'manual-scale-captures.json'
    if not args.overwrite and any(path.exists() for path in (*paths.values(),manifest)):parser.error('Captures exist; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False);w.resize(1560,1250);w.show()
    def save(name,widget):
        for _ in range(4):app.processEvents()
        QTest.qWait(100)
        for cls in (QLabel,QLineEdit,QPlainTextEdit):
            for child in widget.findChildren(cls):validate_public_text(child.toPlainText() if isinstance(child,QPlainTextEdit) else child.text(),name)
        if not widget.grab().save(str(paths[name])):raise RuntimeError('Capture failed.')
    try:
        w.load_example('scale_cavity_3200');assert w.validate_model(),w.validation.text();trainer=next(n for n in w.scene.nodes() if n.kind=='scale_method');w.scene.clearSelection();trainer.setSelected(True);app.processEvents();w.fit_model();save('trainer',w)
        compact=w.collapse_recipe(trainer);assert compact;w.scene.clearSelection();compact.setSelected(True);app.processEvents();w.fit_model();save('compact',w)
        w.scale_examples_menu.popup(w.mapToGlobal(QPoint(20,50)));save('examples',w.scale_examples_menu);w.scale_examples_menu.close()
        manifest.write_text(json.dumps(dict(scope='Actual Scale controls/compact recipe/submenu; no training or accuracy claim.',captures={name:dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in paths.items()}),indent=2)+'\n')
        print('Captured Scale trainer, compact recipe and examples.')
    finally:
        app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()


if __name__=='__main__':main()
