"""Capture the real Expert examples submenu and geometry/condition inspectors."""
import argparse
import hashlib,json,os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root);out=root/'assets/screenshots'
    paths={name:out/('manual-expert-suite-'+name+'.png') for name in ('menu','cylinder','torus')};manifest=out/'manual-expert-suite-captures.json'
    if not args.overwrite and any(p.exists() for p in (*paths.values(),manifest)):parser.error('Captures exist; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False);w.resize(1560,1050);w.show()
    def save(name,widget):
        for _ in range(4):app.processEvents()
        QTest.qWait(100)
        for cls in (QLabel,QLineEdit,QPlainTextEdit):
            for child in widget.findChildren(cls):validate_public_text(child.toPlainText() if isinstance(child,QPlainTextEdit) else child.text(),name)
        if not widget.grab().save(str(paths[name])):raise RuntimeError('Capture failed.')
    try:
        w.expert_examples_menu.popup(w.mapToGlobal(QPoint(30,70)));save('menu',w.expert_examples_menu);w.expert_examples_menu.close()
        for key,kind,name in [('expert_cylinder','domain','cylinder'),('expert_torus','bc','torus')]:
            w.load_example(key);assert w.validate_model(),w.validation.text()
            w.scene.clearSelection();next(n for n in w.scene.nodes() if n.kind==kind).setSelected(True)
            app.processEvents();w.fit_model();save(name,w)
        manifest.write_text(json.dumps(dict(scope='Actual Qt submenu and supported case inspectors; no accuracy claim.',captures={name:dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in paths.items()}),indent=2)+'\n')
        print('Captured Expert examples submenu, cylinder geometry and torus conditions.')
    finally:
        app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()


if __name__=='__main__':main()
