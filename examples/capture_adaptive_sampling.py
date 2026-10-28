"""Capture the two real adaptive sampler inspectors without personal UI state."""
import argparse
import hashlib,json,os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit,QDoubleSpinBox
    from PySide6.QtTest import QTest
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root);out=root/'assets/screenshots'
    paths={kind:out/('manual-sampling-'+kind+'.png') for kind in ('rar','importance')}
    manifest=out/'manual-adaptive-captures.json'
    if not args.overwrite and any(p.exists() for p in (*paths.values(),manifest)):parser.error('Captures exist; use --overwrite to replace them.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False);w.resize(1560,1050);w.show()
    def events():
        for _ in range(4):app.processEvents()
    try:
        for kind,title in [('rar','Residual refinement (RAR)'),('importance','Residual importance sampling')]:
            w.load_heat_template()
            old=next(n for n in w.scene.nodes() if n.kind=='interior');position=old.pos();w.delete_block(old)
            sampler=w.add_node(title,position)
            domain=next(n for n in w.scene.nodes() if n.kind=='domain');pde=next(n for n in w.scene.nodes() if n.kind=='pde')
            w.scene.connect_nodes(domain,sampler);w.scene.connect_nodes(sampler,pde)
            w.scene.clearSelection();sampler.setSelected(True);events();w.fit_model();events()
            if not w.validate_model():raise RuntimeError(w.validation.text())
            for cls in (QLabel,QLineEdit,QPlainTextEdit):
                for child in w.findChildren(cls):validate_public_text(child.toPlainText() if isinstance(child,QPlainTextEdit) else child.text(),kind)
            if kind=='importance':assert w.property_scroll.findChild(QDoubleSpinBox,'sampling_mixture') is not None
            w.property_scroll.widget().repaint();w.repaint();QTest.qWait(100);events()
            if not w.grab().save(str(paths[kind])):raise RuntimeError('Screenshot failed.')
        manifest.write_text(json.dumps(dict(scope='Actual Qt adaptive sampler inspectors, light theme; no training run.',
            captures={kind:dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for kind,path in paths.items()}),indent=2)+'\n')
        print('Captured both adaptive sampler inspectors.')
    finally:
        app.clipboard().clear();w.undo_stack.setClean();w.close();events()


if __name__=='__main__':main()
