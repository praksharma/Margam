"""Capture real Burgers setup, initial conditions and optional result table."""
import argparse
import hashlib,json,os,time
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
    paths={name:out/('manual-burgers-'+name+'.png') for name in ('pde','initial','results')};manifest=out/'manual-burgers-captures.json'
    if not args.overwrite and any(p.exists() for p in (*paths.values(),manifest)):parser.error('Captures exist; use --overwrite to replace them.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False);w.resize(1560,1100);w.show()
    def events():
        for _ in range(4):app.processEvents()
    def save(name):
        events();w.property_scroll.widget().repaint();w.repaint();QTest.qWait(100)
        for cls in (QLabel,QLineEdit,QPlainTextEdit):
            for child in w.findChildren(cls):validate_public_text(child.toPlainText() if isinstance(child,QPlainTextEdit) else child.text(),name)
        if not w.grab().save(str(paths[name])):raise RuntimeError('Capture failed.')
    def finish():
        deadline=time.monotonic()+30
        while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
        if w.worker is not None:raise RuntimeError('Training timeout.')
    try:
        w.load_example('burgers_1d');events();w.fit_model();save('pde')
        bc=next(n for n in w.scene.nodes() if n.kind=='bc');w.scene.clearSelection();bc.setSelected(True);events();save('initial')
        w.change('points',128);w.change('layers',(16,16));w.change('iterations',50)
        assert w.start_training();finish();assert w.result is not None,w.log.toPlainText()
        solution=next(n for n in w.scene.nodes() if n.kind=='solution');w.open_block(solution);events();w.fit_model();save('results')
        manifest.write_text(json.dumps(dict(scope='Actual Burgers setup/IC and optional result table after 50 CPU updates; no accuracy claim.',
            captures={name:dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for name,p in paths.items()}),indent=2)+'\n')
        print('Captured Burgers PDE, initial condition and result table.')
    finally:
        if w.worker is not None:w.cancel_training();finish()
        app.clipboard().clear();w.undo_stack.setClean();w.close();events()


if __name__=='__main__':main()
