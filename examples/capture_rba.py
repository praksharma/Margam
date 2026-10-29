"""Capture actual RBA controls, compact recipe and raw/attention-weighted monitor."""
import argparse,hashlib,json,os,time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args();os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit
    from PySide6.QtTest import QTest
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root);out=root/'assets/screenshots';paths={name:out/('manual-rba-'+name+'.png') for name in ('trainer','compact','monitor')};manifest=out/'manual-rba-captures.json'
    if not args.overwrite and any(path.exists() for path in (*paths.values(),manifest)):parser.error('Captures exist; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False);w.resize(1560,1200);w.show()
    def save(name,widget):
        for _ in range(4):app.processEvents()
        QTest.qWait(100)
        for cls in (QLabel,QLineEdit,QPlainTextEdit):
            for child in widget.findChildren(cls):validate_public_text(child.toPlainText() if isinstance(child,QPlainTextEdit) else child.text(),name)
        if not widget.grab().save(str(paths[name])):raise RuntimeError('Capture failed.')
    try:
        w.load_example('rba_helmholtz_2d');assert w.validate_model();trainer=next(n for n in w.scene.nodes() if n.kind=='rba_method');w.show_block_properties(trainer);w.fit_model();save('trainer',w)
        compact=w.collapse_recipe(trainer);assert compact;w.show_block_properties(compact);w.fit_model();save('compact',w)
        w.expand_recipe(compact);w.change('iterations',30);assert w.start_training();deadline=time.monotonic()+40
        while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
        assert w.worker is None and w.result is not None,w.log.toPlainText()
        monitor=next(n for n in w.scene.nodes() if n.kind=='monitor');w.open_block(monitor);app.processEvents();dock,view=w.output_views.entries[monitor.identifier];dock.setFloating(True);dock.resize(1000,380);app.processEvents();save('monitor',view)
        manifest.write_text(json.dumps(dict(scope='Actual isolated RBA controls and raw/weighted loss monitor after 30 CPU updates; no accuracy claim.',captures={name:dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in paths.items()}),indent=2)+'\n')
        print('Captured RBA trainer, compact recipe and monitor.')
    finally:app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()


if __name__=='__main__':main()
