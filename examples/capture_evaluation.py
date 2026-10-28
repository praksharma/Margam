"""Capture actual supplied-point blocks, mapping and pointwise comparison table."""
import argparse,hashlib,json,os,time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args();os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit
    from PySide6.QtTest import QTest
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from margam.ui.evaluation_mapping import EvaluationMappingDialog
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root);out=root/'assets/screenshots';paths={name:out/('manual-evaluation-'+name+'.png') for name in ('blocks','mapping','comparison')};manifest=out/'manual-evaluation-captures.json'
    if not args.overwrite and any(path.exists() for path in (*paths.values(),manifest)):parser.error('Captures exist; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False);w.resize(1560,1150);w.show()
    def save(name,widget):
        for _ in range(4):app.processEvents()
        QTest.qWait(100)
        for cls in (QLabel,QLineEdit,QPlainTextEdit):
            for child in widget.findChildren(cls):validate_public_text(child.toPlainText() if isinstance(child,QPlainTextEdit) else child.text(),name)
        capture=widget.grab()
        if name=='comparison':capture=capture.copy(0,0,min(capture.width(),max(500,widget.table.horizontalHeader().length()+50)),capture.height())
        if not capture.save(str(paths[name])):raise RuntimeError('Capture failed.')
    try:
        w.load_evaluation_example();assert w.validate_model(),w.validation.text()
        points=next(n for n in w.scene.nodes() if n.kind=='evaluation_points');w.scene.clearSelection();points.setSelected(True);app.processEvents();w.fit_model();save('blocks',w)
        dialog=EvaluationMappingDialog(w.config,points.settings['source_data'],w);dialog.show();assert dialog.check();save('mapping',dialog);dialog.close();dialog.deleteLater()
        w.change('iterations',100);assert w.start_training();deadline=time.monotonic()+40
        while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
        assert w.worker is None and w.result is not None,w.log.toPlainText()
        comparison=next(n for n in w.scene.nodes() if n.kind=='comparison');w.open_block(comparison);app.processEvents();save('comparison',w.output_views.entries[comparison.identifier][1])
        manifest.write_text(json.dumps(dict(scope='Actual optional data blocks, mapping and four-row comparison after 100 CPU updates; no paper accuracy claim.',captures={name:dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in paths.items()}),indent=2)+'\n')
        print('Captured evaluation blocks, mapping and comparison table.')
    finally:
        app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()


if __name__=='__main__':main()
