"""Capture actual source blocks, named boundary mappings and downstream conditions."""
import argparse,hashlib,json,os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args();os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit
    from PySide6.QtTest import QTest
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from margam.ui.data_mapping import DataMappingDialog
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root);out=root/'assets/screenshots';paths={name:out/('manual-training-data-'+name+'.png') for name in ('blocks','mapping','conditions')};manifest=out/'manual-training-data-captures.json'
    if not args.overwrite and any(path.exists() for path in (*paths.values(),manifest)):parser.error('Captures exist; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False);w.resize(1560,1100);w.show()
    def save(name,widget):
        for _ in range(4):app.processEvents()
        QTest.qWait(100)
        for cls in (QLabel,QLineEdit,QPlainTextEdit):
            for child in widget.findChildren(cls):validate_public_text(child.toPlainText() if isinstance(child,QPlainTextEdit) else child.text(),name)
        if not widget.grab().save(str(paths[name])):raise RuntimeError('Capture failed.')
    try:
        assert w.load_training_data_example() and w.validate_model(),w.validation.text()
        w.select_boundary_side('slope');w.fit_model();save('blocks',w)
        dialog=DataMappingDialog(2,w.config.boundary_data['slope'],w);dialog.show();assert dialog.check();save('mapping',dialog);dialog.close();dialog.deleteLater()
        w.show_block_properties(next(n for n in w.scene.nodes() if n.kind=='bc'));save('conditions',w.properties_dock)
        manifest.write_text(json.dumps(dict(scope='Actual isolated prepared triangle UI, named-set mapping and downstream scalar boundary controls; no accuracy claim.',captures={name:dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in paths.items()}),indent=2)+'\n')
        print('Captured prepared training sources, mapping and boundary conditions.')
    finally:app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()


if __name__=='__main__':main()
