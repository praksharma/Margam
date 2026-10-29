"""Capture real Jorvik dataset selection and the Margam geometry block."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit
    from margam.ui.app import Window
    from margam.ui.jorvik_geometry import DatasetDialog
    from margam.ui.theme import configure_theme
    from examples.jorvik_exchange import sample_annulus
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root);output=root/'assets/screenshots'
    paths={name:output/('manual-jorvik-'+name+'.png') for name in ('geometry','datasets')};manifest=output/'manual-jorvik-captures.json'
    if not args.overwrite and any(path.exists() for path in (*paths.values(),manifest)):parser.error('Captures exist; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');window=Window(state_directory=False);window.resize(1480,1000);window.show()
    def save(widget,name):
        for _ in range(5):app.processEvents()
        for cls in (QLabel,QLineEdit,QPlainTextEdit):
            for child in widget.findChildren(cls):validate_public_text(child.toPlainText() if isinstance(child,QPlainTextEdit) else child.text(),name)
        assert widget.grab().save(str(paths[name]))
    try:
        with TemporaryDirectory() as directory:
            export,selections=sample_annulus(directory)
            for node in list(window.scene.nodes()):
                if node.kind in ('domain','interior','boundary'):window.delete_block(node)
            geometry=window.add_node('Jorvik geometry')
            import shutil
            shutil.copyfile(export,window.jorvik_document(geometry).with_suffix('.nc'))
            dialog=DatasetDialog(export,{},window)
            for label,choice,alias in dialog.rows:
                choice.setCurrentText(selections[label][0].title());alias.setText(label)
            dialog.resize(700,430);dialog.show();save(dialog,'datasets');dialog.close()
            assert window.update_jorvik(geometry,selections,confirm=False)
            interior=next(n for n in window.scene.nodes() if n.kind=='interior_data');boundary=next(n for n in window.scene.nodes() if n.kind=='boundary_data')
            window.scene.connect_nodes(interior,next(n for n in window.scene.nodes() if n.kind=='pde'))
            window.scene.connect_nodes(boundary,next(n for n in window.scene.nodes() if n.kind=='bc'))
            positions={'jorvik_geometry':(0,160),'interior_data':(240,40),'boundary_data':(240,270),'pde':(480,40),'bc':(480,270),'network':(480,440),'method':(730,160)}
            for node in window.scene.nodes():node.setPos(*positions[node.kind])
            assert window.validate_model();window.show_block_properties(geometry);window.fit_model();save(window,'geometry')
        manifest.write_text(json.dumps(dict(generator='examples.capture_jorvik',case='Actual Jorvik annulus export; Margam-owned editor document starts empty.',files={path.name:dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths.values()}),indent=2)+'\n')
    finally:window.undo_stack.setClean();window.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
