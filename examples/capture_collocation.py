"""Capture the three explicit PDE point wiring choices and verify their row counts."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from margam.templates.poisson import build_pinn
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root);output=root/'assets/screenshots'
    paths={name:output/('manual-collocation-'+name+'.png') for name in ('both','interior','boundary')};manifest=output/'manual-collocation-captures.json'
    if not args.overwrite and any(path.exists() for path in (*paths.values(),manifest)):parser.error('Captures exist; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');window=Window(state_directory=False);window.resize(1560,1000);window.show()
    counts={}
    try:
        pde=next(node for node in window.scene.nodes() if node.kind=='pde')
        boundary=next(node for node in window.scene.nodes() if node.kind=='boundary')
        def save(name):
            assert window.validate_model() and window.config.pde_points==name
            model=build_pinn(window.config);counts[name]=len(model.collocation_point_sample)
            window.show_block_properties(pde);window.fit_model()
            for _ in range(5):app.processEvents()
            for cls in (QLabel,QLineEdit,QPlainTextEdit):
                for child in window.findChildren(cls):validate_public_text(child.toPlainText() if isinstance(child,QPlainTextEdit) else child.text(),name)
            assert window.grab().save(str(paths[name]))
        save('both')
        wire=next(wire for wire in window.scene.wires if wire.target.parentItem() is pde and wire.target.name=='boundary_points')
        window.scene.wires.remove(wire);window.scene.removeItem(wire);window.graph_changed();save('interior')
        window.delete_block(next(node for node in window.scene.nodes() if node.kind=='interior'))
        window.scene.connect_nodes(boundary,pde,'input');save('boundary')
        assert counts['both']==counts['interior']+counts['boundary']
        manifest.write_text(json.dumps(dict(generator='examples.capture_collocation',counts=counts,files={path.name:dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths.values()}),indent=2)+'\n')
    finally:window.undo_stack.setClean();window.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
