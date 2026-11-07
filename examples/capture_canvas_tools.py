"""Capture real presentation tools without training or personal desktop state."""
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
    from margam.ui.model_document import capture_model
    from margam.ui.theme import configure_theme
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    paths={key:Path('assets/screenshots/manual-'+key+'.png') for key in ('canvas-notes','dependencies','group-menu','blank-model')}
    manifest=Path('assets/screenshots/manual-canvas-tools-capture.json')
    if not args.overwrite and any(p.exists() for p in [*paths.values(),manifest]):parser.error('Capture exists; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False)
    def settle():
        for _ in range(5):app.processEvents()
    def capture(widget,key):
        for cls in (QLabel,QLineEdit,QPlainTextEdit):
            for child in widget.findChildren(cls):
                if child.isVisible():validate_public_text(child.toPlainText() if hasattr(child,'toPlainText') else child.text(),'canvas tools capture')
        assert widget.grab().save(str(paths[key]))
    try:
        w.resize(1400,900);w.set_application_device('cpu');w.show();settle()
        note=w.add_node('Note');note.setPos(0,-180)
        w.change_note(note,'Poisson demonstration\nCoordinates: x, y\nRecord units and assumptions here.\nBoundary arrows select PDE rows;\nBC operators remain separate.')
        w.show_grid_action.setChecked(True);w.snap_grid_action.setChecked(True);w.fit_model();settle();capture(w,'canvas-notes')
        pde=next(n for n in w.scene.nodes() if n.kind=='pde')
        w.scene.clearSelection();pde.setSelected(True);w.set_dependency_mode('upstream');settle();capture(w,'dependencies')
        assert not note.dependency_highlight
        boundary=next(n for n in w.scene.nodes() if n.kind=='boundary');boundary.setSelected(True)
        menu=w.create_group_menu();menu.popup(w.mapToGlobal(w.rect().center()));settle();capture(menu,'group-menu');menu.close()
        doc=capture_model(w);assert doc['version']==31
        w.undo_stack.setClean();assert w.new_blank_model();w.show_grid_action.setChecked(False);w.snap_grid_action.setChecked(False);settle();capture(w,'blank-model')
        assert not w.scene.nodes() and not w.scene.wires and w.result is None
        data=dict(generator='examples.capture_canvas_tools',scope='Actual notes, dependency navigation, grid, group menu and blank model. No training or inference.',
                  model_version=31,notes_are_plain_text=True,files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths.values()})
        manifest.write_text(json.dumps(data,indent=2)+'\n');print('Captured notes, dependencies, group actions and blank model.')
    finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()

if __name__=='__main__':main()
