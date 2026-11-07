"""Capture the cleanup review using isolated temporary project copies only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    path=Path('assets/screenshots/manual-dataset-cleanup.png');manifest=path.with_suffix('.json')
    if not args.overwrite and (path.exists() or manifest.exists()):parser.error('Capture exists; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False);w.change('shape','Interval');w.undo_stack.clear()
    try:
        with TemporaryDirectory() as directory:
            root=Path(directory);sources=[]
            for name in ('old_points','current_points','unused_preview'):
                source=root/(name+'.csv');source.write_text('x\n0.2\n0.8\n')
                sources.append(w.project_workspace.store_dataset(dict(path=str(source),coordinates='x',values='',normals=''),copy=True,validator=lambda mapping:None))
            w.change('interior_data',sources[0]);w.change('interior_data',sources[1])
            dialog=w.show_dataset_cleanup();dialog.select_unused();dialog.resize(1130,520)
            for _ in range(5):app.processEvents()
            for child in dialog.findChildren(QLabel):
                if child.isVisible():validate_public_text(child.text(),'dataset cleanup capture')
            assert len(dialog.rows)==3 and len(dialog.selected())==1
            assert dialog.grab().save(str(path))
            manifest.write_text(json.dumps(dict(generator='examples.capture_dataset_cleanup',scope='Actual review of isolated current, undo-protected and unused import fixtures. No user files removed, no numerical training.',
                rows=len(dialog.rows),selected=len(dialog.selected()),file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+'\n')
            print('Captured unused-import cleanup review.')
    finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
