"""Capture state-isolated model search and diagram layout controls."""
import argparse
import hashlib
import json
import os
from tempfile import TemporaryDirectory
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    paths=[Path('assets/screenshots')/name for name in ('manual-model-search.png','manual-format.png','manual-recent-models.png')]
    manifest=paths[0].with_name('manual-diagram-tools-capture.json')
    if not args.overwrite and any(p.exists() for p in paths+[manifest]):parser.error('Capture exists; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False)
    try:
        w.resize(1280,800);w.show()
        for _ in range(4):app.processEvents()
        w.arrange_blocks('auto');w.fit_model()
        dialog=w.show_model_search();dialog.search.setText('PDE')
        for _ in range(3):app.processEvents()
        assert dialog.results.count()==1
        dialog.locate();assert dialog.grab().save(str(paths[0]));dialog.close()
        w.select_all_blocks();w.update_layout_actions()
        menu=next(action.menu() for action in w.menuBar().actions() if action.text()=='Format')
        menu.popup(w.menuBar().mapToGlobal(w.menuBar().actionGeometry(menu.menuAction()).bottomLeft()))
        for _ in range(3):app.processEvents()
        assert menu.grab().save(str(paths[1]));menu.close()
        with TemporaryDirectory() as directory:
            assert w.save_model(Path(directory)/'Poisson.margam.json')
            assert w.save_model(Path(directory)/'Experiment.margam.json')
            w.refresh_recent_models();menu=w.recent_models_menu
            menu.popup(w.mapToGlobal(w.rect().center()))
            for _ in range(3):app.processEvents()
            assert menu.grab().save(str(paths[2]));menu.close()
        data=dict(generator='examples.capture_diagram_tools',scope='Actual model search, Format and recent-model menus; no numerical accuracy claim.',files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths})
        manifest.write_text(json.dumps(data,indent=2)+'\n')
        print('Captured model search and Format menu.')
    finally:
        app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()

if __name__=='__main__':main()
