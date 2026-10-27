"""Capture real Margam widgets for docs, without reading/writing personal UI state.

Run from the repository root with the existing .venv. Captures one small CPU run.
"""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('assets/screenshots'))
    parser.add_argument('--overwrite',action='store_true',help='Explicitly replace this guide\'s existing captures')
    args=parser.parse_args()
    names=['file-menu','architectures','highway-settings','siren-settings','training-settings',
           'recipe-expanded','recipe-compact','training-monitor','interval-comparison']
    paths={name:args.output_dir/('guide-'+name+'.png') for name in names}
    manifest_path=args.output_dir/'guide-captures.json'
    if not args.overwrite and any(p.exists() for p in [*paths.values(),manifest_path]):
        parser.error('Guide captures already exist; choose a fresh output directory or explicitly use --overwrite.')
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
    from PySide6.QtCore import Qt,QPoint
    from PySide6.QtWidgets import QApplication,QComboBox
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    app=QApplication([]);w=Window(state_directory=False)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    def events():
        for _ in range(4):app.processEvents()
    def save(widget,name):
        events();pixmap=widget.grab()
        if pixmap.isNull() or not pixmap.save(str(paths[name])):raise RuntimeError('Could not capture '+name)
    def select(node):
        w.scene.clearSelection();node.setSelected(True);w.view.setFocus();events()
    def wait_worker():
        deadline=time.monotonic()+45
        while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
        if w.worker is not None:raise RuntimeError('Training did not finish within 45 seconds.')
    try:
        configure_theme(app,'light');w.resize(1480,950);w.show();events()
        file_menu=w.menuBar().actions()[0].menu()
        file_menu.popup(w.mapToGlobal(QPoint(10,25)));events();save(file_menu,'file-menu');file_menu.close()
        w.load_example('self_adaptive_poisson');events()
        network=next(n for n in w.scene.nodes() if n.kind=='network');select(network)
        combo=w.property_scroll.findChild(QComboBox,'network_architecture')
        combo.showPopup();events();save(combo.view().window(),'architectures');combo.hidePopup()
        w.change_network_type('highway_fourier');events();save(w.properties_dock,'highway-settings')
        w.change_network_type('siren');w.change_network_option('first_omega',10.);w.change_network_option('omega',10.)
        select(network);save(w.properties_dock,'siren-settings')
        root=next(n for n in w.scene.nodes() if n.kind=='sa_method');select(root)
        w.change('iterations',700);w.change('lr',.001)
        for node in w.scene.nodes():
            if node.kind in ('solution','comparison','export'):w.change_output(node,'resolution',201)
        select(root);save(w.properties_dock,'training-settings')
        config=asdict(w.scene.compile_config());w.fit_model();save(w,'recipe-expanded')
        compact=w.collapse_recipe(root)
        assert asdict(w.scene.compile_config())==config
        w.fit_model();save(w,'recipe-compact')
        root=w.expand_recipe(compact)
        assert asdict(w.scene.compile_config())==config
        assert not w.output_views.entries
        assert w.start_training();wait_worker()
        assert w.result is not None,w.log.toPlainText()
        assert w.result['steps']==700 and not w.output_views.entries
        monitor=next(n for n in w.scene.nodes() if n.kind=='monitor')
        comparison=next(n for n in w.scene.nodes() if n.kind=='comparison')
        select(monitor);w.open_block(monitor);events()
        w.resizeDocks([w.output_views.entries[monitor.identifier][0]],[280],Qt.Orientation.Vertical)
        events();w.fit_model();save(w,'training-monitor')
        select(comparison);w.open_block(comparison);events();w.fit_model();save(w,'interval-comparison')
        result=w.result['outputs'][comparison.identifier]
        assert result['rmse']<.003 and len(result['values'])==201
        source_root=Path(__file__).resolve().parents[1]
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source_root,text=True).strip()
        app_modified=subprocess.run(['git','diff','--quiet','HEAD','--','margam'],cwd=source_root).returncode!=0
        import torch,PySide6
        report=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),application_commit=revision,
                    application_tracked_changes=app_modified,qt_platform=app.platformName(),
                    torch=str(torch.__version__),pyside=PySide6.__version__,theme='light',window_size=[w.width(),w.height()],
                    capture_tool=dict(file='examples/capture_ui_docs.py',sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                    configuration=config,comparison=dict(steps=result['steps'],evaluation_points=201,rmse=result['rmse'],
                                                         reference='x*(1-x)',reference_expression='x*(1-x)'),
                    numerical_scope='One actual CPU screenshot demonstration; not a new multi-seed architecture benchmark.',
                    captures={name:dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in paths.items()})
        manifest_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(f"Captured {len(paths)} real UI images. SIREN/SA demonstration RMSE: {result['rmse']:.8g}")
        print('Capture settings and provenance: '+str(manifest_path))
    finally:
        if w.worker is not None:w.cancel_training();wait_worker()
        app.clipboard().clear();w.undo_stack.setClean();w.close();events()


if __name__=='__main__':main()
