"""Capture current, real Qt walkthrough images without personal desktop state."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import subprocess,time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtCore import Qt,QPoint
    from PySide6.QtWidgets import QApplication,QComboBox,QLineEdit,QPlainTextEdit,QLabel
    from margam.ui.app import Window
    from margam.ui.data_mapping import DataMappingDialog
    from margam.ui.theme import configure_theme
    from tempfile import TemporaryDirectory
    root=Path(__file__).resolve().parents[1];os.chdir(root);out=root/'assets/screenshots'
    names=['workspace','library','examples','examples-1d','domain','samples','interior','boundary-points',
           'boundary-condition','map-data','helmholtz-pde','network-choices','fourier-settings','sa-settings',
           'training-settings','expanded','compact','monitor','comparison','save-menu','paraview','help']
    paths={n:out/('manual-'+n+'.png') for n in names};manifest=out/'manual-captures.json'
    if not args.overwrite and any(p.exists() for p in [*paths.values(),manifest]):parser.error('Manual captures exist; explicitly use --overwrite to replace them.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False);w.resize(1480,1000);w.show()
    def events():
        for _ in range(4):app.processEvents()
    def select(node):w.scene.clearSelection();node.setSelected(True);events()
    def save(widget,name):
        events()
        from examples.documentation import validate_public_text
        for child in widget.findChildren(QLineEdit):validate_public_text(child.text(),name)
        for child in widget.findChildren(QPlainTextEdit):validate_public_text(child.toPlainText(),name)
        for child in widget.findChildren(QLabel):validate_public_text(child.text(),name)
        if not widget.grab().save(str(paths[name])):raise RuntimeError('Capture failed: '+name)
    def popup(menu,name):
        menu.popup(w.mapToGlobal(QPoint(20,50)));save(menu,name);menu.close();events()
    def wait():
        deadline=time.monotonic()+45
        while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
        if w.worker is not None:raise RuntimeError('Training timeout')
    try:
        events();save(w,'workspace');save(w.library_dock,'library')
        popup(w.examples_menu,'examples');popup(w.examples_1d_menu,'examples-1d')
        domain=next(n for n in w.scene.nodes() if n.kind=='domain');select(domain);save(w.properties_dock,'domain')
        w.open_rectangle_preview(domain,True);events()
        preview=next(iter(w.rectangle_previews.values()))[0];save(preview,'samples');preview.hide()
        for kind,name in [('interior','interior'),('boundary','boundary-points'),('bc','boundary-condition')]:
            select(next(n for n in w.scene.nodes() if n.kind==kind));save(w.properties_dock,name)
        dialog=DataMappingDialog(2,dict(path='assets/examples/manual-interior.csv',coordinates='x,y',normals='',values=''),w,coordinate_only=True)
        dialog.show();assert dialog.check();save(dialog,'map-data');dialog.reject()
        w.load_example('helmholtz_lra_2d');select(next(n for n in w.scene.nodes() if n.kind=='predefined_pde'));save(w.properties_dock,'helmholtz-pde')
        network=next(n for n in w.scene.nodes() if n.kind=='network');select(network)
        combo=w.property_scroll.findChild(QComboBox,'network_architecture');combo.showPopup();save(combo.view().window(),'network-choices');combo.hidePopup()
        save(w.properties_dock,'fourier-settings')
        w.load_example('self_adaptive_poisson');select(next(n for n in w.scene.nodes() if n.kind=='sa_method'));save(w.properties_dock,'sa-settings')
        w.load_example('interval_poisson');trainer=next(n for n in w.scene.nodes() if n.kind=='method');select(trainer)
        save(w.properties_dock,'training-settings');w.fit_model();save(w,'expanded')
        config=asdict(w.scene.compile_config());compact=w.collapse_recipe(trainer);select(compact);w.fit_model();save(w,'compact')
        assert asdict(w.scene.compile_config())==config
        trainer=w.expand_recipe(compact);select(trainer);assert asdict(w.scene.compile_config())==config
        assert not w.output_views.entries and w.start_training();wait()
        assert w.result is not None and w.result['steps']==config['iterations'] and not w.output_views.entries,w.log.toPlainText()
        for kind,name in [('monitor','monitor'),('comparison','comparison')]:
            node=next(n for n in w.scene.nodes() if n.kind==kind);select(node);w.open_block(node);events()
            w.resizeDocks([w.output_views.entries[node.identifier][0]],[300],Qt.Orientation.Vertical);events();w.fit_model();save(w,name)
        comparison=next(n for n in w.scene.nodes() if n.kind=='comparison');metric=w.result['outputs'][comparison.identifier]['rmse']
        with TemporaryDirectory() as directory:
            assert w.save_model(Path(directory)/'first-pinn.margam.json')
            assert w.save_current_run(Path(directory)/'first-pinn.margam-run.pt')
            popup(w.menuBar().actions()[0].menu(),'save-menu')
        export=w.add_node('ParaView Export');export.setPos(1000,650);w.scene.connect_nodes(trainer,export);select(export);save(w.properties_dock,'paraview')
        popup(w.menuBar().actions()[-1].menu(),'help')
        import torch,PySide6
        report=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),application_parent=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                    tracked_changes=subprocess.run(['git','diff','--quiet','HEAD'],cwd=root).returncode!=0,
                    scope='Actual Qt walkthrough captures and one default baseline run; not a numerical benchmark.',
                    qt_platform=app.platformName(),pyside=PySide6.__version__,torch=str(torch.__version__),configuration=config,
                    run_steps=config['iterations'],observed_comparison_rmse=metric,
                    capture_tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    captures={n:dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in paths.items()})
        manifest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(f'Captured {len(paths)} real UI screenshots and verified one default baseline run.')
    finally:
        if w.worker is not None:w.cancel_training();wait()
        app.clipboard().clear();w.undo_stack.setClean();w.close();events()


if __name__=='__main__':main()
