"""Capture actual graph explanations and result states using isolated CPU fixtures."""
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
    from margam.ui.run_summary import execution_plan
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    paths={key:Path('assets/screenshots/manual-'+key+'.png') for key in ('wire-inspection','settings-search','run-summary','result-freshness')}
    manifest=Path('assets/screenshots/manual-graph-explanations-capture.json')
    if not args.overwrite and any(p.exists() for p in [*paths.values(),manifest]):parser.error('Capture exists; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False)
    def settle():
        for _ in range(5):app.processEvents()
    def capture(widget,key):
        for cls in (QLabel,QLineEdit,QPlainTextEdit):
            for child in widget.findChildren(cls):
                if child.isVisible():validate_public_text(child.toPlainText() if hasattr(child,'toPlainText') else child.text(),'graph explanations capture')
        assert widget.grab().save(str(paths[key]))
    try:
        w.set_application_device('cpu');w.resize(1400,900);w.show();settle()
        pde=next(n for n in w.scene.nodes() if n.kind=='pde')
        wire=next(wire for wire in w.scene.wires if wire.target is pde.inputs['boundary_points'])
        w.scene.clearSelection();wire.setSelected(True);settle();capture(w,'wire-inspection')
        search=w.show_settings_search();search.search.setText('learning rate');settle();search.locate();settle();capture(search,'settings-search');search.close()
        summary=w.show_run_summary();settle();capture(summary,'run-summary')
        _,counts,_=execution_plan(w);summary.close()
        w.load_heat_template();w.change('iterations',2);w.change('layers',(4,));assert w.start_training()
        while w.worker is not None:app.processEvents()
        assert w.result is not None
        monitor=next(n for n in w.scene.nodes() if n.kind=='monitor');w.output_views.open(monitor)
        outputs=[n for n in w.scene.nodes() if n.kind in ('monitor','solution','comparison')]
        assert all(n.result_status=='Current' for n in outputs)
        w.change('source',3);assert all(n.result_status=='Settings changed' for n in outputs)
        w.scene.clearSelection();monitor.setSelected(True);w.fit_model();settle();capture(w,'result-freshness')
        data=dict(generator='examples.capture_graph_explanations',scope='Actual graph explanations and stale-result clearing; two CPU optimizer updates are a lifecycle fixture, not an accuracy benchmark.',
                  initial_plan=counts,result_statuses={n.kind:n.result_status for n in outputs},
                  files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths.values()})
        manifest.write_text(json.dumps(data,indent=2)+'\n');print('Captured graph explanations, setting navigation, run plan and result freshness.')
    finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()

if __name__=='__main__':main()
