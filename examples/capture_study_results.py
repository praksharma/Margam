"""Capture a real read-only study collection after small CLI trials; no accuracy claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit,QTableWidget
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from margam.ui.model_document import capture_model
    from margam.studies import catalogue
    from margam.study_document import capture,write
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];image=root/'assets/screenshots/manual-study-results.png';manifest=image.with_suffix('.json')
    if not args.overwrite and (image.exists() or manifest.exists()):parser.error('Capture exists; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');window=Window(state_directory=False)
    try:
        window.set_application_device('cpu');window.load_burgers_reference_example()
        for key,value in [('points',8),('boundary_points',4),('boundary_sampling',{}),('layers',(4,)),('iterations',2)]:window.change(key,value)
        model=capture_model(window);owner=next(row for row in catalogue(model) if row['parameter']=='layers')
        study,files=capture(model,[dict(node_id=owner['node_id'],parameter='layers',mode='candidates',text='[[4],[-1],[8]]')],window.project_workspace,execution=dict(device='cpu',wall_time_seconds=0),scene=window.scene)
        with TemporaryDirectory(prefix='margam-study-capture-') as folder:
            directory=Path(folder);source=directory/'input.margam-study';output=directory/'Burgers-network-study';write(source,study,files)
            child=subprocess.run([sys.executable,'-m','margam.cli','study','run',str(source),'--output',str(output)],capture_output=True,text=True,timeout=120)
            if child.returncode!=1:raise RuntimeError('Expected a mixed-grid CLI outcome: '+child.stderr)
            before=capture_model(window);index=window.undo_stack.index();dialog=window.show_study_results();assert dialog.open_folder(output)
            deadline=time.monotonic()+45
            while dialog.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
            assert dialog.collection is not None,dialog.status.text()
            assert dialog.collection['report']['counts']=={'completed':2,'invalid':1}
            assert before==capture_model(window) and index==window.undo_stack.index() and window.result is None
            dialog.details.setMaximumHeight(210)
            for _ in range(4):app.processEvents()
            for cls in (QLabel,QLineEdit,QPlainTextEdit):
                for child_widget in dialog.findChildren(cls):validate_public_text(child_widget.toPlainText() if isinstance(child_widget,QPlainTextEdit) else child_widget.text(),'Study collection capture')
            for table in dialog.findChildren(QTableWidget):
                for r in range(table.rowCount()):
                    for c in range(table.columnCount()):
                        if table.item(r,c):validate_public_text(table.item(r,c).text(),'Study collection table')
            assert dialog.grab().save(str(image))
            report=dialog.collection['report']
            evidence=dict(generator='examples.capture_study_results',scope='Actual Qt collection of a real three-trial CLI study; two updates per valid trial, not solution accuracy. Browsing leaves the current model unchanged.',
                counts=report['counts'],headless=report['headless'],source_sha256=report['source']['sha256'],study_id=report['study_id'],
                runtime=report['runtime'],capture_tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                file=image.name,sha256=hashlib.sha256(image.read_bytes()).hexdigest())
            text=json.dumps(evidence,indent=2)+'\n';validate_public_text(text,'Study collection capture evidence');manifest.write_text(text);dialog.close()
        print('Captured study collection from real CLI outcomes; current model preserved.')
    finally:window.undo_stack.setClean();window.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
