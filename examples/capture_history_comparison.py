"""Capture a real retained-monitor overlay with unmodified update indices."""
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
    from PySide6.QtWidgets import QApplication,QLabel,QPlainTextEdit,QTableWidget
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from margam.ui.model_document import capture_model
    from margam.studies import catalogue
    from margam.study_document import capture,write
    from margam.execution.study_runner import source_identity
    from examples.documentation import validate_public_text
    import torch
    root=Path(__file__).resolve().parents[1];image=root/'assets/screenshots/manual-history-comparison.png';manifest=image.with_suffix('.json')
    if not args.overwrite and (image.exists() or manifest.exists()):parser.error('Capture exists; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');window=Window(state_directory=False)
    try:
        window.set_application_device('cpu');window.load_burgers_reference_example()
        for key,value in [('points',8),('boundary_points',4),('boundary_sampling',{}),('layers',(4,)),('iterations',30)]:window.change(key,value)
        next(node for node in window.scene.nodes() if node.kind=='monitor').settings['history_limit']=25
        model=capture_model(window);owner=next(row for row in catalogue(model) if row['parameter']=='iterations')
        study,files=capture(model,[dict(node_id=owner['node_id'],parameter='iterations',mode='candidates',text='[30,45]')],window.project_workspace,execution=dict(device='cpu',wall_time_seconds=0),scene=window.scene)
        with TemporaryDirectory(prefix='margam-history-capture-') as folder:
            directory=Path(folder);source=directory/'input.margam-study';output=directory/'run';write(source,study,files)
            child=subprocess.run([sys.executable,'-m','margam.cli','study','run',str(source),'--output',str(output)],capture_output=True,text=True,timeout=120)
            if child.returncode:raise RuntimeError(child.stderr)
            before=capture_model(window);index=window.undo_stack.index();rng=torch.get_rng_state().clone();dialog=window.show_history_comparison()
            for side in (0,1):
                assert dialog.choose_file(side,output/f'DP{side+1:04d}/artifacts/run.margam-run.pt')
                deadline=time.monotonic()+45
                while dialog.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
                assert dialog.runs[side] is not None,dialog.status.text()
            assert dialog.compare(),dialog.status.text();value=dialog.comparison
            assert value['a']['recorded']==value['b']['recorded']==25
            assert (value['a']['first_step'],value['a']['last_step'])==(6,30)
            assert (value['b']['first_step'],value['b']['last_step'])==(21,45)
            assert before==capture_model(window) and index==window.undo_stack.index() and window.result is None and torch.equal(rng,torch.get_rng_state())
            for _ in range(4):app.processEvents()
            for cls in (QLabel,QPlainTextEdit):
                for widget in dialog.findChildren(cls):validate_public_text(widget.toPlainText() if isinstance(widget,QPlainTextEdit) else widget.text(),'History capture')
            for table in dialog.findChildren(QTableWidget):
                for r in range(table.rowCount()):
                    for c in range(table.columnCount()):
                        if table.item(r,c):validate_public_text(table.item(r,c).text(),'History table capture')
            assert dialog.grab().save(str(image))
            record=dict(generator='examples.capture_history_comparison',scope='Real CPU CLI runs with 30/45 updates and only 25 retained monitor records each. Overlay preserves 6–30 and 21–45 indices; software history semantics, not accuracy.',
                source_sha256=source_identity()['sha256'],capture_tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),retained_ranges=[[6,30],[21,45]],
                current_model_preserved=True,parent_rng_preserved=True,file=image.name,sha256=hashlib.sha256(image.read_bytes()).hexdigest())
            text=json.dumps(record,indent=2,ensure_ascii=False)+'\n';validate_public_text(text,'History capture evidence');manifest.write_text(text);dialog.close()
        print('Captured retained-history overlay; original update indices/model/RNG preserved.')
    finally:window.undo_stack.setClean();window.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
