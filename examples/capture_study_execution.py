"""Exercise and capture the GUI-controlled CLI study without personal state changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from margam.ui.model_document import capture_model
    from margam.studies import catalogue
    from margam.study_document import capture,write
    from margam.execution.study_runner import source_identity
    from margam.ui.run_document import load_run_payload
    from examples.documentation import validate_public_text
    import torch
    root=Path(__file__).resolve().parents[1];output=args.output.absolute()
    if output.exists():parser.error('Choose a fresh workflow folder.')
    image=root/'assets/screenshots/manual-study-execution.png';manifest=image.with_suffix('.json')
    if not args.overwrite and (image.exists() or manifest.exists()):parser.error('Capture exists; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');window=Window(state_directory=False);dialog=None
    try:
        window.set_application_device('cpu');window.load_burgers_reference_example()
        for key,value in [('points',8),('boundary_points',4),('boundary_sampling',{}),('layers',(4,)),('iterations',2)]:window.change(key,value)
        model=capture_model(window);owner=next(row for row in catalogue(model) if row['parameter']=='iterations')
        study,files=capture(model,[dict(node_id=owner['node_id'],parameter='iterations',mode='candidates',text='[2,3]')],window.project_workspace,execution=dict(device='cpu',wall_time_seconds=0),scene=window.scene)
        output.mkdir(parents=True);source=output/'input.margam-study';write(source,study,files)
        before=capture_model(window);index=window.undo_stack.index();rng=torch.get_rng_state().clone();dialog=window.show_study_execution()
        # Relative paths keep documentation captures free of local home directories.
        dialog.source.setText(os.path.relpath(source,Path.cwd()));dialog.output.setText(os.path.relpath(output/'results',Path.cwd()));dialog.exports.setChecked(True)
        def wait(predicate):
            deadline=time.monotonic()+90
            while not predicate() and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
            assert predicate(),dialog.status.text()
        assert dialog.review();wait(lambda:dialog.reader is None);assert dialog.prepared is not None,dialog.status.text()
        assert dialog.start();wait(lambda:not dialog.active)
        report=json.loads((output/'results/summary.json').read_text());assert report['counts']=={'completed':2} and report['headless']
        assert report['study_archive_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
        assert before==capture_model(window) and index==window.undo_stack.index() and window.result is None
        assert torch.equal(rng,torch.get_rng_state()) and not window.scene.locked
        for number,steps in ((1,2),(2,3)):
            data,_=load_run_payload(output/f'results/DP{number:04d}/artifacts/run.margam-run.pt')
            assert data['result']['steps']==steps and len(data['owned_files'])==1
            detail=json.loads((output/f'results/DP{number:04d}/trial.json').read_text());assert len(detail['exports'])==2
        for _ in range(4):app.processEvents()
        for cls in (QLabel,QLineEdit,QPlainTextEdit):
            for widget in dialog.findChildren(cls):validate_public_text(widget.toPlainText() if isinstance(widget,QPlainTextEdit) else widget.text(),'Study execution capture')
        assert dialog.grab().save(str(image))
        record=dict(generator='examples.capture_study_execution',scope='Actual GUI-controlled headless CPU CLI study, two/three updates with owned reference inputs and explicit exports; software workflow, not accuracy.',
            source_sha256=source_identity()['sha256'],capture_tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            headless=report['headless'],counts=report['counts'],current_model_preserved=True,parent_rng_preserved=True,
            file=image.name,sha256=hashlib.sha256(image.read_bytes()).hexdigest())
        text=json.dumps(record,indent=2)+'\n';validate_public_text(text,'Study execution capture manifest');manifest.write_text(text)
        print('Captured real GUI-controlled CLI study; model/results/RNG preserved.')
    finally:
        if dialog is not None and dialog.active:
            dialog.stop(force=True)
            while dialog.active:app.processEvents();time.sleep(.005)
        window.undo_stack.setClean();window.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
