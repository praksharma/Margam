"""Exercise the reviewed desktop probe without adopting numerical results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from margam.ui.model_document import capture_model
    from margam.execution.study_runner import source_identity
    from examples.documentation import validate_public_text
    import torch
    root=Path(__file__).resolve().parents[1];image=root/'assets/screenshots/manual-resource-probe.png';manifest=image.with_suffix('.json')
    if not args.overwrite and (image.exists() or manifest.exists()):parser.error('Capture exists; use --overwrite.')
    previous=Path.cwd();os.chdir(root);app=QApplication([]);configure_theme(app,'light');window=Window(state_directory=False);dialog=None
    try:
        window.set_application_device('cpu');model=capture_model(window);undo=window.undo_stack.index();rng=torch.get_rng_state().clone()
        source=Path('assets/studies/poisson-grid.margam-study');original=source.read_bytes();dialog=window.show_resource_probe();dialog.source.setText(source.as_posix());dialog.updates.setValue(2)
        def wait(predicate):
            deadline=time.monotonic()+90
            while not predicate() and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
            assert predicate(),dialog.status.text()
        assert dialog.review();wait(lambda:dialog.reader is None);assert dialog.prepared is not None,dialog.status.text()
        assert dialog.start();wait(lambda:not dialog.active);assert dialog.result is not None,dialog.text.toPlainText();assert dialog.result['completed_updates']==2
        assert dialog.copy_report() and json.loads(app.clipboard().text())==dialog.result
        assert source.read_bytes()==original and model==capture_model(window) and undo==window.undo_stack.index() and torch.equal(rng,torch.get_rng_state()) and window.result is None and not window.scene.locked
        for _ in range(4):app.processEvents()
        for cls in (QLabel,QLineEdit,QPlainTextEdit):
            for widget in dialog.findChildren(cls):validate_public_text(widget.toPlainText() if isinstance(widget,QPlainTextEdit) else widget.text(),'Resource probe capture')
        assert dialog.grab().save(str(image))
        record=dict(generator='examples.capture_resource_probe',scope='Actual desktop-reviewed CPU probe of two native updates; measured child scope only, not fit or solution accuracy. No trained result adopted.',source_sha256=source_identity()['sha256'],generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),current_model_preserved=True,parent_rng_preserved=True,study_unchanged=True,report=dialog.result,file=image.name,sha256=hashlib.sha256(image.read_bytes()).hexdigest())
        text=json.dumps(record,indent=2,allow_nan=False)+'\n';validate_public_text(text,'Resource probe evidence');manifest.write_text(text);print('Captured desktop resource probe; model, inputs and parent RNG preserved.')
    finally:
        if dialog is not None:
            if dialog.active:dialog.stop(force=True);wait(lambda:not dialog.active)
            dialog.shutdown_review()
        window.undo_stack.setClean();window.close();app.clipboard().clear();app.processEvents();os.chdir(previous)


if __name__=='__main__':main()
