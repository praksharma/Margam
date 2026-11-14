"""Capture the real local job-time policy dialog without training or user state."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel
    from margam.ui.app import Window
    from margam.ui.job_limits import JobLimitDialog
    from margam.ui.model_document import capture_model
    from margam.ui.theme import configure_theme
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    path=root/'assets/screenshots/manual-job-time-limit.png';manifest=path.with_suffix('.json')
    if not args.overwrite and (path.exists() or manifest.exists()):parser.error('Capture exists; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');window=Window(state_directory=False);dialog=None
    try:
        before=capture_model(window);index=window.undo_stack.index()
        assert window.set_job_time_limit(300)
        dialog=JobLimitDialog(window.job_time_limit_seconds,window);dialog.show()
        for _ in range(5):app.processEvents()
        assert capture_model(window)==before and window.undo_stack.index()==index
        for child in dialog.findChildren(QLabel):validate_public_text(child.text(),'time-limit dialog')
        assert dialog.grab().save(str(path))
        manifest.write_text(json.dumps(dict(generator='examples.capture_job_limits',scope='Actual local job-time dialog at 300 seconds. State-isolated; no numerical job or model mutation.',
            file=path.name,seconds=dialog.seconds.value(),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+'\n')
        print('Captured job time-limit dialog.')
    finally:
        if dialog is not None:dialog.close()
        window.undo_stack.setClean();window.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
