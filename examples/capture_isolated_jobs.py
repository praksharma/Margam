"""Capture the real Run-menu force-stop control during an isolated job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QApplication
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    path=Path('assets/screenshots/manual-isolated-stop.png');manifest=path.with_suffix('.json')
    if not args.overwrite and any(p.exists() for p in (path,manifest)):parser.error('Capture exists; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False);w.resize(1400,900);w.show()
    try:
        w.set_application_device('cpu');w.change('layers',(4,));w.change('points',8);w.change('iterations',100000)
        assert w.start_training();deadline=time.monotonic()+15
        while w.worker.child_pid is None and time.monotonic()<deadline:app.processEvents();time.sleep(.002)
        assert w.worker.child_pid is not None and w.force_stop_action.isEnabled()
        for action in w.run_menu.actions():validate_public_text(action.text(),'isolated worker menu')
        w.run_menu.popup(w.mapToGlobal(QPoint(300,50)));w.run_menu.setActiveAction(w.force_stop_action)
        for _ in range(4):app.processEvents()
        assert w.run_menu.grab().save(str(path));w.run_menu.close();w.force_stop_training()
        deadline=time.monotonic()+15
        while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.002)
        assert w.worker is None and w.result is None
        manifest.write_text(json.dumps(dict(generator='examples.capture_isolated_jobs',scope='Actual Run menu during a small isolated CPU job, explicitly force-stopped/discarded after capture; no real memory exhaustion or accuracy claim.',
            separate_process=True,discarded_job=True,file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+'\n')
        print('Captured explicit force-stop control and verified discarded-job cleanup.')
    finally:
        if w.worker is not None:
            w.force_stop_training()
            while w.worker is not None:app.processEvents();time.sleep(.002)
        w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
