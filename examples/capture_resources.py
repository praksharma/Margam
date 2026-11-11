"""Capture advisory resource arithmetic using clearly labelled simulated availability."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtGui import QTextCursor
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit
    from margam.ui.app import Window
    from margam.ui.model_document import capture_model
    from margam.ui.theme import configure_theme
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    path=Path('assets/screenshots/manual-resources.png');manifest=path.with_suffix('.json')
    if not args.overwrite and any(p.exists() for p in (path,manifest)):parser.error('Capture exists; use --overwrite.')
    profile=dict(host_available=32*1024**2,host_total=4*1024**3,gpu_available=None,gpu_total=None,dtype='torch.float32',scalar_bytes=4)
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False)
    try:
        w.set_application_device('cpu');w.load_heat_template();w.change('points',10000);w.change('layers',(128,)*4)
        before=capture_model(w);index=w.undo_stack.index()
        with patch('margam.components.resources.memory_snapshot',return_value=profile),patch('margam.nn.factory.build_network',side_effect=AssertionError('Summary must not build a network')):
            dialog=w.show_run_summary();dialog.resize(1160,1040)
            # Keep the fixture explicit inside the actual captured window too.
            dialog.setWindowTitle('Run summary · simulated 32 MiB available-memory example')
            next(label for label in dialog.findChildren(QLabel) if label.text().startswith('Review the assembled')).setText('Illustration: available host RAM is simulated as 32 MiB to show the advisory warning. No training or capacity probe was run.')
            for _ in range(5):app.processEvents()
            text=dialog.text.toPlainText();assert 'WARNING:' in text
            block=dialog.text.document().findBlock(text.index('Memory and workload')).blockNumber()
            dialog.text.verticalScrollBar().setValue(block)
            for _ in range(3):app.processEvents()
            for cls in (QLabel,QLineEdit,QPlainTextEdit):
                for child in dialog.findChildren(cls):
                    if child.isVisible():validate_public_text(child.toPlainText() if hasattr(child,'toPlainText') else child.text(),'resource capture')
            assert capture_model(w)==before and w.undo_stack.index()==index and w.result is None and dialog.train.isEnabled()
            assert dialog.grab().save(str(path))
        manifest.write_text(json.dumps(dict(generator='examples.capture_resources',scope='Read-only resource summary with simulated host availability to demonstrate advisory warnings. No training or capacity benchmark.',
            simulated_memory=profile,file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+'\n')
        print('Captured advisory resource summary with explicitly simulated availability.')
    finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
