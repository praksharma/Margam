"""Capture the actual MCP panel while stopped, without credentials or personal state."""
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
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    path=Path('assets/screenshots/manual-mcp.png');manifest=path.with_suffix('.json')
    if not args.overwrite and (path.exists() or manifest.exists()):parser.error('Capture exists; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False)
    try:
        panel=w.show_automation_server();panel.resize(700,780)
        for _ in range(5):app.processEvents()
        assert w.automation_server.state=='Stopped' and not w.automation_server.token
        for cls in (QLabel,QLineEdit,QPlainTextEdit):
            for child in panel.findChildren(cls):
                if child.isVisible():validate_public_text(child.toPlainText() if hasattr(child,'toPlainText') else child.text(),'MCP capture')
        assert panel.grab().save(str(path))
        manifest.write_text(json.dumps(dict(generator='examples.capture_mcp',scope='Actual native MCP panel, server stopped, no credentials, no training.',file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+'\n')
        print('Captured stopped MCP permissions panel.')
    finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
