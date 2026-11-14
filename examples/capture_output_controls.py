"""Capture an actual disabled output inspector without personal desktop state."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit,QCheckBox
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    path=root/'assets/screenshots/manual-output-disabled.png';manifest=path.with_suffix('.json')
    if not args.overwrite and (path.exists() or manifest.exists()):parser.error('Capture exists; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False)
    try:
        w.resize(1560,1000);w.show();w.load_heat_template()
        node=next(n for n in w.scene.nodes() if n.kind=='solution')
        before=w.scene.compile_config();wires=len(w.scene.wires)
        assert w.set_output_enabled(node,False)
        assert w.scene.compile_config()==before and len(w.scene.wires)==wires
        w.scene.clearSelection();node.setSelected(True);w.selection_changed();w.fit_model()
        for _ in range(5):app.processEvents()
        assert node.result_status=='Disabled'
        assert not w.property_scroll.findChild(QCheckBox,'output_enabled').isChecked()
        for cls in (QLabel,QLineEdit,QPlainTextEdit):
            for child in w.findChildren(cls):
                validate_public_text(child.toPlainText() if isinstance(child,QPlainTextEdit) else child.text(),'output capture')
        assert w.grab().save(str(path))
        manifest.write_text(json.dumps(dict(generator='examples.capture_output_controls',scope='Actual output inspector and Disabled badge; unchanged physics/wiring, no training.',
            file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),model_version=33),indent=2)+'\n')
        print('Captured disabled output controls.')
    finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
