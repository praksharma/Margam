"""Capture the real application Device menu and expanded inspector help."""
import argparse,hashlib,json,os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QApplication,QTextBrowser,QToolButton,QLabel,QLineEdit,QPlainTextEdit,QDialog
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root);output=root/'assets/screenshots'
    paths={name:output/('manual-'+name+'.png') for name in ('device-menu','setting-info','block-info')};manifest=output/'manual-device-help-captures.json'
    if not args.overwrite and any(path.exists() for path in (*paths.values(),manifest)):parser.error('Captures exist; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');window=Window(state_directory=False);window.resize(1480,1000);window.show()
    def save(widget,name):
        for _ in range(4):app.processEvents()
        for cls in (QLabel,QLineEdit,QPlainTextEdit,QTextBrowser):
            for child in widget.findChildren(cls):validate_public_text(child.toPlainText() if isinstance(child,(QPlainTextEdit,QTextBrowser)) else child.text(),name)
        assert widget.grab().save(str(paths[name]))
    try:
        window.device_menu.popup(window.mapToGlobal(QPoint(40,70)));save(window.device_menu,'device-menu');window.device_menu.close()
        trainer=next(node for node in window.scene.nodes() if node.kind=='method');window.show_block_properties(trainer)
        info=window.property_scroll.findChild(QToolButton,'setting_info_adam_learning_rate');assert info;info.click();app.processEvents()
        dialog=next(widget for widget in window.findChildren(QDialog) if widget.objectName()=='information_dialog')
        save(dialog,'setting-info');dialog.close();app.processEvents()
        dialog=window.block_help(trainer);save(dialog,'block-info');dialog.close()
        manifest.write_text(json.dumps(dict(generator='examples.capture_device_help',files={path.name:dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths.values()}),indent=2)+'\n')
    finally:window.undo_stack.setClean();window.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
