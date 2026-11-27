"""Capture a state-isolated manual screenshot of the FBPINN desktop workflow."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QPlainTextEdit
    from margam.ui.app import Window
    from margam.ui.model_document import CURRENT_VERSION, capture_model
    from margam.ui.theme import configure_theme
    from examples.documentation import validate_public_text

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    image = Path('assets/screenshots/manual-fbpinn.png')
    manifest = Path('assets/screenshots/manual-fbpinn.json')
    if not args.overwrite and any(p.exists() for p in (image, manifest)):
        parser.error('Capture exists; use --overwrite.')

    app = QApplication([])
    configure_theme(app, 'light')
    w = Window(state_directory=False)
    try:
        w.resize(1400, 900)
        w.set_application_device('cpu')
        assert w.load_example('fbpinn_multiscale_2d')
        trainer = next(n for n in w.scene.nodes() if n.kind == 'fbpinn_method')
        w.scene.clearSelection()
        trainer.setSelected(True)
        w.show_block_properties(trainer)
        w.fit_model()
        w.show()
        for _ in range(6):
            app.processEvents()
        bar = w.property_scroll.verticalScrollBar()
        bar.setValue(min(bar.maximum(), 360))
        for _ in range(4):
            app.processEvents()
        for cls in (QLabel, QLineEdit, QPlainTextEdit):
            for child in w.findChildren(cls):
                if child.isVisible():
                    text = child.toPlainText() if hasattr(child, 'toPlainText') else child.text()
                    validate_public_text(text, 'FBPINN manual capture')
        assert w.grab().save(str(image))
        doc = capture_model(w)
        assert doc['version'] == CURRENT_VERSION
        payload = dict(
            generator='examples.capture_fbpinn',
            example='fbpinn_multiscale_2d',
            model_version=CURRENT_VERSION,
            scope='FBPINN multiscale 2D graph and FBPINN trainer inspector (no training run).',
            files={image.name: dict(bytes=image.stat().st_size, sha256=hashlib.sha256(image.read_bytes()).hexdigest())},
        )
        manifest.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
        print(f'Captured {image} and {manifest}.')
    finally:
        w.undo_stack.setClean()
        w.close()
        app.clipboard().clear()
        app.processEvents()


if __name__ == '__main__':
    main()
