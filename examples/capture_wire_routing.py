"""Reproduce a crowded heat diagram and capture optional block-avoiding routes."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def arrange_fixture(window):
    """Leave physics/wires intact; put a note across the interior-to-PDE curve."""
    positions=dict(domain=(0,180),interior=(380,0),boundary=(380,400),pde=(950,0),
        bc=(950,400),network=(950,650),method=(1320,230),monitor=(1720,100),
        solution=(1720,250),comparison=(1720,420),reference=(1320,600),csv=(1720,560))
    for node in window.scene.nodes():
        if node.kind in positions:node.setPos(*positions[node.kind])
    note=window.add_node('Note');note.setPos(620,-35)
    window.change_note(note,'A crowded diagram\nThis note obscures the direct connection.\nRouting goes around it; physics stays unchanged.')
    wire=next(w for w in window.scene.wires if w.source.parentItem().kind=='interior' and w.target.parentItem().kind=='pde')
    window.scene.reroute_wires()
    return note,wire


def wire_hits(wire,node):
    # Open paths have implicit fill closure; inspect the painted stroke instead.
    from PySide6.QtGui import QPainterPathStroker
    stroke=QPainterPathStroker();stroke.setWidth(wire.pen().widthF())
    return stroke.createStroke(wire.path()).intersects(node.sceneBoundingRect())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from margam.ui.model_document import capture_model
    from examples.documentation import validate_public_text
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    paths=[Path('assets/screenshots/manual-wire-routing-'+key+'.png') for key in ('before','after')]
    manifest=Path('assets/screenshots/manual-wire-routing.json')
    if not args.overwrite and any(p.exists() for p in [*paths,manifest]):parser.error('Capture exists; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');window=Window(state_directory=False)
    try:
        window.set_application_device('cpu');window.load_heat_template()
        note,wire=arrange_fixture(window);before=capture_model(window);index=window.undo_stack.index()
        window.resize(1600,960);window.show()
        for _ in range(5):app.processEvents()
        window.fit_model()
        files=[]
        assert wire_hits(wire,note)
        for enabled,path in zip((False,True),paths):
            window.route_wires_action.setChecked(enabled)
            for _ in range(4):app.processEvents()
            if enabled:assert wire.route_status=='routed' and not wire_hits(wire,note)
            for cls in (QLabel,QLineEdit,QPlainTextEdit):
                for child in window.findChildren(cls):
                    if child.isVisible():validate_public_text(child.toPlainText() if hasattr(child,'toPlainText') else child.text(),'wire routing capture')
            assert window.grab().save(str(path))
            files.append(dict(file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        assert capture_model(window)==before and window.undo_stack.index()==index
        manifest.write_text(json.dumps(dict(generator='examples.capture_wire_routing',scope='The same heat graph and note before/after a visual preference; no training or accuracy claim.',
            fixture_wire=wire.source.parentItem().kind+' -> '+wire.target.parentItem().kind,
            routed=sum(w.route_status=='routed' for w in window.scene.wires),fallback=sum(w.route_status=='fallback' for w in window.scene.wires),files=files),indent=2)+'\n')
        print('Captured crowded diagram before/after routing without model changes.')
    finally:window.undo_stack.setClean();window.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
