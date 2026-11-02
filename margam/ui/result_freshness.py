"""Ephemeral output availability badges; never certify external file freshness."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel
from .output_views import OUTPUT_KINDS
from .block_catalogue import describe_block


def refresh(window,phase=None):
    outputs=(window.result or {}).get('outputs',{})
    for node in window.scene.nodes():
        if node.kind not in OUTPUT_KINDS:continue
        previous=getattr(node,'result_status','Not run')
        connected=any(w.target.parentItem() is node for w in window.scene.wires)
        if connected and previous=='Not connected':previous='Not run'
        has_data=bool(window.monitor_records.get(node.identifier)) if node.kind=='monitor' else node.identifier in outputs
        if phase=='changed':
            status='Not run' if window._resetting else 'Settings changed' if previous not in ('Not run','Not connected') else previous
        elif phase in ('Running','Evaluating','Failed'):
            status=phase if node.identifier in getattr(window,'active_output_ids',set()) else previous
        elif phase is None and (previous=='Failed' or previous in ('Running','Evaluating') and window.is_running()):status=previous
        elif phase=='complete' or has_data:
            status=('Stopped / partial' if (window.result or {}).get('cancelled') else 'Current') if has_data else 'Not available' if node.identifier in getattr(window,'active_output_ids',set()) else 'Not run'
        else:status=previous
        if not connected:status='Not connected'
        node.result_status=status;node.update()
        node.setToolTip(node.title+'\n'+node.detail+'\nResult: '+status+'\n\n'+describe_block(node.kind)+'\n\nFreshness refers to the current in-memory model; use Project data overview to check changed external files.')
        entry=window.output_views.entries.get(node.identifier)
        if entry:
            view=entry[1]
            label=getattr(view,'freshness_label',None)
            if label is None:
                label=QLabel();label.setObjectName('result_freshness');label.setTextFormat(Qt.TextFormat.PlainText)
                view.layout().insertWidget(0,label);view.freshness_label=label
            label.setText('Result: '+status)
