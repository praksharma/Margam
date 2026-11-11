"""Read-only pages of retained adaptive collocation points; never touches a worker."""
import math
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QLabel,QSpinBox,QPushButton,QTableWidget,QTableWidgetItem

PAGE_SIZE=200


def snapshot_page(points,weights,*,initial_count,boundary_count,method,steps,history,page=1):
    """Copy at most 200 stored tensor rows to CPU; no sampling or residual scoring."""
    import torch
    if method not in ('rar','importance'):raise ValueError('Choose a supported adaptive sampler.')
    if not isinstance(points,torch.Tensor) or points.ndim!=2 or not 1<=points.shape[1]<=2:
        raise ValueError('Retained points must be an interval/rectangle coordinate tensor.')
    count=len(points)
    if not 0<=boundary_count<=initial_count<=count or method=='importance' and count!=initial_count:
        raise ValueError('Retained sampler counts are inconsistent.')
    if type(page) is not int or page<1:raise ValueError('Page must be a positive integer.')
    pages=max(1,math.ceil(count/PAGE_SIZE));page=min(page,pages);start=(page-1)*PAGE_SIZE;end=min(start+PAGE_SIZE,count)
    coordinates=points[start:end].detach().cpu()
    if not torch.isfinite(coordinates).all():raise ValueError('Non-finite retained coordinates.')
    if weights is None:corrections=[1.]*(end-start)
    else:
        if not isinstance(weights,torch.Tensor) or weights.shape!=(count,1):raise ValueError('Invalid retained importance weights.')
        selected=weights[start:end].detach().cpu()
        if not torch.isfinite(selected).all() or (selected<=0).any():raise ValueError('Invalid retained importance weights.')
        corrections=selected[:,0].tolist()
    anchor_start=initial_count-boundary_count if method=='rar' else count-boundary_count
    rows=[]
    for index,coordinate,weight in zip(range(start,end),coordinates.tolist(),corrections):
        anchor=anchor_start<=index<anchor_start+boundary_count
        kind='Boundary/initial PDE anchor' if anchor else 'RAR addition' if method=='rar' and index>=initial_count else 'Initial interior' if method=='rar' or weights is None else 'Importance draw'
        rows.append(dict(row=index+1,coordinates=coordinate,kind=kind,correction=weight))
    last={key:history[-1][key] for key in ('step','points','mean_residual','max_residual')} if history else None
    return dict(rows=rows,page=page,pages=pages,count=count,boundary_count=boundary_count,
        initial_count=initial_count,steps=steps,method=method,last_event=last,events=len(history),corrected=weights is not None)


class SamplePointsDialog(QDialog):
    def __init__(self,window,node_id):
        super().__init__(window);self.window=window;self.node_id=node_id;self.snapshot=None
        self.resize(920,640);layout=QVBoxLayout(self)
        note=QLabel('Retained training PDE coordinates only, including explicitly connected boundary/initial anchors. Read after a completed or stopped run, or from a saved run. No live worker access, new sampling, residual scoring or prediction. Evaluation/export points are separate.')
        note.setWordWrap(True);note.setTextFormat(Qt.TextFormat.PlainText);layout.addWidget(note)
        self.status=QLabel();self.status.setWordWrap(True);self.status.setTextFormat(Qt.TextFormat.PlainText);layout.addWidget(self.status)
        controls=QHBoxLayout();layout.addLayout(controls);controls.addWidget(QLabel('Page (200 rows)'))
        self.page=QSpinBox();self.page.setToolTip('Inspect consecutive pages of 200 retained rows without sorting, deduplicating or rescoring.');self.page.setMinimum(1);self.page.setKeyboardTracking(False);self.page.valueChanged.connect(self.refresh);controls.addWidget(self.page)
        self.page_total=QLabel();controls.addWidget(self.page_total);controls.addStretch()
        refresh=QPushButton('Refresh retained points');refresh.clicked.connect(self.refresh);controls.addWidget(refresh)
        self.table=QTableWidget();self.table.verticalHeader().hide();self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);layout.addWidget(self.table)
        note=QLabel('Correction multiplies the squared PDE residual; it is not a sampling probability. Unit weights apply before importance refresh and to fixed boundary anchors. Importance draws can repeat coordinates. RAR additions stay after the initial set; rows are not sorted or deduplicated.')
        note.setWordWrap(True);note.setTextFormat(Qt.TextFormat.PlainText);layout.addWidget(note)
        close=QPushButton('Close');close.clicked.connect(self.close);layout.addWidget(close)

    def owner(self):
        return next((n for n in self.window.scene.nodes() if n.identifier==self.node_id and n.kind in ('rar','importance')),None)

    def clear(self,message):
        self.snapshot=None;self.table.setRowCount(0);self.table.setColumnCount(0);self.page.setEnabled(False);self.page_total.clear();self.status.setText(message)

    def refresh(self):
        node=self.owner()
        if node is None:self.clear('The sampler block was removed.');return
        self.setWindowTitle(node.title+' · Retained PDE points')
        if self.window.is_running() or self.window.scene.locked:
            self.clear('A job/editor is active. Refresh after it finishes; live tensors are not read.');return
        config=getattr(self.window,'_trained_config',None)
        if self.window.result is None or config is None or config.adaptive_sampling.get('method')!=node.kind or not any(w.source.parentItem() is node and w.target.parentItem().kind in ('pde','predefined_pde') and w.target.name=='input' for w in self.window.scene.wires):
            self.clear('Train or reopen a matching run with this sampler connected to the PDE.');return
        model=getattr(getattr(self.window,'last_run',None),'model',None)
        controller=getattr(model,'sampling_controller',None)
        checkpoint=self.window.loaded_checkpoint
        if controller is not None:
            points=model.collocation_point_sample;weights=controller.weights;initial=controller.initial_count;history=controller.history;steps=model.iter
        elif checkpoint is not None and 'sampling' in checkpoint:
            state=checkpoint['sampling'];points=state['points'];weights=state['weights'];initial=state['initial_count'];history=state['history'];steps=checkpoint['iter']
        else:self.clear('No retained adaptive point set is available for this run.');return
        # Reconstruct counts only, never coordinates or boundary samples. This is
        # the existing sample_box complete-layer rule for built-in 1D/2D domains.
        interior=math.isqrt(config.points)**2 if config.sampling=='grid' and len(config.bounds[0])==2 else config.points
        boundary=initial-interior if config.pde_points=='both' else 0
        try:
            data=snapshot_page(points,weights,initial_count=initial,boundary_count=boundary,
                method=node.kind,steps=steps,history=history,page=self.page.value())
        except ValueError as error:self.clear(str(error));return
        self.snapshot=data;blocked=self.page.blockSignals(True);self.page.setRange(1,data['pages']);self.page.setValue(data['page']);self.page.blockSignals(blocked);self.page.setEnabled(True)
        self.page_total.setText('of '+str(data['pages']))
        headers=['Stored row',*config.coordinate_names,'Role','PDE correction']
        self.table.setColumnCount(len(headers));self.table.setHorizontalHeaderLabels(headers);self.table.setRowCount(len(data['rows']))
        for index,row in enumerate(data['rows']):
            values=[str(row['row']),*(repr(v) for v in row['coordinates']),row['kind'],repr(row['correction'])]
            for column,value in enumerate(values):self.table.setItem(index,column,QTableWidgetItem(value))
        self.table.resizeColumnsToContents()
        text=f"Retained at update {steps} · {data['count']} PDE rows ({boundary} fixed boundary/initial anchors) · {data['events']} adaptive event(s)"
        event=data['last_event']
        if event:text+=f"\nLast candidate scoring: update {event['step']} · mean residual {event['mean_residual']:.6g}, max {event['max_residual']:.6g} (historical, not rescored)"
        self.status.setText(text)
