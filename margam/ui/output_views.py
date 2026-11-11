"""Views belong to output-block identities; hiding a dock does not remove its block."""
from PySide6.QtCore import Qt,QAbstractListModel,QModelIndex
from .plots import CurvePlot, SolutionView
from PySide6.QtWidgets import QWidget,QVBoxLayout,QLabel,QTableWidget,QTableWidgetItem,QComboBox,QHBoxLayout

OUTPUT_KINDS = {'monitor', 'solution', 'comparison', 'export', 'paraview'}


class StoredTimesModel(QAbstractListModel):
    """Lazy combo rows: no widget/item per supplied physical-time value."""
    def __init__(self,parent=None):
        super().__init__(parent);self.times=[]

    def replace(self,times):
        self.beginResetModel();self.times=times;self.endResetModel()

    def rowCount(self,parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.times)+1

    def data(self,index,role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0<=index.row()<=len(self.times):return None
        time=self.times[index.row()-1] if index.row() else None
        if role==Qt.ItemDataRole.UserRole:return time
        if role==Qt.ItemDataRole.DisplayRole:
            return 'All stored rows' if time is None else f'Frame {index.row()} · t = {time!r}'
        if role==Qt.ItemDataRole.ToolTipRole:
            return 'Filter exact stored physical times only; no interpolation or inference. Training updates are not physical time.'


class VectorResultView(QWidget):
    def __init__(self):
        super().__init__();layout=QVBoxLayout(self)
        self.time_controls=QWidget();controls=QHBoxLayout(self.time_controls);controls.setContentsMargins(0,0,0,0)
        controls.addWidget(QLabel('Physical time'))
        self.time=QComboBox();self.time.setObjectName('result_physical_time')
        self.times_model=StoredTimesModel(self.time);self.time.setModel(self.times_model)
        self.time.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.time.setMinimumContentsLength(24)
        self.time.setToolTip('Show all rows or one exact stored physical-time frame. Display filter only; exports and error norms still use the full stored data.')
        controls.addWidget(self.time);controls.addStretch();layout.addWidget(self.time_controls);self.time_controls.hide()
        self.caption=QLabel('Connect and train to calculate fields.');self.caption.setWordWrap(True);layout.addWidget(self.caption)
        self.table=QTableWidget();self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);layout.addWidget(self.table)
        self._result_data=None;self._time_column=None
        self.time.currentIndexChanged.connect(self.refresh_rows)

    def clear(self):
        self._result_data=None;self._time_column=None
        blocked=self.time.blockSignals(True);self.times_model.replace([]);self.time.setCurrentIndex(0);self.time.blockSignals(blocked)
        self.time_controls.hide();self.time.setEnabled(False)
        self.table.clear();self.table.setRowCount(0);self.table.setColumnCount(0)
        self.caption.setText('Model changed. Train again or open a saved run.')

    def show_result(self,data):
        names=list(data['config'].coordinate_names)
        column=names.index('t') if data['config'].is_time_dependent and 't' in names else None
        if self._result_data is not data:
            self._result_data=data;self._time_column=column
            times=sorted({float(point[column]) for point in data['points']}) if column is not None else []
            blocked=self.time.blockSignals(True);self.times_model.replace(times);self.time.setCurrentIndex(0);self.time.blockSignals(blocked)
        self.time_controls.setVisible(column is not None);self.time.setEnabled(column is not None)
        self.refresh_rows()

    def refresh_rows(self):
        data=self._result_data
        if data is None:return
        fields=data.get('fields',list(data['config'].field_names));points=data['points'];reference=data.get('reference')
        ids=data.get('point_ids');time=self.time.currentData() if self._time_column is not None else None
        rows=range(len(points)) if time is None else [i for i,p in enumerate(points) if p[self._time_column]==time]
        headers=(['point_id'] if ids is not None else [])+[*data['config'].coordinate_names,*fields]+(['ref '+f for f in fields]+['error '+f for f in fields] if reference is not None else [])
        count=min(200,len(rows));self.table.setColumnCount(len(headers));self.table.setHorizontalHeaderLabels(headers);self.table.setRowCount(count)
        self.table.setVerticalHeaderLabels([str(index+1) for index in rows[:count]])
        for row_index,index in enumerate(rows[:count]):
            values=data['values'][index];values=values if isinstance(values,list) else [values]
            row=([ids[index]] if ids is not None else [])+[*points[index],*values]
            if reference is not None:
                refs=reference[index];refs=refs if isinstance(refs,list) else [refs]
                row.extend(refs);row.extend(a-b for a,b in zip(values,refs))
            for column,value in enumerate(row):self.table.setItem(row_index,column,QTableWidgetItem(str(value) if ids is not None and column==0 else f'{value:.7g}'))
        self.table.resizeColumnsToContents()
        caption=f"Fields {', '.join(fields)} · first {count} of {len(rows)} rows"
        if self._time_column is not None:
            caption+=' · all stored times' if time is None else f' · t = {time!r} · {len(points)} rows across all times'
        caption+=f" · {data['steps']} updates"
        if data.get('rmse_by_field'):
            caption+=(' · RMSE (all stored rows) ' if self._time_column is not None else ' · RMSE ')+', '.join(name+'='+f'{value:.4g}' for name,value in data['rmse_by_field'].items())
        self.caption.setText(caption)


class TrainingMonitorView(QWidget):
    def __init__(self):
        super().__init__();layout=QVBoxLayout(self)
        controls=QHBoxLayout();controls.addWidget(QLabel('Loss view'))
        self.constraint=QComboBox();self.constraint.setObjectName('monitor_constraint')
        controls.addWidget(self.constraint);controls.addStretch();layout.addLayout(controls)
        self.detail=CurvePlot(logarithmic=True);layout.addWidget(self.detail)
        self.records=[];self.summary_series=[];self.summary_caption=''
        self.constraint.currentIndexChanged.connect(self.refresh_detail)

    @property
    def series(self):return self.detail.series

    def set_series(self,series,caption):
        self.summary_series=series;self.summary_caption=caption;self.refresh_detail()

    def show_constraints(self,records,names):
        selected=self.constraint.currentData();self.records=records
        self.constraint.blockSignals(True);self.constraint.clear();self.constraint.addItem('Summary (all losses)',None)
        for name in names:self.constraint.addItem(name,name)
        index=self.constraint.findData(selected)
        self.constraint.setCurrentIndex(max(0,index));self.constraint.blockSignals(False);self.refresh_detail()

    def refresh_detail(self):
        name=self.constraint.currentData()
        if name is None:
            self.detail.set_series(self.summary_series,self.summary_caption);return
        raw=[(step,row[name]) for step,row in self.records if name in row]
        weighted=[(step,row['weighted:'+name]) for step,row in self.records if 'weighted:'+name in row]
        series=[('Raw',raw,'#7089d5'),('Weighted contribution',weighted,'#d58b35')] if raw else []
        note=name+' · loss before update · log scale' if raw else 'Train to inspect a named boundary/initial constraint.'
        if raw and len(weighted)<len(raw):note+=' · older records have no weighted contribution'
        self.detail.set_series(series,note)


class OutputViews:
    def __init__(self, window):
        self.window = window
        self.entries = {}

    def open(self, node):
        if node.kind not in OUTPUT_KINDS or node.kind in ('export','paraview'):
            return
        if node.identifier not in self.entries:
            view = TrainingMonitorView() if node.kind == 'monitor' else VectorResultView() if (len(self.window.config.field_names)>1 or self.window.config.is_time_dependent or any(w.target.parentItem() is node and w.target.name=='points' for w in self.window.scene.wires)) else SolutionView()
            dock = self.window.dock(node.title, view, Qt.DockWidgetArea.BottomDockWidgetArea)
            dock.setObjectName('output-' + node.identifier)
            for other,_ in self.entries.values():
                if not other.isFloating():
                    hidden=other.isHidden()
                    self.window.tabifyDockWidget(other, dock)
                    if hidden:other.hide()
                    break
            self.entries[node.identifier] = (dock, view)
            self.window.view_menu.addAction(dock.toggleViewAction())
        self.refresh(node)
        dock, _ = self.entries[node.identifier]
        dock.setWindowTitle(node.title)
        self.window.refresh_result_status()
        dock.show()
        dock.raise_()
        return dock

    def refresh(self, node):
        entry = self.entries.get(node.identifier)
        if entry is None:
            return
        _, view = entry
        if node.kind == 'monitor':
            records = self.window.monitor_records.get(node.identifier, [])
            sides = tuple(name for name in self.window.config.constraint_names if name not in ('pressure_gauge','initial'))
            series = [
                ('Total', [(step, row['total']) for step,row in records], '#2b9d91'),
                ('PDE (raw)', [(step, row['pde']) for step,row in records], '#d58b35'),
                ('Observations (raw)' if self.window.config.method=='inverse_burgers' else 'Mean BC (raw)', [(step,sum(row[s] for s in sides)/len(sides)) for step,row in records], '#7089d5'),
            ] if records else []
            if records and 'initial' in self.window.config.constraint_names:series.append(('Initial (raw)',[(step,row['initial']) for step,row in records],'#ad75ce'))
            if records and self.window.config.method=='rba':series.append(('PDE attention-weighted (raw)',[(step,row['pde_weighted']) for step,row in records],'#cc7840'))
            if self.window.config.method=='gpinn':
                from ..components.gpinn_config import gpinn_terms
                for term in gpinn_terms(self.window.config):series.append((term+' (raw)',[(step,row[term]) for step,row in records],{'x':'#ad75ce','y':'#cc7840','t':'#417ec1'}[term[-1]]))
            if records and 'pressure_gauge' in self.window.config.constraint_names:series.append(('Pressure gauge (raw)',[(step,row['pressure_gauge']) for step,row in records],'#ad75ce'))
            view.show_constraints(records,self.window.config.constraint_names)
            view.set_series(series, 'Requested training losses · log scale · retained updates only' if records else 'Connect this monitor and train to record losses.')
        else:
            data = (self.window.result or {}).get('outputs', {}).get(node.identifier)
            vector=(len(data.get('fields',[]))>1 or data['config'].is_time_dependent or 'evaluation' in data) if data is not None else (len(self.window.config.field_names)>1 or self.window.config.is_time_dependent or any(w.target.parentItem() is node and w.target.name=='points' for w in self.window.scene.wires))
            if vector!=isinstance(view,VectorResultView):
                dock,old=entry;hidden=dock.isHidden();view=VectorResultView() if vector else SolutionView()
                dock.setWidget(view);old.deleteLater();self.entries[node.identifier]=(dock,view)
                if hidden:dock.hide()
            if data is None:
                view.clear()
                if hasattr(view,'curve'):view.curve.caption='Connect this output, then train to calculate its data.'
            else:
                view.show_result(data)
        self.window.refresh_result_status()

    def clear_data(self):
        for _,view in self.entries.values():
            if isinstance(view, (CurvePlot,TrainingMonitorView)):
                view.set_series([], 'Model changed. Run training again.')
                if isinstance(view,TrainingMonitorView):view.show_constraints([],self.window.config.constraint_names)
            else:
                view.clear()

    def sync(self, nodes):
        identifiers = {node.identifier for node in nodes}
        for identifier in list(self.entries):
            if identifier not in identifiers:
                dock,_ = self.entries.pop(identifier)
                dock.close()
                self.window.removeDockWidget(dock)
                dock.deleteLater()
