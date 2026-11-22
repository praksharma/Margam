"""Views belong to output-block identities; hiding a dock does not remove its block."""
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from PySide6.QtCore import Qt,QAbstractListModel,QModelIndex
from .plots import CurvePlot, SolutionView
from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout,QLabel,QTableWidget,QTableWidgetItem,QComboBox,QHBoxLayout,QCheckBox,QPushButton,QFileDialog

from ..components.output_settings import OUTPUT_KINDS,output_enabled


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
    PAGE_SIZE = 200

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
        self.table_controls=QWidget();nav=QHBoxLayout(self.table_controls);nav.setContentsMargins(0,0,0,0)
        self.prev_page_button=QPushButton('Previous');self.prev_page_button.setObjectName('result_prev_page');self.prev_page_button.setEnabled(False)
        self.page_label=QLabel('');self.page_label.setObjectName('result_page_label')
        self.next_page_button=QPushButton('Next');self.next_page_button.setObjectName('result_next_page');self.next_page_button.setEnabled(False)
        self.locate_max_error_button=QPushButton('Locate max |error|');self.locate_max_error_button.setObjectName('result_locate_max_error');self.locate_max_error_button.setEnabled(False)
        self.copy_rows_button=QPushButton('Copy rows CSV');self.copy_rows_button.setObjectName('result_copy_rows_csv');self.copy_rows_button.setEnabled(False)
        for widget in (self.prev_page_button,self.page_label,self.next_page_button):nav.addWidget(widget)
        nav.addStretch()
        nav.addWidget(self.locate_max_error_button);nav.addWidget(self.copy_rows_button)
        layout.addWidget(self.table_controls)
        self.caption=QLabel('Connect and train to calculate fields.');self.caption.setWordWrap(True);layout.addWidget(self.caption)
        self.table=QTableWidget();self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);layout.addWidget(self.table)
        self._result_data=None;self._time_column=None;self.page=0;self._filtered_rows=[]
        self.time.currentIndexChanged.connect(self._on_time_changed)
        self.prev_page_button.clicked.connect(lambda:self.set_page(self.page-1))
        self.next_page_button.clicked.connect(lambda:self.set_page(self.page+1))
        self.locate_max_error_button.clicked.connect(self.locate_max_error)
        self.copy_rows_button.clicked.connect(self.copy_rows_csv)

    def _on_time_changed(self):
        self.page=0
        self.refresh_rows()

    def clear(self):
        self._result_data=None;self._time_column=None;self.page=0;self._filtered_rows=[]
        blocked=self.time.blockSignals(True);self.times_model.replace([]);self.time.setCurrentIndex(0);self.time.blockSignals(blocked)
        self.time_controls.hide();self.time.setEnabled(False)
        self.prev_page_button.setEnabled(False);self.next_page_button.setEnabled(False)
        self.locate_max_error_button.setEnabled(False);self.copy_rows_button.setEnabled(False)
        self.page_label.setText('')
        self.table.clear();self.table.setRowCount(0);self.table.setColumnCount(0)
        self.caption.setText('Model changed. Train again or open a saved run.')

    def show_result(self,data):
        names=list(data['config'].coordinate_names)
        column=names.index('t') if data['config'].is_time_dependent and 't' in names else None
        if self._result_data is not data:
            self._result_data=data;self._time_column=column;self.page=0
            times=sorted({float(point[column]) for point in data['points']}) if column is not None else []
            blocked=self.time.blockSignals(True);self.times_model.replace(times);self.time.setCurrentIndex(0);self.time.blockSignals(blocked)
        self.time_controls.setVisible(column is not None);self.time.setEnabled(column is not None)
        self.refresh_rows()

    def set_page(self,page):
        if self._result_data is None:return False
        total_pages=max(1,(len(self._filtered_rows)+self.PAGE_SIZE-1)//self.PAGE_SIZE)
        target=max(0,min(int(page),total_pages-1))
        if target==self.page:return False
        self.page=target
        self.refresh_rows()
        return True

    def refresh_rows(self):
        data=self._result_data
        if data is None:return
        fields=data.get('fields',list(data['config'].field_names));points=data['points'];reference=data.get('reference')
        ids=data.get('point_ids');time=self.time.currentData() if self._time_column is not None else None
        rows=list(range(len(points))) if time is None else [i for i,p in enumerate(points) if p[self._time_column]==time]
        self._filtered_rows=rows
        total_pages=max(1,(len(rows)+self.PAGE_SIZE-1)//self.PAGE_SIZE)
        self.page=max(0,min(self.page,total_pages-1))
        start=self.page*self.PAGE_SIZE
        page_rows=rows[start:start+self.PAGE_SIZE]
        headers=(['point_id'] if ids is not None else [])+[*data['config'].coordinate_names,*fields]+(['ref '+f for f in fields]+['error '+f for f in fields] if reference is not None else [])
        count=len(page_rows);self.table.setColumnCount(len(headers));self.table.setHorizontalHeaderLabels(headers);self.table.setRowCount(count)
        self.table.setVerticalHeaderLabels([str(index+1) for index in page_rows])
        for row_index,index in enumerate(page_rows):
            values=data['values'][index];values=values if isinstance(values,list) else [values]
            row=([ids[index]] if ids is not None else [])+[*points[index],*values]
            if reference is not None:
                refs=reference[index];refs=refs if isinstance(refs,list) else [refs]
                row.extend(refs);row.extend(a-b for a,b in zip(values,refs))
            for column,value in enumerate(row):self.table.setItem(row_index,column,QTableWidgetItem(str(value) if ids is not None and column==0 else f'{value:.7g}'))
        self.table.resizeColumnsToContents()
        self.prev_page_button.setEnabled(self.page>0)
        self.next_page_button.setEnabled(self.page+1<total_pages)
        self.locate_max_error_button.setEnabled(bool(rows) and reference is not None)
        self.copy_rows_button.setEnabled(bool(page_rows))
        self.page_label.setText(f'Page {self.page+1} of {total_pages} (rows {start+1}–{start+count} of {len(rows)})' if rows else 'No rows')
        row_summary=f'first {count} of {len(rows)} rows' if self.page==0 else f'rows {start+1}–{start+count} of {len(rows)}'
        caption=f"Fields {', '.join(fields)} · {row_summary}"
        if self._time_column is not None:
            caption+=' · all stored times' if time is None else f' · t = {time!r} · {len(points)} rows across all times'
        caption+=f" · {data['steps']} updates"
        if data.get('rmse_by_field'):
            caption+=(' · RMSE (all stored rows) ' if self._time_column is not None else ' · RMSE ')+', '.join(name+'='+f'{value:.4g}' for name,value in data['rmse_by_field'].items())
        self.caption.setText(caption)

    def locate_max_error(self):
        data=self._result_data
        if data is None or data.get('reference') is None or not self._filtered_rows:return None
        values=data['values'];reference=data['reference']
        def row_max_err(idx):
            v=values[idx] if isinstance(values[idx],list) else [values[idx]]
            r=reference[idx] if isinstance(reference[idx],list) else [reference[idx]]
            return max(abs(a-b) for a,b in zip(v,r))
        best_pos=max(range(len(self._filtered_rows)),key=lambda pos:row_max_err(self._filtered_rows[pos]))
        target_page=best_pos//self.PAGE_SIZE
        row_in_page=best_pos%self.PAGE_SIZE
        if self.page!=target_page:
            self.page=target_page
            self.refresh_rows()
        self.table.selectRow(row_in_page)
        item=self.table.item(row_in_page,0)
        if item is not None:self.table.scrollToItem(item)
        return self._filtered_rows[best_pos],target_page,row_in_page

    def copy_rows_csv(self):
        if self.table.rowCount()==0 or self.table.columnCount()==0:return False
        selected_rows=sorted({index.row() for index in self.table.selectedIndexes()})
        if not selected_rows:selected_rows=list(range(self.table.rowCount()))
        headers=[self.table.horizontalHeaderItem(c).text() for c in range(self.table.columnCount())]
        lines=[','.join(headers)]
        for r in selected_rows:
            lines.append(','.join(self.table.item(r,c).text() for c in range(self.table.columnCount())))
        QApplication.clipboard().setText('\n'.join(lines)+'\n')
        return True


class TrainingMonitorView(QWidget):
    def __init__(self):
        super().__init__();layout=QVBoxLayout(self)
        controls=QHBoxLayout();controls.addWidget(QLabel('Loss view'))
        self.constraint=QComboBox();self.constraint.setObjectName('monitor_constraint')
        controls.addWidget(self.constraint)
        self.log_scale=QCheckBox('Log scale');self.log_scale.setObjectName('monitor_log_scale');self.log_scale.setChecked(True)
        controls.addWidget(self.log_scale)
        controls.addStretch()
        self.copy_csv_button=QPushButton('Copy CSV');self.copy_csv_button.setObjectName('monitor_copy_csv');self.copy_csv_button.setEnabled(False)
        self.export_csv_button=QPushButton('Export CSV…');self.export_csv_button.setObjectName('monitor_export_csv');self.export_csv_button.setEnabled(False)
        controls.addWidget(self.copy_csv_button);controls.addWidget(self.export_csv_button)
        layout.addLayout(controls)
        self.detail=CurvePlot(logarithmic=True);layout.addWidget(self.detail)
        self.hover_label=QLabel('');self.hover_label.setObjectName('monitor_hover_readout');layout.addWidget(self.hover_label)
        self.records=[];self.summary_series=[];self.summary_caption=''
        self.constraint.currentIndexChanged.connect(self.refresh_detail)
        self.log_scale.toggled.connect(self.set_log_scale)
        self.copy_csv_button.clicked.connect(self.copy_csv)
        self.export_csv_button.clicked.connect(lambda:self.export_csv())
        self.detail.hoverChanged.connect(self.hover_label.setText)

    @property
    def series(self):return self.detail.series

    def set_log_scale(self,enabled):
        self.detail.set_logarithmic(enabled)
        self.refresh_detail()

    def set_series(self,series,caption):
        self.summary_series=series;self.summary_caption=caption;self.refresh_detail()

    def show_constraints(self,records,names):
        selected=self.constraint.currentData();self.records=records
        has_records=bool(records)
        self.copy_csv_button.setEnabled(has_records)
        self.export_csv_button.setEnabled(has_records)
        self.constraint.blockSignals(True);self.constraint.clear();self.constraint.addItem('Summary (all losses)',None)
        for name in names:self.constraint.addItem(name,name)
        index=self.constraint.findData(selected)
        self.constraint.setCurrentIndex(max(0,index));self.constraint.blockSignals(False);self.refresh_detail()

    def refresh_detail(self):
        name=self.constraint.currentData()
        scale_text='log scale' if self.detail.logarithmic else 'linear scale'
        if name is None:
            caption=self.summary_caption if self.detail.logarithmic else self.summary_caption.replace('log scale','linear scale')
            self.detail.set_series(self.summary_series,caption);return
        raw=[(step,row[name]) for step,row in self.records if name in row]
        weighted=[(step,row['weighted:'+name]) for step,row in self.records if 'weighted:'+name in row]
        series=[('Raw',raw,'#7089d5'),('Weighted contribution',weighted,'#d58b35')] if raw else []
        note=name+f' · loss before update · {scale_text}' if raw else 'Train to inspect a named boundary/initial constraint.'
        if raw and len(weighted)<len(raw):note+=' · older records have no weighted contribution'
        self.detail.set_series(series,note)

    def format_csv(self):
        if not self.records:return ''
        keys=[]
        for preferred in ('total','pde'):
            if any(preferred in row for _,row in self.records):keys.append(preferred)
        for _,row in self.records:
            for key in row:
                if key not in keys:keys.append(key)
        lines=['step,'+','.join(keys)]
        for step,row in self.records:
            lines.append(str(step)+','+','.join('' if key not in row else f'{float(row[key]):.10g}' for key in keys))
        return '\n'.join(lines)+'\n'

    def copy_csv(self):
        if not self.records:return False
        QApplication.clipboard().setText(self.format_csv())
        return True

    def export_csv(self,path=None):
        if not self.records:return False
        if path is None:
            path,_=QFileDialog.getSaveFileName(self,'Export training loss history','loss_history.csv','CSV (*.csv)')
        if not path:return False
        target=Path(path)
        if target.suffix.lower()!='.csv':target=target.with_suffix(target.suffix+'.csv' if target.suffix else '.csv')
        target.parent.mkdir(parents=True,exist_ok=True)
        text=self.format_csv()
        with NamedTemporaryFile('w',encoding='utf-8',dir=target.parent,delete=False) as handle:
            temp=Path(handle.name)
            handle.write(text)
        os.replace(temp,target)
        return True


class OutputViews:
    def __init__(self, window):
        self.window = window
        self.entries = {}

    def open(self, node):
        if node.kind not in OUTPUT_KINDS or not output_enabled(node) or node.kind in ('export','paraview'):
            return
        if node.identifier not in self.entries:
            view = TrainingMonitorView() if node.kind == 'monitor' else VectorResultView() if (len(self.window.config.field_names)>1 or self.window.config.is_time_dependent or getattr(self.window.config,'is_3d',False) or bool(getattr(self.window.config,'parametric_axes',())) or any(w.target.parentItem() is node and w.target.name=='points' for w in self.window.scene.wires)) else SolutionView()
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
            vector=(len(data.get('fields',[]))>1 or data['config'].is_time_dependent or getattr(data['config'],'is_3d',False) or bool(getattr(data['config'],'parametric_axes',())) or 'evaluation' in data) if data is not None else (len(self.window.config.field_names)>1 or self.window.config.is_time_dependent or getattr(self.window.config,'is_3d',False) or bool(getattr(self.window.config,'parametric_axes',())) or any(w.target.parentItem() is node and w.target.name=='points' for w in self.window.scene.wires))
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
