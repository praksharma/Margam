"""Modeless read-only collection; explicit verified trial opening uses native archives."""
import json
from pathlib import Path
from PySide6.QtCore import Qt,QThread
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QFileDialog,
    QTableWidget,QTableWidgetItem,QHeaderView,QAbstractItemView,QPlainTextEdit,QComboBox)
from ..study_results import collect,prepare_open,export_csv
from .icons import icon


class Reader(QThread):
    def __init__(self,operation,parent):super().__init__(parent);self.operation=operation;self.value=None;self.error=None;self.abort_requested=False
    def run(self):
        try:self.value=self.operation(lambda:self.abort_requested or self.isInterruptionRequested())
        except Exception as error:self.error=str(error)


class StudyResults(QDialog):
    def __init__(self,owner):
        super().__init__(owner);self.owner=owner;self.collection=None;self.worker=None;self.root=None;self.operation=None
        self.setWindowTitle('Study results');self.resize(1120,780);layout=QVBoxLayout(self)
        note=QLabel('Browse a CLI results folder without changing the open model. Status and reference errors are reported metadata until you explicitly verify and open a completed trial. No training starts here.')
        note.setWordWrap(True);layout.addWidget(note)
        row=QHBoxLayout();layout.addLayout(row)
        self.browse=QPushButton('Open results folder…');self.browse.setIcon(icon('open'));self.browse.clicked.connect(lambda:self.open_folder());row.addWidget(self.browse)
        self.refresh_button=QPushButton('Refresh');self.refresh_button.clicked.connect(self.refresh);row.addWidget(self.refresh_button)
        row.addStretch();row.addWidget(QLabel('Reported reference error:'));self.metric_choice=QComboBox();self.metric_choice.addItem('No reference diagnostics',None);self.metric_choice.currentIndexChanged.connect(self.populate);row.addWidget(self.metric_choice)
        self.heading=QLabel('No results folder open.');self.heading.setTextFormat(Qt.TextFormat.PlainText);self.heading.setWordWrap(True);layout.addWidget(self.heading)
        self.table=QTableWidget(0,6);self.table.setHorizontalHeaderLabels(['Trial','Choices','Reported status','Attempt','Actual steps','Reference relative L2'])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers);self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows);self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeMode.Stretch);self.table.setColumnWidth(2,130);self.table.setColumnWidth(3,65);self.table.setColumnWidth(4,100);self.table.setColumnWidth(5,185)
        self.table.currentCellChanged.connect(self.selection_changed);layout.addWidget(self.table,3)
        self.details=QPlainTextEdit();self.details.setReadOnly(True);layout.addWidget(self.details,2)
        self.status=QLabel('Choose the directory containing summary.json and study.margam-study.');self.status.setTextFormat(Qt.TextFormat.PlainText);self.status.setWordWrap(True);layout.addWidget(self.status)
        row=QHBoxLayout();layout.addLayout(row)
        self.open_button=QPushButton('Verify and open completed trial');self.open_button.setIcon(icon('open'));self.open_button.clicked.connect(self.open_selected);row.addWidget(self.open_button)
        self.export_button=QPushButton('Export table CSV…');self.export_button.setIcon(icon('export'));self.export_button.clicked.connect(lambda:self.export_table());row.addWidget(self.export_button)
        self.cancel=QPushButton('Cancel reading');self.cancel.clicked.connect(self.cancel_reading);row.addWidget(self.cancel);row.addStretch()
        close=QPushButton('Close');close.clicked.connect(self.close);row.addWidget(close);self.buttons()

    def buttons(self):
        busy=self.worker is not None;index=self.table.currentRow()
        self.browse.setEnabled(not busy);self.refresh_button.setEnabled(not busy and self.root is not None);self.cancel.setEnabled(busy)
        self.export_button.setEnabled(not busy and self.collection is not None)
        completed=self.collection is not None and 0<=index<len(self.collection['records']) and self.collection['records'][index]['row']['status']=='completed'
        self.open_button.setEnabled(not busy and completed and not self.owner.is_running() and not self.owner.scene.locked and not self.owner.project_workspace.busy)

    def start_read(self,operation,kind):
        if self.worker is not None:return False
        self.operation=kind;self.worker=Reader(operation,self);self.worker.finished.connect(self.read_finished)
        self.status.setText('Reading study metadata…' if kind=='collect' else 'Verifying published payloads and preparing a private run copy…')
        self.buttons();self.worker.start();return True

    def open_folder(self,path=None):
        if self.worker is not None:return False
        if path is None:path=QFileDialog.getExistingDirectory(self,'Open study results folder')
        if not path:return False
        self.root=Path(path).absolute();self.collection=None;self.table.setRowCount(0);self.details.clear();self.heading.setText(self.root.name)
        return self.refresh()

    def refresh(self):
        if self.root is None or self.worker is not None:return False
        root=self.root
        # Invalidate before reading so a failed refresh cannot open an older selection.
        self.collection=None;self.table.setRowCount(0);self.details.clear()
        return self.start_read(lambda check:collect(root,check=check),'collect')

    def read_finished(self):
        worker=self.sender()
        if worker is None or worker is not self.worker:return
        self.worker=None
        if worker.error:self.status.setText('Study read failed: '+worker.error)
        elif worker.abort_requested or worker.isInterruptionRequested():
            if self.operation=='open' and worker.value is not None:worker.value[0].cleanup()
            self.status.setText('Study reading cancelled.')
        elif self.operation=='collect':
            self.collection=worker.value;report=self.collection['report'];execution=report['effective_execution']
            self.heading.setText(f"{self.root.name} · {report['study_id'][:16]} · {report['state']} · {execution['device']} · "+', '.join(f'{value} {key}' for key,value in sorted(report['counts'].items())))
            identifiers=sorted({identifier for record in self.collection['records'] for identifier in (record['row']['metrics'] or {})})
            titles={node['id']:node['title'] for node in self.collection['study']['model']['nodes']}
            self.metric_choice.blockSignals(True);self.metric_choice.clear()
            for identifier in identifiers:self.metric_choice.addItem(titles.get(identifier,identifier)+' · '+identifier[:8],identifier)
            if not identifiers:self.metric_choice.addItem('No reference diagnostics',None)
            self.metric_choice.blockSignals(False);self.populate();self.status.setText('Reported outcomes loaded. Payload hashes are verified only when opening a completed trial. Refresh explicitly for new progress.')
        else:
            temporary,path=worker.value
            try:
                if self.owner.is_running() or self.owner.scene.locked or self.owner.project_workspace.busy:self.status.setText('Finish the job or close the geometry editor before opening this trial.')
                else:self.status.setText('Verified trial opened; output views remain explicit.' if self.owner.open_saved_run(path) else 'Trial opening cancelled or rejected; current model preserved.')
            finally:temporary.cleanup()
        worker.deleteLater();self.buttons()

    def populate(self):
        if self.collection is None:return
        selected=self.table.currentRow();self.table.setRowCount(0);identifier=self.metric_choice.currentData()
        for index,record in enumerate(self.collection['records']):
            row=record['row'];metric=(row['metrics'] or {}).get(identifier)
            diagnostic='—'
            if metric is not None:diagnostic=f"{metric['value']:.6g}" if metric['value'] is not None else metric['status']
            choices='; '.join(f"{choice['block']} · {choice['parameter']} = {json.dumps(choice['value'])}" for choice in record['choices'])
            values=[f'DP{index+1:04d}',choices,row['status'],str(row['attempt']),str(row['steps']) if row['steps'] is not None else '—',diagnostic]
            self.table.insertRow(index)
            for column,value in enumerate(values):self.table.setItem(index,column,QTableWidgetItem(value))
        if self.table.rowCount():self.table.selectRow(selected if 0<=selected<self.table.rowCount() else 0)
        self.selection_changed();self.buttons()

    def selection_changed(self,*_):
        index=self.table.currentRow()
        if self.collection is None or not 0<=index<len(self.collection['records']):self.details.clear();self.buttons();return
        record=self.collection['records'][index];report=self.collection['report']
        row=record['row'];execution=report['effective_execution'];runtime=report['runtime']
        lines=[f"DP{index+1:04d} · {row['status']} (reported) · attempt {row['attempt']}",
            f"Actual steps: {row['steps'] if row['steps'] is not None else 'not completed'}",'','Parameter choices:']
        lines.extend(f"  {choice['block']} · {choice['parameter']} = {json.dumps(choice['value'])}" for choice in record['choices'])
        if row['reason']:lines.extend(['','Reason: '+row['reason']])
        lines.extend(['','Reference diagnostics:'])
        titles={node['id']:node['title'] for node in self.collection['study']['model']['nodes']}
        for identifier,diagnostic in (row['metrics'] or {}).items():
            value=f"{diagnostic['value']:.8g}" if diagnostic['value'] is not None else diagnostic['status']
            lines.append(f"  {titles.get(identifier,identifier)}: relative L2 {value}; fields {', '.join(map(str,diagnostic['fields']))}; {diagnostic['rows']} rows; {diagnostic['status']}")
        if not row['metrics']:lines.append('  No completed reference diagnostic.')
        if row['history']:
            lines.extend(['','Previous attempts:'])
            lines.extend(f"  Attempt {old['attempt']}: {old['status']} · {old.get('reason') or 'no reason recorded'}" for old in row['history'])
        lines.extend(['',f"Execution: {execution['device']} (requested {execution['requested_device']}); job limit {execution['wall_time_seconds']} s (0 unlimited); explicit exports {execution['exports']}",
            f"Recorded runtime: Python {runtime['python']}; PyTorch {runtime['torch']}; {runtime['system']} {runtime['machine']}",
            f"Trial ID: {row['trial_id']}",f"Recorded source: {report['source']['sha256']}",'',
            'Reported metadata; payload not yet verified. Verify and open a completed trial to check its saved files.',
            'Errors use aligned recorded rows and fields. No interpolation, unit normalization or automatic cross-physics ranking.'])
        self.details.setPlainText('\n'.join(lines)[:200000]);self.buttons()

    def open_selected(self):
        self.buttons()
        if not self.open_button.isEnabled():return False
        collection=self.collection;index=self.table.currentRow()
        return self.start_read(lambda check:prepare_open(collection,index,check=check),'open')

    def export_table(self,path=None):
        if self.worker is not None or self.collection is None:return False
        if path is None:
            path,_=QFileDialog.getSaveFileName(self,'Export reported study snapshot to a new CSV','study-results.csv','CSV (*.csv)',options=QFileDialog.Option.DontConfirmOverwrite)
        if not path:return False
        try:count=export_csv(self.collection,path)
        except (OSError,ValueError,TypeError,KeyError) as error:
            self.status.setText('CSV export failed: '+str(error));return False
        self.status.setText(f'Exported {count} trials from the displayed metadata snapshot. All reference outputs included; payloads unverified. Refresh explicitly for newer progress. Existing files are never overwritten.')
        return True

    def cancel_reading(self):
        if self.worker is not None:self.worker.abort_requested=True;self.worker.requestInterruption();self.status.setText('Cancelling study reading…')

    def shutdown(self):
        if self.worker is not None:
            worker=self.worker;worker.abort_requested=True;worker.requestInterruption();worker.wait()
            if self.operation=='open' and worker.value is not None:worker.value[0].cleanup()
            worker.finished.disconnect(self.read_finished);self.worker=None;worker.deleteLater()
        self.status.setText('Study reading cancelled.');self.buttons()

    def closeEvent(self,event):self.shutdown();super().closeEvent(event)
    def showEvent(self,event):super().showEvent(event);self.buttons()
