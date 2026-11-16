"""Explicit saved-run prediction comparison, independent of the current model."""
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QComboBox,
    QFileDialog,QTableWidget,QTableWidgetItem,QHeaderView,QPlainTextEdit)
from .study_results import Reader
from .icons import icon
from ..run_comparison import read_predictions,compare_predictions,compare_prediction_group


def diagnostic_text(value):
    if value is None:return 'No reference'
    return f"{value['value']:.8g}" if value['value'] is not None else value['status']


def append_run_slot(dialog,changed):
    if dialog.worker is not None or len(dialog.runs)>=8:return False
    index=len(dialog.runs);letter=chr(65+index);dialog.runs.append(None)
    row=QHBoxLayout();label=QLabel(letter+': choose a saved run');label.setTextFormat(Qt.TextFormat.PlainText);row.addWidget(label,1);dialog.labels.append(label)
    button=QPushButton('Choose run '+letter+'…');button.setIcon(dialog.choose_buttons[0].icon());button.clicked.connect(lambda checked=False,i=index:dialog.choose_file(i));row.addWidget(button);dialog.choose_buttons.append(button)
    combo=QComboBox();combo.setMinimumWidth(240);combo.currentIndexChanged.connect(changed);row.addWidget(combo);dialog.choices.append(combo)
    dialog.layout().insertLayout(1+index,row);dialog.file_rows.append(row);changed();return True


def remove_run_slot(dialog,changed):
    if dialog.worker is not None or len(dialog.runs)<=2:return False
    row=dialog.file_rows.pop()
    while row.count():row.takeAt(0).widget().deleteLater()
    dialog.layout().removeItem(row);row.deleteLater()
    dialog.runs.pop();dialog.labels.pop();dialog.choose_buttons.pop();dialog.choices.pop();changed();return True


class RunComparison(QDialog):
    def __init__(self,owner,*,extra_runs=False):
        super().__init__(owner);self.worker=None;self.runs=[None,None];self.pending=None;self.comparison=None;self.file_rows=[]
        self.setWindowTitle('Compare saved predictions');self.resize(1120,820);layout=QVBoxLayout(self)
        note=QLabel('Compare saved predictions on exactly matching evaluation rows, IDs and fields. Use Add run for up to eight selections; every run is compared with A. No interpolation, training, network restoration or changes to the current model. Relative differences use A as denominator; they are not solution accuracy.')
        note.setWordWrap(True);layout.addWidget(note);self.labels=[];self.choices=[];self.choose_buttons=[]
        for index,letter in enumerate(('A','B')):
            row=QHBoxLayout();layout.addLayout(row);label=QLabel(letter+': choose a saved run');label.setTextFormat(Qt.TextFormat.PlainText);row.addWidget(label,1);self.labels.append(label)
            button=QPushButton('Choose run '+letter+'…');button.setIcon(icon('open'));button.clicked.connect(lambda checked=False,i=index:self.choose_file(i));row.addWidget(button);self.choose_buttons.append(button)
            combo=QComboBox();combo.setMinimumWidth(240);combo.currentIndexChanged.connect(self.invalidate);row.addWidget(combo);self.choices.append(combo)
        row=QHBoxLayout();layout.addLayout(row);self.compare_button=QPushButton('Compare selected outputs');self.compare_button.clicked.connect(self.compare);row.addWidget(self.compare_button)
        self.cancel=QPushButton('Cancel reading');self.cancel.clicked.connect(self.cancel_reading);row.addWidget(self.cancel);row.addStretch()
        if extra_runs:
            self.add_run=QPushButton('Add run');self.add_run.clicked.connect(self.add_slot);row.addWidget(self.add_run)
            self.remove_run=QPushButton('Remove last run');self.remove_run.clicked.connect(self.remove_slot);row.addWidget(self.remove_run)
        self.status=QLabel('Choose two native .margam-run.pt archives.');self.status.setTextFormat(Qt.TextFormat.PlainText);self.status.setWordWrap(True);layout.addWidget(self.status)
        self.metrics=QTableWidget(0,6);self.metrics.setHorizontalHeaderLabels(['Field','RMS B−A','Max |B−A|','Relative L2 B−A / A','A vs own reference L2','B vs own reference L2'])
        self.metrics.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);self.metrics.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);layout.addWidget(self.metrics,1)
        self.preview=QTableWidget();self.preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);self.preview.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);layout.addWidget(self.preview,3)
        self.details=QPlainTextEdit();self.details.setReadOnly(True);layout.addWidget(self.details,1)
        close=QPushButton('Close');close.clicked.connect(self.close);layout.addWidget(close);self.buttons()

    def buttons(self):
        busy=self.worker is not None
        for button in self.choose_buttons:button.setEnabled(not busy)
        for combo in self.choices:combo.setEnabled(not busy)
        if hasattr(self,'add_run'):
            self.add_run.setEnabled(not busy and len(self.runs)<8);self.remove_run.setEnabled(not busy and len(self.runs)>2)
        self.compare_button.setEnabled(not busy and all(combo.currentData() is not None for combo in self.choices));self.cancel.setEnabled(busy)

    def add_slot(self):return append_run_slot(self,self.invalidate)
    def remove_slot(self):return remove_run_slot(self,self.invalidate)

    def invalidate(self,*_):
        self.comparison=None;self.metrics.setRowCount(0);self.preview.setRowCount(0);self.preview.setColumnCount(0);self.details.clear();self.status.setText('Choose outputs, then compare. Loaded files are fixed snapshots; choose again to refresh them.');self.buttons()

    def choose_file(self,index,path=None):
        if self.worker is not None or not 0<=index<len(self.runs):return False
        if path is None:path,_=QFileDialog.getOpenFileName(self,'Choose run '+chr(65+index),'','Margam run (*.margam-run.pt)')
        if not path:return False
        self.runs[index]=None;self.choices[index].clear();self.labels[index].setText(chr(65+index)+': '+Path(path).name);self.invalidate()
        self.pending=index;self.worker=Reader(lambda check:read_predictions(path,check=check),self);self.worker.finished.connect(self.read_finished)
        self.status.setText('Reading a restricted CPU snapshot; no network will be constructed…');self.buttons();self.worker.start();return True

    def read_finished(self):
        worker=self.sender()
        if worker is None or worker is not self.worker:return
        self.worker=None
        if worker.error:self.status.setText(('Comparison unavailable: ' if self.pending=='compare' else 'Run could not be read: ')+worker.error)
        elif worker.abort_requested or worker.isInterruptionRequested():self.status.setText('Reading/comparison cancelled.')
        elif self.pending=='compare':self.display_comparison(worker.value)
        else:
            run=worker.value;index=self.pending;self.runs[index]=run;combo=self.choices[index];combo.blockSignals(True)
            for identifier in run['outputs']:combo.addItem(run['titles'].get(identifier,identifier)+' · '+identifier[:8],identifier)
            combo.blockSignals(False);self.labels[index].setText(chr(65+index)+f": {run['name']} · {run['steps']} steps · {run['device']}"+(' · partial' if run['cancelled'] else ''))
            self.status.setText('Snapshot loaded. Choose an explicit output and compare.' if run['outputs'] else 'This run has no saved prediction outputs. Choose another archive.')
        worker.value=None;worker.operation=None;worker.deleteLater();self.buttons()

    def compare(self):
        if self.worker is not None:return False
        self.invalidate()
        if not all(self.runs) or any(combo.currentData() is None for combo in self.choices):return False
        if len(self.runs)>2:
            selections=[(run,combo.currentData()) for run,combo in zip(self.runs,self.choices)]
            self.pending='compare';self.worker=Reader(lambda check:compare_prediction_group(selections,check=check),self);self.worker.finished.connect(self.read_finished)
            self.status.setText('Comparing all saved rows with A in the background…');self.buttons()
            try:self.worker.start()
            except Exception as error:
                self.worker.operation=None;self.worker.deleteLater();self.worker=None;self.status.setText('Comparison could not start: '+(str(error) or type(error).__name__));self.buttons();return False
            return True
        try:value=compare_predictions(self.runs[0],self.choices[0].currentData(),self.runs[1],self.choices[1].currentData())
        except (ValueError,KeyError,TypeError) as error:self.status.setText('Comparison unavailable: '+str(error));return False
        self.display_comparison(value);return True

    def display_comparison(self,value):
        self.comparison=value;group='comparisons' in value
        headers=['Field','RMS B−A','Max |B−A|','Relative L2 B−A / A','A vs own reference L2','B vs own reference L2']
        if group:headers=['Run','Field','RMS run−A','Max |run−A|','Relative L2 run−A / A','A vs own reference L2','Run vs own reference L2']
        self.metrics.setColumnCount(len(headers));self.metrics.setHorizontalHeaderLabels(headers)
        for index,record in enumerate(value['diagnostics']):
            self.metrics.insertRow(index)
            values=([record['run']] if group else [])+[record['field'],*[diagnostic_text(record[key]) for key in ('rms_difference','max_difference','relative_difference','reference_a','reference_b')]]
            for column,text in enumerate(values):self.metrics.setItem(index,column,QTableWidgetItem(text))
        self.preview.setColumnCount(len(value['headers']));self.preview.setHorizontalHeaderLabels(value['headers']);self.preview.setRowCount(len(value['preview']))
        for row,items in enumerate(value['preview']):
            for column,item in enumerate(items):self.preview.setItem(row,column,QTableWidgetItem(f'{item:.8g}' if type(item) is float else str(item)))
        self.status.setText(f"Compared all {value['rows']} rows; showing the first {len(value['preview'])}. Identical reference values: {'yes' if value['identical_reference_values'] else 'no / absent'}.")
        lines=[value['scope'],'Changed configuration keys versus A: '+(', '.join(value['configuration_differences']) or 'none')]
        for i,run in enumerate(self.runs):lines.append(chr(65+i)+' · '+run['name']+' · SHA256 '+run['sha256'])
        if group:
            for item in value['comparisons']:lines.append(item['run']+' versus A · identical saved references: '+('yes' if item['identical_reference_values'] else 'no / absent')+' · changed configuration keys: '+(', '.join(item['configuration_differences']) or 'none'))
        lines.append('Matching rows do not establish matching units, PDEs or reference protocols.')
        self.details.setPlainText('\n'.join(lines))

    def cancel_reading(self):
        if self.worker is not None:self.worker.abort_requested=True;self.worker.requestInterruption()
    def shutdown(self):
        if self.worker is not None:
            worker=self.worker;worker.abort_requested=True;worker.requestInterruption();worker.wait();worker.finished.disconnect(self.read_finished)
            worker.value=None;worker.operation=None;worker.deleteLater();self.worker=None
        self.runs=[None]*len(self.choices)
        for index,combo in enumerate(self.choices):combo.clear();self.labels[index].setText(chr(65+index)+': choose a saved run')
        self.invalidate()
    def closeEvent(self,event):self.shutdown();super().closeEvent(event)
