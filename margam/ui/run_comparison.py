"""Explicit saved-run prediction comparison, independent of the current model."""
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QComboBox,
    QFileDialog,QTableWidget,QTableWidgetItem,QHeaderView,QPlainTextEdit)
from .study_results import Reader
from .icons import icon
from ..run_comparison import read_predictions,compare_predictions


def diagnostic_text(value):
    if value is None:return 'No reference'
    return f"{value['value']:.8g}" if value['value'] is not None else value['status']


class RunComparison(QDialog):
    def __init__(self,owner):
        super().__init__(owner);self.worker=None;self.runs=[None,None];self.pending=None;self.comparison=None
        self.setWindowTitle('Compare saved predictions');self.resize(1120,820);layout=QVBoxLayout(self)
        note=QLabel('Compare saved predictions on exactly matching evaluation rows, IDs and fields. No interpolation, training, network restoration or changes to the current model. Relative differences use A as denominator; they are not solution accuracy.')
        note.setWordWrap(True);layout.addWidget(note);self.labels=[];self.choices=[];self.choose_buttons=[]
        for index,letter in enumerate(('A','B')):
            row=QHBoxLayout();layout.addLayout(row);label=QLabel(letter+': choose a saved run');label.setTextFormat(Qt.TextFormat.PlainText);row.addWidget(label,1);self.labels.append(label)
            button=QPushButton('Choose run '+letter+'…');button.setIcon(icon('open'));button.clicked.connect(lambda checked=False,i=index:self.choose_file(i));row.addWidget(button);self.choose_buttons.append(button)
            combo=QComboBox();combo.setMinimumWidth(240);combo.currentIndexChanged.connect(self.invalidate);row.addWidget(combo);self.choices.append(combo)
        row=QHBoxLayout();layout.addLayout(row);self.compare_button=QPushButton('Compare selected outputs');self.compare_button.clicked.connect(self.compare);row.addWidget(self.compare_button)
        self.cancel=QPushButton('Cancel reading');self.cancel.clicked.connect(self.cancel_reading);row.addWidget(self.cancel);row.addStretch()
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
        self.compare_button.setEnabled(not busy and all(combo.currentData() is not None for combo in self.choices));self.cancel.setEnabled(busy)

    def invalidate(self,*_):
        self.comparison=None;self.metrics.setRowCount(0);self.preview.setRowCount(0);self.preview.setColumnCount(0);self.details.clear();self.status.setText('Choose outputs, then compare. Loaded files are fixed snapshots; choose again to refresh them.');self.buttons()

    def choose_file(self,index,path=None):
        if self.worker is not None or index not in (0,1):return False
        if path is None:path,_=QFileDialog.getOpenFileName(self,'Choose run '+('A' if index==0 else 'B'),'','Margam run (*.margam-run.pt)')
        if not path:return False
        self.runs[index]=None;self.choices[index].clear();self.labels[index].setText(('A' if index==0 else 'B')+': '+Path(path).name);self.invalidate()
        self.pending=index;self.worker=Reader(lambda check:read_predictions(path,check=check),self);self.worker.finished.connect(self.read_finished)
        self.status.setText('Reading a restricted CPU snapshot; no network will be constructed…');self.buttons();self.worker.start();return True

    def read_finished(self):
        worker=self.sender()
        if worker is None or worker is not self.worker:return
        self.worker=None
        if worker.error:self.status.setText('Run could not be read: '+worker.error)
        elif worker.abort_requested:self.status.setText('Run reading cancelled.')
        else:
            run=worker.value;index=self.pending;self.runs[index]=run;combo=self.choices[index];combo.blockSignals(True)
            for identifier in run['outputs']:combo.addItem(run['titles'].get(identifier,identifier)+' · '+identifier[:8],identifier)
            combo.blockSignals(False);self.labels[index].setText(('A' if index==0 else 'B')+f": {run['name']} · {run['steps']} steps · {run['device']}"+(' · partial' if run['cancelled'] else ''))
            self.status.setText('Snapshot loaded. Choose an explicit output and compare.' if run['outputs'] else 'This run has no saved prediction outputs. Choose another archive.')
        worker.value=None;worker.deleteLater();self.buttons()

    def compare(self):
        if self.worker is not None:return False
        self.invalidate()
        if not all(self.runs) or any(combo.currentData() is None for combo in self.choices):return False
        try:value=compare_predictions(self.runs[0],self.choices[0].currentData(),self.runs[1],self.choices[1].currentData())
        except (ValueError,KeyError,TypeError) as error:self.status.setText('Comparison unavailable: '+str(error));return False
        self.comparison=value
        for index,record in enumerate(value['diagnostics']):
            self.metrics.insertRow(index)
            values=[record['field'],*[diagnostic_text(record[key]) for key in ('rms_difference','max_difference','relative_difference','reference_a','reference_b')]]
            for column,text in enumerate(values):self.metrics.setItem(index,column,QTableWidgetItem(text))
        self.preview.setColumnCount(len(value['headers']));self.preview.setHorizontalHeaderLabels(value['headers']);self.preview.setRowCount(len(value['preview']))
        for row,items in enumerate(value['preview']):
            for column,item in enumerate(items):self.preview.setItem(row,column,QTableWidgetItem(f'{item:.8g}' if type(item) is float else str(item)))
        self.status.setText(f"Compared all {value['rows']} rows; showing the first {len(value['preview'])}. Identical reference values: {'yes' if value['identical_reference_values'] else 'no / absent'}.")
        self.details.setPlainText(value['scope']+'\nChanged configuration keys: '+(', '.join(value['configuration_differences']) or 'none')+'\nA snapshot SHA256: '+self.runs[0]['sha256']+'\nB snapshot SHA256: '+self.runs[1]['sha256']+'\nMatching coordinates/fields do not establish matching units, PDEs or reference protocols. Inspect the runs before drawing scientific conclusions.')
        return True

    def cancel_reading(self):
        if self.worker is not None:self.worker.abort_requested=True;self.worker.requestInterruption()
    def shutdown(self):
        if self.worker is not None:
            worker=self.worker;worker.abort_requested=True;worker.requestInterruption();worker.wait();worker.finished.disconnect(self.read_finished)
            worker.value=None;worker.deleteLater();self.worker=None
        self.buttons()
    def closeEvent(self,event):
        self.shutdown();self.runs=[None,None]
        for index,combo in enumerate(self.choices):combo.clear();self.labels[index].setText(('A' if index==0 else 'B')+': choose a saved run')
        self.invalidate();super().closeEvent(event)
