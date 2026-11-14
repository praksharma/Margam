"""Draft parameter/design-point table; previews never change or execute the model."""
import json
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,
    QComboBox,QLineEdit,QTableWidget,QTableWidgetItem,QHeaderView,QAbstractItemView,QPlainTextEdit)
from ..studies import catalogue,preview,MAX_AXES,FIELDS
from .model_document import capture_model


class DesignPoints(QDialog):
    def __init__(self,window):
        super().__init__(window);self.owner=window;self.setWindowTitle('Parameter study preview');self.resize(1060,760)
        self.connected=False;self.result=None
        layout=QVBoxLayout(self)
        note=QLabel('Choose controls on existing blocks and preview their Cartesian combinations. This draft stays in this window only; saving/export and study execution are not available yet. The model and its results are unchanged.')
        note.setWordWrap(True);layout.addWidget(note)
        self.axes=QTableWidget(0,3);self.axes.setHorizontalHeaderLabels(['Block · parameter','Mode','Values (JSON)'])
        self.axes.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch)
        self.axes.setColumnWidth(1,120);self.axes.setColumnWidth(2,390);layout.addWidget(self.axes,2)
        row=QHBoxLayout();layout.addLayout(row)
        self.add=QPushButton('Add parameter');self.add.clicked.connect(self.add_axis);row.addWidget(self.add)
        remove=QPushButton('Remove selected');remove.clicked.connect(self.remove_axis);row.addWidget(remove)
        clear=QPushButton('Clear draft');clear.clicked.connect(lambda:(self.axes.setRowCount(0),self.invalidate()));row.addWidget(clear)
        help_button=QPushButton('Parameter help…');help_button.clicked.connect(self.parameter_help);row.addWidget(help_button)
        row.addStretch();button=QPushButton('Preview design points');button.clicked.connect(self.refresh);row.addWidget(button)
        self.status=QLabel('Add a parameter. Fixed uses one JSON value; Candidates uses a list; Range uses [start, stop, step].')
        self.status.setTextFormat(Qt.TextFormat.PlainText);self.status.setWordWrap(True);layout.addWidget(self.status)
        self.table=QTableWidget(0,5);self.table.setHorizontalHeaderLabels(['Design point','Choices','Preflight','Updates / attempts','Parameters'])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers);self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeMode.Stretch);self.table.setColumnWidth(2,205);self.table.setColumnWidth(3,145);self.table.setColumnWidth(4,105)
        self.table.currentCellChanged.connect(self.details_for);layout.addWidget(self.table,3)
        self.details=QPlainTextEdit();self.details.setReadOnly(True);layout.addWidget(self.details,1)
        note=QLabel('Budget totals include structurally/numerically valid trials only. Parameter counts allocate no network. Counts are not runtime or peak-memory predictions; datasets, outputs and full execution still need validation. No trials run here.')
        note.setWordWrap(True);layout.addWidget(note)
        close=QPushButton('Close');close.clicked.connect(self.close);layout.addWidget(close)

    def watch(self):
        if not self.connected:
            if self.result is not None:self.invalidate('Preview again against the current graph.')
            self.owner.scene.changed.connect(self.model_changed);self.connected=True
    def closeEvent(self,event):
        if self.connected:self.owner.scene.changed.disconnect(self.model_changed);self.connected=False
        super().closeEvent(event)
    def model_changed(self,*args):self.invalidate('Canvas/model changed. Preview again to use the current graph.')
    def invalidate(self,message='Draft changed. Preview again.'):
        self.result=None;self.table.setRowCount(0);self.details.clear();self.status.setText(message);self.add.setEnabled(self.axes.rowCount()<MAX_AXES)
    def available(self):return catalogue(capture_model(self.owner))
    def fill_options(self,combo,selected=None):
        combo.blockSignals(True);combo.clear()
        for item in self.available():combo.addItem(f"{item['title']} [{item['node_id'][:8]}] · {item['label']}",json.dumps([item['node_id'],item['parameter']]))
        if selected is not None:
            index=combo.findData(selected)
            if index<0:combo.addItem('Missing block/parameter — choose again',selected);index=combo.count()-1
            combo.setCurrentIndex(index)
        combo.blockSignals(False)
    def add_axis(self):
        if self.axes.rowCount()>=MAX_AXES:return
        row=self.axes.rowCount();self.axes.insertRow(row);combo=QComboBox();self.fill_options(combo)
        selected={self.axes.cellWidget(i,0).currentData() for i in range(row)}
        for i in range(combo.count()):
            if combo.itemData(i) not in selected:combo.setCurrentIndex(i);break
        mode=QComboBox()
        for label,key in [('Fixed','fixed'),('Candidates','candidates'),('Range','range')]:mode.addItem(label,key)
        text=QLineEdit();text.setMaxLength(4096);text.setPlaceholderText('e.g. [1000,2000] in Candidates mode')
        self.axes.setCellWidget(row,0,combo);self.axes.setCellWidget(row,1,mode);self.axes.setCellWidget(row,2,text)
        def use_current():
            item=next((item for item in self.available() if json.dumps([item['node_id'],item['parameter']])==combo.currentData()),None)
            if item is not None:text.setText(json.dumps(item['value']));mode.setCurrentIndex(0)
            self.invalidate()
        combo.currentIndexChanged.connect(use_current);mode.currentIndexChanged.connect(lambda:self.invalidate());text.textChanged.connect(lambda:self.invalidate())
        use_current();self.axes.setCurrentCell(row,0);self.invalidate()
    def remove_axis(self):
        row=self.axes.currentRow()
        if row>=0:self.axes.removeRow(row);self.invalidate()
    def parameter_help(self):
        row=self.axes.currentRow()
        if row<0:return
        data=self.axes.cellWidget(row,0).currentData()
        if data is None:return
        key=json.loads(data)
        from .setting_help import HELP,show_information
        show_information(self,'Study parameter',HELP.get(key[1],'Choose a supported block parameter.')+'\n\nFixed: one JSON value. Candidates: a list of values. Range: [start, stop, step], inclusive when stop is reached; descending ranges are supported. Counts/seeds require integers. Layers use lists such as [[16,16],[32,32]] in Candidates mode. Limits: 8 parameters, 64 values each, 512 total design points. No training or model edit occurs.')
    def refresh(self):
        if self.owner.is_running() or self.owner.scene.locked or self.owner.project_workspace.busy:
            self.invalidate('A job/editor is active. Preview after it finishes.');return False
        axes=[]
        for row in range(self.axes.rowCount()):
            combo=self.axes.cellWidget(row,0);self.fill_options(combo,combo.currentData());key=combo.currentData()
            if key is None:self.invalidate('No supported block parameter is available.');return False
            key=json.loads(key)
            axes.append(dict(node_id=key[0],parameter=key[1],mode=self.axes.cellWidget(row,1).currentData(),text=self.axes.cellWidget(row,2).text()))
        try:result=preview(capture_model(self.owner),axes,scene=self.owner.scene)
        except (ValueError,TypeError,KeyError,OSError) as error:self.invalidate(str(error));return False
        self.result=result;self.table.setRowCount(len(result['trials']))
        for index,trial in enumerate(result['trials']):
            choices='; '.join(item['parameter']+'='+json.dumps(item['value']) for item in trial['choices'])
            for column,value in enumerate((trial['design_point'],choices,trial['status'],trial['updates'],trial['parameters'])):
                item=QTableWidgetItem('—' if value is None else str(value));item.setToolTip(trial['reason'] or choices);self.table.setItem(index,column,item)
        self.status.setText(f"{result['total']} design points · {result['valid']} pass numerical preflight · {result['invalid']} invalid. Total requested updates/attempts: {result['total_updates']:,}; largest network: {result['maximum_parameters']:,} parameters. Execution/data validation still required.")
        if result['trials']:self.table.setCurrentCell(0,0)
        return True
    def details_for(self,*args):
        row=self.table.currentRow()
        if self.result is None or not 0<=row<len(self.result['trials']):self.details.clear();return
        trial=self.result['trials'][row]
        lines=[trial['design_point']+' · '+trial['status']]
        lines += [FIELDS[item['parameter']]+' ['+item['node_id'][:8]+']: '+json.dumps(item['value']) for item in trial['choices']]
        if trial['reason']:lines.append('Reason: '+trial['reason'])
        self.details.setPlainText('\n'.join(lines))
