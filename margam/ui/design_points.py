"""Study table with read-only previews and explicit portable snapshot save/open."""
import json
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,
    QComboBox,QLineEdit,QTableWidget,QTableWidgetItem,QHeaderView,QAbstractItemView,QPlainTextEdit,QFileDialog,QMessageBox,QCheckBox,QSpinBox)
from ..studies import catalogue,preview,MAX_AXES,FIELDS
from .model_document import capture_model


class DesignPoints(QDialog):
    def __init__(self,window):
        super().__init__(window);self.owner=window;self.setWindowTitle('Parameter study');self.resize(1060,900)
        self.connected=False;self.result=None;self.study_path=None
        layout=QVBoxLayout(self)
        note=QLabel('Choose controls on existing blocks and preview their combinations. Save a portable study with its base model and input snapshots; opening one replaces the current model and this table. Training and headless study execution are not started here.')
        note.setWordWrap(True);layout.addWidget(note)
        files_row=QHBoxLayout();layout.addLayout(files_row)
        button=QPushButton('Open study…');button.clicked.connect(lambda:self.open_study());files_row.addWidget(button)
        button=QPushButton('Save portable study…');button.clicked.connect(lambda:self.save_study());files_row.addWidget(button)
        self.include_links=QCheckBox('Include linked inputs');self.include_links.setToolTip('Explicitly copy linked CSV/NPZ/NetCDF inputs into this study. Owned files are always included. Original inputs/model mappings stay unchanged.');files_row.addWidget(self.include_links)
        files_row.addStretch()
        self.snapshot=QLabel('No study snapshot saved or opened. Model Save does not save this table.');self.snapshot.setTextFormat(Qt.TextFormat.PlainText);self.snapshot.setWordWrap(True);layout.addWidget(self.snapshot)
        execution=QHBoxLayout();layout.addLayout(execution);execution.addWidget(QLabel('Future execution device'))
        self.device=QComboBox()
        for label,key in [('Automatic','auto'),('CPU','cpu'),('GPU / CUDA','cuda')]:self.device.addItem(label,key)
        self.device.setCurrentIndex(self.device.findData(window.application_device));execution.addWidget(self.device)
        execution.addWidget(QLabel('Per-job limit (seconds)'));self.time_limit=QSpinBox();self.time_limit.setRange(0,604800);self.time_limit.setSpecialValueText('Unlimited');self.time_limit.setValue(window.job_time_limit_seconds);execution.addWidget(self.time_limit);execution.addStretch()
        self.device.setToolTip('Saved execution intent for a future study runner; does not change the open application device. Availability is checked when executing.')
        self.time_limit.setToolTip('Saved launch-to-exit limit for each future numerical job, 0 for unlimited; not a memory quota. Does not change the application preference.')
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
        self.device.currentIndexChanged.connect(lambda:self.invalidate());self.time_limit.valueChanged.connect(lambda:self.invalidate())
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
    def axes_values(self):
        axes=[]
        for row in range(self.axes.rowCount()):
            key=self.axes.cellWidget(row,0).currentData()
            if key is None:raise ValueError('Choose a supported block parameter.')
            key=json.loads(key)
            axes.append(dict(node_id=key[0],parameter=key[1],mode=self.axes.cellWidget(row,1).currentData(),text=self.axes.cellWidget(row,2).text()))
        return axes

    def save_study(self,path=None,*,include_links=None,overwrite=False):
        from pathlib import Path
        from ..study_document import capture,write
        if self.owner.is_running() or self.owner.scene.locked or self.owner.project_workspace.busy:
            self.status.setText('Finish the job or close the geometry editor before saving a study.');return False
        try:
            study,files=capture(capture_model(self.owner),self.axes_values(),self.owner.project_workspace,
                execution=dict(device=self.device.currentData(),wall_time_seconds=self.time_limit.value()),
                include_links=self.include_links.isChecked() if include_links is None else include_links,scene=self.owner.scene)
            if path is None:
                path,_=QFileDialog.getSaveFileName(self,'Save portable study',self.study_path or 'study.margam-study','Margam study (*.margam-study)',options=QFileDialog.Option.DontConfirmOverwrite)
                if not path:return False
                if not path.endswith('.margam-study'):path+='.margam-study'
                if Path(path).exists():
                    if QMessageBox.question(self,'Replace study?','Replace '+Path(path).name+'?',QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)!=QMessageBox.StandardButton.Yes:return False
                    overwrite=True
            if self.owner.is_running() or self.owner.scene.locked or self.owner.project_workspace.busy:
                self.status.setText('A job or editor started while saving. Try again when it finishes.');return False
            write(path,study,files,overwrite=overwrite)
        except (OSError,ValueError,TypeError,KeyError,RecursionError) as error:
            self.status.setText('Study could not be saved: '+str(error));return False
        self.study_path=str(Path(path).absolute());self.snapshot.setText('Last saved snapshot: '+Path(path).name+' · '+study['study_id'][:16]+'. Save again to freeze later model/table changes.')
        self.status.setText(f"Saved portable study with {len(files)} input snapshots. No training/checkpoint/results were saved or executed.");return True

    def open_study(self,path=None):
        from pathlib import Path
        from ..study_document import read,restore
        if self.owner.is_running() or self.owner.scene.locked or self.owner.project_workspace.busy:
            self.status.setText('Finish the job or close the geometry editor before opening a study.');return False
        interactive=path is None
        if path is None:path,_=QFileDialog.getOpenFileName(self,'Open study — replaces current model and table',self.study_path or '', 'Margam study (*.margam-study)')
        if not path:return False
        workspace=None
        try:
            study,files=read(path)
            if interactive and self.axes.rowCount() and QMessageBox.question(self,'Replace study table?','Opening a study replaces this table and the current model. Save the current study first if you need to keep it. Continue?',QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)!=QMessageBox.StandardButton.Yes:return False
            if not self.owner.confirm_discard():return False
            if self.owner.is_running() or self.owner.scene.locked or self.owner.project_workspace.busy:
                self.status.setText('A job or editor started while opening. Try again when it finishes.');return False
            workspace=restore(files)
            self.owner.replace_model(study['model'],workspace=workspace)
        except (OSError,ValueError,TypeError,KeyError,RecursionError,RuntimeError) as error:
            if workspace is not None:workspace.close()
            self.status.setText('Study could not be opened: '+str(error));return False
        self.axes.setRowCount(0)
        for axis in study['axes']:
            self.add_axis();row=self.axes.rowCount()-1;combo=self.axes.cellWidget(row,0)
            combo.setCurrentIndex(combo.findData(json.dumps([axis['node_id'],axis['parameter']])))
            mode=self.axes.cellWidget(row,1);mode.setCurrentIndex(mode.findData(axis['mode']));self.axes.cellWidget(row,2).setText(axis['text'])
        self.device.setCurrentIndex(self.device.findData(study['execution']['device']));self.time_limit.setValue(study['execution']['wall_time_seconds'])
        self.study_path=str(Path(path).absolute());self.include_links.setChecked(False)
        self.snapshot.setText('Opened snapshot: '+Path(path).name+' · '+study['study_id'][:16]+'. Private input copies restored; application device/preferences unchanged.')
        self.invalidate('Study opened. Preview to recheck the restored model and parameter choices. No training was started.');return True

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
