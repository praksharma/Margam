"""Compare saved model definitions and the current open graph without replacing it."""
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QPushButton,QLabel,
    QCheckBox,QLineEdit,QTableWidget,QTableWidgetItem,QHeaderView,QPlainTextEdit,QFileDialog)
from .model_document import read_model,capture_model
from .model_diff import compare_models,value_text


class ModelComparison(QDialog):
    def __init__(self,window):
        super().__init__(window)
        self.window=window;self.baseline_path=None;self.comparison_path=None;self.changes=[];self.visible_changes=[]
        self.setWindowTitle('Compare model definitions');self.resize(1100,720)
        layout=QVBoxLayout(self)
        note=QLabel('Compare block settings and connections by stable ID. This is a snapshot: Refresh after edits or file changes. No models or input data are loaded into the editor.')
        note.setWordWrap(True);layout.addWidget(note)
        row=QHBoxLayout();layout.addLayout(row)
        self.baseline_label=QLabel('Baseline: choose a saved model');row.addWidget(self.baseline_label,1)
        button=QPushButton('Choose baseline…');button.clicked.connect(lambda:self.choose_file(False));row.addWidget(button)
        row=QHBoxLayout();layout.addLayout(row)
        self.comparison_label=QLabel('Comparison: current open model');row.addWidget(self.comparison_label,1)
        button=QPushButton('Choose comparison…');button.clicked.connect(lambda:self.choose_file(True));row.addWidget(button)
        button=QPushButton('Use current model');button.clicked.connect(self.use_current);row.addWidget(button)
        row=QHBoxLayout();layout.addLayout(row)
        self.presentation=QCheckBox('Include names, notes and positions');self.presentation.toggled.connect(self.refresh);row.addWidget(self.presentation)
        self.search=QLineEdit();self.search.setPlaceholderText('Filter differences…');self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.populate);row.addWidget(self.search,1)
        self.status=QLabel('Choose a baseline to compare.');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.table=QTableWidget(0,5);self.table.setHorizontalHeaderLabels(['Change','Block / ID','Field','Before','After'])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        for index,width in enumerate((115,190,230,230)):self.table.setColumnWidth(index,width)
        self.table.currentCellChanged.connect(self.show_details);layout.addWidget(self.table,3)
        self.details=QPlainTextEdit();self.details.setReadOnly(True);self.details.setPlaceholderText('Select a change for full values.');layout.addWidget(self.details,2)
        row=QHBoxLayout();layout.addLayout(row)
        button=QPushButton('Refresh');button.clicked.connect(self.refresh);row.addWidget(button)
        for label in (note,self.baseline_label,self.comparison_label,self.status):label.setTextFormat(Qt.TextFormat.PlainText)
        self.locate_button=QPushButton('Select current block');self.locate_button.clicked.connect(self.locate);self.locate_button.setEnabled(False);row.addWidget(self.locate_button)
        row.addStretch();button=QPushButton('Close');button.clicked.connect(self.close);row.addWidget(button)

    def choose_file(self,comparison=False):
        path,_=QFileDialog.getOpenFileName(self,'Choose comparison model' if comparison else 'Choose baseline model',
            str(self.window.model_dialog_directory()),'Margam model (*.margam.json *.json)')
        if path:self.set_file(path,comparison=comparison)

    def set_file(self,path,*,comparison=False):
        try:read_model(path)
        except (OSError,ValueError,TypeError,KeyError,RecursionError) as error:
            self.status.setText('Could not read selected model: '+str(error));return False
        if comparison:self.comparison_path=str(path)
        else:self.baseline_path=str(path)
        self.refresh();return True

    def use_current(self):
        self.comparison_path=None;self.refresh()

    def refresh(self):
        self.baseline_label.setText('Baseline: '+Path(self.baseline_path).name if self.baseline_path else 'Baseline: choose a saved model')
        self.comparison_label.setText('Comparison: '+Path(self.comparison_path).name if self.comparison_path else 'Comparison: current open model')
        self.changes=[]
        if self.baseline_path is None:
            self.populate();self.status.setText('Choose a baseline to compare.');return
        try:
            left=read_model(self.baseline_path)
            right=read_model(self.comparison_path) if self.comparison_path else capture_model(self.window)
            self.changes=compare_models(left,right,include_presentation=self.presentation.isChecked())
        except (OSError,ValueError,TypeError,KeyError,RecursionError) as error:
            self.populate();self.status.setText('Comparison unavailable: '+str(error));return
        self.populate()

    def populate(self):
        query=self.search.text().strip().casefold()
        self.visible_changes=[change for change in self.changes if not query or query in
            (' '.join((change.category,change.node_id,change.block,change.field,value_text(change.before),value_text(change.after)))).casefold()]
        self.table.setRowCount(0);self.details.clear();self.locate_button.setEnabled(False)
        for change in self.visible_changes:
            row=self.table.rowCount();self.table.insertRow(row)
            texts=(change.category,change.block+' ['+change.node_id[:8]+']',change.field,value_text(change.before),value_text(change.after))
            for column,text in enumerate(texts):
                item=QTableWidgetItem(text if len(text)<=150 else text[:147]+'…');self.table.setItem(row,column,item)
        self.status.setText(f'{len(self.changes)} difference(s); {len(self.visible_changes)} shown. '+
            ('Names, notes and positions included.' if self.presentation.isChecked() else 'Names, notes and positions hidden.')+
            ' Input file contents and trained weights are not compared.')
        if self.visible_changes:self.table.setCurrentCell(0,0)

    def show_details(self,*args):
        index=self.table.currentRow()
        if not 0<=index<len(self.visible_changes):self.details.clear();self.locate_button.setEnabled(False);return
        change=self.visible_changes[index]
        self.details.setPlainText(change.category+' · '+change.block+'\nBlock ID: '+change.node_id+'\nField: '+change.field+
            '\n\nBefore:\n'+value_text(change.before,True)+'\n\nAfter:\n'+value_text(change.after,True))
        self.locate_button.setEnabled(self.comparison_path is None and any(n.identifier==change.node_id for n in self.window.scene.nodes()))

    def locate(self):
        if self.comparison_path is not None:return False
        index=self.table.currentRow()
        if not 0<=index<len(self.visible_changes):return False
        identifier=self.visible_changes[index].node_id
        node=next((n for n in self.window.scene.nodes() if n.identifier==identifier),None)
        if node is None:
            self.locate_button.setEnabled(False);self.details.appendPlainText('\nBlock no longer exists in the current model. Refresh the comparison.');return False
        self.window.scene.clearSelection();node.setSelected(True);self.window.view.centerOn(node);return True
