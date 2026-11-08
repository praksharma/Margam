"""Comparison-owned read-only error norms from existing stored result arrays."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QLabel,QComboBox,
    QListWidget,QListWidgetItem,QTableWidget,QTableWidgetItem,QHeaderView,QPushButton,QToolButton)
from ..components.error_norms import error_norms,NORMS,ZERO_POLICIES,FORMULAS

STATUS={'defined':'Defined','defined_zero_reference':'Zero by selected exact-zero convention',
        'undefined_zero_reference':'Undefined: zero reference norm',
        'overflow':'Outside float64 range (overflow)','underflow':'Outside float64 range (underflow)'}


class ErrorNormsDialog(QDialog):
    def __init__(self,window,node_id):
        super().__init__(window);self.window=window;self.node_id=node_id;self.data=None;self.reports=[]
        self.setWindowTitle('Comparison error norms');self.resize(800,620)
        layout=QVBoxLayout(self)
        note=QLabel('Uses stored, aligned prediction/reference rows only. Select fields with compatible units: vector norms combine their raw values without scaling. All stored rows and physical times are included equally; these are discrete norms, not spatial integrals.')
        note.setWordWrap(True);note.setTextFormat(Qt.TextFormat.PlainText);layout.addWidget(note)
        self.fields=QListWidget();self.fields.setMaximumHeight(110);self.fields.itemChanged.connect(self.calculate);layout.addWidget(self.fields)
        row=QHBoxLayout();layout.addLayout(row);row.addWidget(QLabel('Norm'))
        self.norm=QComboBox()
        for key,label in NORMS.items():self.norm.addItem(label,key)
        self.norm.currentIndexChanged.connect(self.calculate);row.addWidget(self.norm,1)
        info=QToolButton();info.setText('i');info.setAccessibleName('Error norm definitions');info.clicked.connect(self.show_definitions);row.addWidget(info)
        row=QHBoxLayout();layout.addLayout(row);row.addWidget(QLabel('Zero reference'))
        self.zero=QComboBox()
        for key,label in ZERO_POLICIES.items():self.zero.addItem(label,key)
        self.zero.currentIndexChanged.connect(self.calculate);row.addWidget(self.zero,1)
        self.status=QLabel();self.status.setWordWrap(True);self.status.setTextFormat(Qt.TextFormat.PlainText);layout.addWidget(self.status)
        self.table=QTableWidget(0,3);self.table.setHorizontalHeaderLabels(['Selected fields / grouping','Value','Status'])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True);layout.addWidget(self.table)
        row=QHBoxLayout();layout.addLayout(row)
        refresh=QPushButton('Refresh from current result');refresh.clicked.connect(self.refresh);row.addWidget(refresh)
        row.addStretch();close=QPushButton('Close');close.clicked.connect(self.close);row.addWidget(close)

    def show_definitions(self):
        from .setting_help import show_information
        text='For selected fields, || · || is the Euclidean magnitude at one stored row. For one field it is the absolute value.\n\n'
        text+='\n\n'.join(NORMS[key]+':\n'+FORMULAS[key] for key in NORMS)
        text+='\n\nRelative norms are ratios of global norms, not averages of pointwise relative errors. A zero reference makes the ratio undefined by default, including 0/0. The optional exact-zero convention returns 0 only when both prediction and reference are exactly zero. No epsilon or field scaling is inserted. Vector RMS divides by rows, not by rows times components. All physical times/duplicate rows are included equally; no spatial quadrature is inferred.'
        return show_information(self,'Error norm definitions',text)

    def owner(self):
        return next((node for node in self.window.scene.nodes() if node.identifier==self.node_id and node.kind=='comparison'),None)

    def current_data(self):
        if self.owner() is None:return None
        data=(self.window.result or {}).get('outputs',{}).get(self.node_id)
        return data if data is not None and data.get('reference') is not None else None

    def clear(self,message):
        self.data=None;self.reports=[];self.table.setRowCount(0);self.status.setText(message)

    def check_current(self,force=False):
        owner=self.owner()
        if owner is not None:self.setWindowTitle(owner.title+' · error norms')
        data=self.current_data()
        if force:self.clear('Results changed. Refresh after training or evaluating this Comparison block.')
        elif self.data is not None and self.data is not data:
            self.clear('New comparison data available. Refresh to calculate norms.' if data is not None else 'No current comparison data. Train or evaluate this block with an explicit reference.')

    def refresh(self):
        data=self.current_data()
        if data is None:
            self.clear('No current comparison data. Train or evaluate this Comparison block with an explicit reference.');return
        selected=[self.fields.item(i).text() for i in range(self.fields.count()) if self.fields.item(i).checkState()==Qt.CheckState.Checked]
        names=data.get('fields',list(data['config'].field_names))
        selected=[name for name in selected if name in names] or names[:1]
        self.fields.blockSignals(True);self.fields.clear()
        for name in names:
            item=QListWidgetItem(name);item.setFlags(item.flags()|Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if name in selected else Qt.CheckState.Unchecked);self.fields.addItem(item)
        self.fields.blockSignals(False);self.data=data;self.check_current();self.calculate()

    def calculate(self,*args):
        self.zero.setEnabled(str(self.norm.currentData()).startswith('relative_'))
        if self.data is None or self.data is not self.current_data():
            self.clear('No current comparison data. Refresh after training or evaluating this block.');return
        selected=[self.fields.item(i).text() for i in range(self.fields.count()) if self.fields.item(i).checkState()==Qt.CheckState.Checked]
        self.reports=[];self.table.setRowCount(0)
        if not selected:self.status.setText('Select at least one field.');return
        try:
            names=self.data.get('fields',list(self.data['config'].field_names))
            groups=[selected]+[[name] for name in selected] if len(selected)>1 else [selected]
            for fields in groups:
                self.reports.append(error_norms(self.data['values'],self.data['reference'],fields=names,selected=fields,
                    norm=self.norm.currentData(),zero_reference=self.zero.currentData()))
        except (ValueError,TypeError,OverflowError) as error:
            self.reports=[];self.status.setText('Cannot calculate error norms: '+str(error));return
        self.table.setRowCount(len(self.reports))
        for row,report in enumerate(self.reports):
            group=', '.join(report['fields'])+(' (vector)' if report['components']>1 else '')
            value='—' if report['value'] is None else f"{report['value']:.12g}"
            for column,text in enumerate((group,value,STATUS[report['status']])):self.table.setItem(row,column,QTableWidgetItem(text))
        first=self.reports[0]
        self.status.setText(f"{first['rows']} stored row(s) · "+('dimensionless ratio' if first['relative'] else 'raw field units')+
            ' · no quadrature or interpolation · '+self.data.get('reference_name','Explicit reference'))
