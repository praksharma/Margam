"""Explicit mapping/preview for evaluation points and independent reference data."""
from pathlib import Path
from PySide6.QtWidgets import QDialog,QVBoxLayout,QFormLayout,QLineEdit,QLabel,QPushButton,QPlainTextEdit,QDialogButtonBox,QFileDialog
from ..components.evaluation_data import read_point_data


class EvaluationMappingDialog(QDialog):
    def __init__(self,config,source=None,parent=None,*,reference=False):
        super().__init__(parent);self.config=config;self.reference=reference;source=source or {};self.resize(700,460)
        self.setWindowTitle('Map reference data' if reference else 'Map evaluation points')
        layout=QVBoxLayout(self);form=QFormLayout();layout.addLayout(form);self.fields={}
        defaults=dict(path='',coordinates=','.join(config.coordinate_names),values=','.join(config.field_names) if reference else '',ids='')
        for key,label in [('path','CSV / NPZ file'),('coordinates','Coordinate columns / array'),('values','Reference columns / array'),('ids','Point ID column / array (optional)')]:
            edit=QLineEdit(source.get(key,defaults[key]));edit.setObjectName('evaluation_'+key);self.fields[key]=edit;form.addRow(label,edit)
        if not reference:self.fields['values'].clear();self.fields['values'].setEnabled(False)
        browse=QPushButton('Choose CSV / NPZ…');browse.clicked.connect(self.browse);form.addRow(browse)
        note=QLabel('Coordinates ordered '+', '.join(config.coordinate_names)+'. '+('Fields ordered '+', '.join(config.field_names)+'. ' if reference else '')+'CSV mappings name columns; NPZ mappings name arrays. IDs may be integers or text and must be unique per physical time. Original rows are retained. No interpolation or training-data changes.');note.setWordWrap(True);layout.addWidget(note)
        self.preview=QPlainTextEdit();self.preview.setReadOnly(True);layout.addWidget(self.preview)
        check=QPushButton('Preview mapped rows');check.clicked.connect(self.check);layout.addWidget(check)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);buttons.accepted.connect(self.accept_checked);buttons.rejected.connect(self.reject);layout.addWidget(buttons)

    def source(self):
        data={key:edit.text().strip() for key,edit in self.fields.items()}
        if data['path']:data['path']=str(Path(data['path']).expanduser().absolute())
        return data

    def browse(self):
        path,_=QFileDialog.getOpenFileName(self,'Choose prepared evaluation data','','Point data (*.csv *.npz)')
        if path:
            self.fields['path'].setText(path)
            if path.lower().endswith('.npz'):self.fields['coordinates'].setText('points');self.fields['values'].setText('values' if self.reference else '')

    def check(self):
        try:
            data=read_point_data(self.source(),self.config,reference=self.reference)
            lines=[f'{len(data.points)} supplied rows · point order preserved']
            for i,point in enumerate(data.points[:8]):lines.append((str(data.point_ids[i])+' · ' if data.point_ids is not None else '')+str(point)+(' → '+str(data.values[i]) if data.values is not None else ''))
            self.preview.setPlainText('\n'.join(lines));return True
        except Exception as error:self.preview.setPlainText(str(error));return False

    def accept_checked(self):
        if self.check():self.accept()
