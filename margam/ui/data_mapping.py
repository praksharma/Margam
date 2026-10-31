"""Explicit mapping dialog shared by imported point sets and reference sources."""
from PySide6.QtWidgets import QDialog,QVBoxLayout,QFormLayout,QLineEdit,QPushButton,QFileDialog,QPlainTextEdit,QDialogButtonBox,QLabel
from ..components.prepared_data import read_points

from .dataset_storage import DatasetStorage,show_source_path,source_from_fields


class DataMappingDialog(QDialog):
    def __init__(self,dimension,source=None,parent=None,coordinate_only=False,components=1,allow_values=True,require_values=False):
        super().__init__(parent);self.setWindowTitle('Map prepared point data');self.resize(640,440)
        self.dimension=dimension;self.components=components;self.require_values=require_values;source=source or {}
        layout=QVBoxLayout(self);form=QFormLayout();layout.addLayout(form)
        self.fields={}
        defaults={'path':'','coordinates':','.join(('x','y')[:dimension]),'normals':'','values':''}
        for key,default in defaults.items():
            edit=QLineEdit(source.get(key,default));self.fields[key]=edit;form.addRow(key.capitalize(),edit)
        if coordinate_only:
            for key in ('normals','values'):self.fields[key].clear();self.fields[key].setEnabled(False)
        if not allow_values:self.fields['values'].clear();self.fields['values'].setEnabled(False)
        show_source_path(self.fields['path'],source.get('path',''),getattr(parent,'project_workspace',None))
        self.storage=DatasetStorage(source,parent);form.addRow('File handling',self.storage)
        browse=QPushButton('Choose CSV / NPZ…');browse.clicked.connect(self.browse);form.addRow(browse)
        note=QLabel('CSV: comma-separated column names for coordinates/normals; value columns ordered by field. NPZ: each mapping names an array (e.g. points, normals, values). Blank normals/values means unused. Orientation and units are supplied by you.');note.setWordWrap(True)
        if components==3:note.setText(note.text()+' Values are ordered u, v, p.')
        layout.addWidget(note)
        self.preview=QPlainTextEdit();self.preview.setReadOnly(True);layout.addWidget(self.preview)
        check=QPushButton('Preview mapped rows');check.clicked.connect(self.check);layout.addWidget(check)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept_checked);buttons.rejected.connect(self.reject);layout.addWidget(buttons)

    def source(self):
        return source_from_fields(self.fields)

    def browse(self):
        path,_=QFileDialog.getOpenFileName(self,'Choose point data','','Point data (*.csv *.npz)')
        if path:
            self.fields['path'].setText(path);self.fields['path'].setToolTip('')
            if path.lower().endswith('.npz'):self.fields['coordinates'].setText('points')

    def read_source(self,source):
        result=read_points(source,self.dimension,self.components)
        if self.require_values and result[2] is None:raise ValueError('Map the explicit reference value columns/array before applying.')
        return result

    def store_source(self):return self.storage.store(self.source(),self.read_source)

    def check(self):
        try:
            self.storage.validate_choice(self.source())
            points,normals,values=self.read_source(self.storage.resolve(self.source()))
            rows=[f'{len(points)} points · normals: {normals is not None} · values: {values is not None}']
            rows.extend(str(row.tolist())+(' n='+str(normals[i].tolist()) if normals is not None else '')+(' → '+str(values[i].tolist()) if values is not None else '') for i,row in enumerate(points[:8]))
            self.preview.setPlainText('\n'.join(rows));return True
        except Exception as error:self.preview.setPlainText(str(error));return False

    def accept_checked(self):
        if self.check():self.accept()
