"""Explicit mapping dialog shared by imported point sets and reference sources."""
from pathlib import Path
from PySide6.QtWidgets import QDialog,QVBoxLayout,QFormLayout,QLineEdit,QPushButton,QFileDialog,QPlainTextEdit,QDialogButtonBox,QLabel
from ..components.prepared_data import read_points


class DataMappingDialog(QDialog):
    def __init__(self,dimension,source=None,parent=None,coordinate_only=False):
        super().__init__(parent);self.setWindowTitle('Map prepared point data');self.resize(640,440)
        self.dimension=dimension;source=source or {}
        layout=QVBoxLayout(self);form=QFormLayout();layout.addLayout(form)
        self.fields={}
        defaults={'path':'','coordinates':','.join(('x','y')[:dimension]),'normals':'','values':''}
        for key,default in defaults.items():
            edit=QLineEdit(source.get(key,default));self.fields[key]=edit;form.addRow(key.capitalize(),edit)
        if coordinate_only:
            for key in ('normals','values'):self.fields[key].clear();self.fields[key].setEnabled(False)
        browse=QPushButton('Choose CSV / NPZ…');browse.clicked.connect(self.browse);form.addRow(browse)
        note=QLabel('CSV: comma-separated column names for coordinates/normals; one value column. NPZ: each mapping names an array (e.g. points, normals, values). Blank normals/values means unused. Orientation and units are supplied by you.');note.setWordWrap(True);layout.addWidget(note)
        self.preview=QPlainTextEdit();self.preview.setReadOnly(True);layout.addWidget(self.preview)
        check=QPushButton('Preview mapped rows');check.clicked.connect(self.check);layout.addWidget(check)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept_checked);buttons.rejected.connect(self.reject);layout.addWidget(buttons)

    def source(self):
        data={key:edit.text().strip() for key,edit in self.fields.items()}
        if data['path']:data['path']=str(Path(data['path']).expanduser().absolute())
        return data

    def browse(self):
        path,_=QFileDialog.getOpenFileName(self,'Choose point data','','Point data (*.csv *.npz)')
        if path:
            self.fields['path'].setText(path)
            if path.lower().endswith('.npz'):self.fields['coordinates'].setText('points')

    def check(self):
        try:
            points,normals,values=read_points(self.source(),self.dimension)
            rows=[f'{len(points)} points · normals: {normals is not None} · values: {values is not None}']
            rows.extend(str(row.tolist())+(' → '+str(values[i,0]) if values is not None else '') for i,row in enumerate(points[:8]))
            self.preview.setPlainText('\n'.join(rows));return True
        except Exception as error:self.preview.setPlainText(str(error));return False

    def accept_checked(self):
        if self.check():self.accept()
