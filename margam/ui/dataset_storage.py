"""The concrete copy/link choice shared by point-data mapping dialogs."""
from pathlib import Path
from PySide6.QtWidgets import QWidget,QVBoxLayout,QComboBox,QLabel


def mapping_path(source):
    result=dict(source)
    if result.get('path') and not result['path'].startswith('margam-owned:'):
        result['path']=str(Path(result['path']).expanduser().absolute())
    return result


def show_source_path(edit,path,workspace):
    logical=workspace.portable(path) if workspace is not None else path
    if logical.startswith('margam-owned:'):
        caption='Project copy: '+Path(logical).name
        edit.setProperty('owned_reference',logical);edit.setProperty('owned_caption',caption)
        edit.setText(caption);edit.setToolTip('Stored with this project: '+logical.removeprefix('margam-owned:'))


def source_from_fields(fields):
    data={key:edit.text().strip() for key,edit in fields.items()}
    path=fields['path']
    if path.property('owned_reference') and data['path']==path.property('owned_caption'):
        data['path']=path.property('owned_reference')
    return mapping_path(data)


class DatasetStorage(QWidget):
    def __init__(self,source,parent=None):
        super().__init__(parent);self.workspace=getattr(parent,'project_workspace',None)
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        self.choice=QComboBox();self.choice.setObjectName('dataset_storage')
        self.choice.addItem('Copy into project','copy');self.choice.addItem('Link existing file','link')
        path=source.get('path','')
        owned=self.workspace is not None and self.workspace.portable(path).startswith('margam-owned:')
        self.choice.setCurrentIndex(0 if self.workspace is not None and (not path or owned) else 1)
        if self.workspace is None:self.choice.setEnabled(False)
        layout.addWidget(self.choice);self.note=QLabel();self.note.setWordWrap(True);layout.addWidget(self.note)
        self.choice.currentIndexChanged.connect(self.describe);self.describe()

    def describe(self):
        self.note.setText('A fixed copy travels with this project, Save As, recovery and run archives. Later changes to the original file are not applied automatically.'
                          if self.choice.currentData()=='copy' else
                          'Uses the external file at its original location. Keep it available when moving this project. Changed training data cannot resume an older checkpoint.')

    def resolve(self,source):return self.workspace.resolve_source(source) if self.workspace is not None else source

    def validate_choice(self,source):
        if self.choice.currentData()=='link' and self.workspace is not None and self.workspace.portable(source['path']).startswith('margam-owned:'):
            raise ValueError('Choose an external file to create a link, or keep Copy into project.')

    def store(self,source,validator):
        self.validate_choice(source)
        if self.workspace is None:validator(source);return source
        return self.workspace.store_dataset(source,copy=self.choice.currentData()=='copy',validator=validator)
