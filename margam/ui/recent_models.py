"""Local-only recent model navigation; never embedded in model/run documents."""
from pathlib import Path
from PySide6.QtWidgets import QMessageBox

LIMIT=10


def valid_paths(value):
    if isinstance(value,str):value=[value]
    if not isinstance(value,list):return []
    paths=[]
    for item in value[:100]:
        if not isinstance(item,str) or not item or len(item)>4096 or '\x00' in item:continue
        path=Path(item)
        if not path.is_absolute():continue
        text=str(path)
        if text not in paths:paths.append(text)
        if len(paths)==LIMIT:break
    return paths


class RecentModels:
    def recent_model_paths(self):
        state=getattr(self,'desktop_state',None)
        return valid_paths(state.settings.value('recent_models',[]) if state is not None else getattr(self,'_recent_models',[]))

    def store_recent_models(self,paths):
        paths=valid_paths(paths)
        state=getattr(self,'desktop_state',None)
        if state is not None:
            state.settings.setValue('recent_models',paths);state.settings.sync()
        else:self._recent_models=paths

    def remember_recent_model(self,path):
        text=str(Path(path).absolute())
        self.store_recent_models([text]+[p for p in self.recent_model_paths() if p!=text])

    def refresh_recent_models(self):
        menu=self.recent_models_menu;menu.clear()
        paths=self.recent_model_paths()
        locked=self.is_running() or self.scene.locked
        if not paths:
            menu.addAction('No recent models').setEnabled(False);return
        for index,path in enumerate(paths,1):
            # Escape ampersands so filenames do not turn into menu mnemonics.
            action=menu.addAction(f'{index}. '+Path(path).name.replace('&','&&'))
            action.setToolTip(path);action.setStatusTip(path);action.setEnabled(not locked)
            action.triggered.connect(lambda checked=False,p=path:self.open_recent_model(p))
        menu.addSeparator()
        menu.addAction('Clear recent models').triggered.connect(lambda:self.store_recent_models([]))

    def open_recent_model(self,path):
        if self.is_running() or self.scene.locked:return False
        if not Path(path).is_file():
            answer=QMessageBox.question(self,'Recent model unavailable',
                'This model has moved or is unavailable:\n'+path+'\n\nRemove it from recent models? The project files will not be deleted.',
                QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)
            if answer==QMessageBox.StandardButton.Yes:
                self.store_recent_models([p for p in self.recent_model_paths() if p!=path])
            return False
        return self.load_model(path)
