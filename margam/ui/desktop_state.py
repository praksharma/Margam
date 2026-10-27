"""Local layout preferences and locked draft recovery, inspired by GeoSampler.

Model recovery is separate from saved model files and numerical run archives.
"""
from pathlib import Path
from uuid import uuid4
from PySide6.QtCore import QSettings, QStandardPaths, QTimer, QLockFile
from PySide6.QtWidgets import QMessageBox, QApplication
from .model_document import read_model, write_model, capture_model
from .theme import configure_theme, theme_name


class DesktopState:
    def __init__(self,window,directory=None):
        self.window=window
        self.directory=Path(directory) if directory is not None else Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericDataLocation))/'Margam'
        self.directory.mkdir(parents=True,exist_ok=True)
        self.settings=QSettings(str(self.directory/'desktop.ini'),QSettings.Format.IniFormat)
        self.recovery_directory=self.directory/'recovery';self.recovery_directory.mkdir(exist_ok=True)
        self.path=self.recovery_directory/(uuid4().hex+'.margam.json')
        self.lock=None
        self.timer=QTimer(window);self.timer.setSingleShot(True);self.timer.setInterval(1500)
        self.timer.timeout.connect(self.autosave)
        window.undo_stack.indexChanged.connect(self.queue)
        window.undo_stack.cleanChanged.connect(self.queue)

    def restore(self):
        w=self.window
        if self.settings.contains('geometry'):w.restoreGeometry(self.settings.value('geometry'))
        if self.settings.contains('layout'):w.restoreState(self.settings.value('layout'))
        configure_theme(QApplication.instance(),self.settings.value('theme','light'))
        collapsed=self.settings.value('collapsed',[]) or []
        if isinstance(collapsed,str):collapsed=[collapsed]
        tree=w.library.tree
        for i in range(tree.topLevelItemCount()):
            item=tree.topLevelItem(i);item.setExpanded(item.text(0) not in collapsed)

    def queue(self,*args):
        self.timer.start()

    def autosave(self):
        w=self.window
        if w.undo_stack.isClean():self.clear();return False
        try:
            if self.lock is None:
                self.lock=QLockFile(str(self.path)+'.lock');self.lock.setStaleLockTime(0)
                if not self.lock.tryLock(0):self.lock=None;return False
            write_model(self.path,capture_model(w));return True
        except (OSError,ValueError,TypeError) as error:
            w.statusBar().showMessage('Draft recovery could not be saved: '+str(error));return False

    def clear(self):
        try:self.path.unlink(missing_ok=True)
        except OSError as error:self.window.statusBar().showMessage('Could not remove recovery draft: '+str(error))
        if self.lock is not None:self.lock.unlock();self.lock=None

    def offer_recovery(self):
        candidates=sorted(self.recovery_directory.glob('*.margam.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        for path in candidates[:20]:
            if path==self.path:continue
            lock=QLockFile(str(path)+'.lock');lock.setStaleLockTime(0)
            if not lock.tryLock(0):continue
            try:
                try:data=read_model(path)
                except (OSError,ValueError,TypeError):continue
                answer=QMessageBox.question(self.window,'Recover unsaved model?',
                    'An unsaved model from a previous session is available. Recover it?',
                    QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No|QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Yes)
                if answer==QMessageBox.StandardButton.Cancel:return False
                if answer==QMessageBox.StandardButton.No:path.unlink(missing_ok=True);continue
                if not self.window.confirm_discard():return False
                self.window.replace_model(data)
                self.window.undo_stack.resetClean()
                if self.autosave():path.unlink(missing_ok=True)
                self.window.statusBar().showMessage('Recovered unsaved model · use Save model as to keep it')
                return True
            finally:lock.unlock()
        return False

    def close(self):
        self.timer.stop()
        w=self.window
        self.settings.setValue('geometry',w.saveGeometry());self.settings.setValue('layout',w.saveState())
        self.settings.setValue('theme',theme_name())
        tree=w.library.tree
        self.settings.setValue('collapsed',[tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount()) if not tree.topLevelItem(i).isExpanded()])
        self.settings.sync();self.clear()
