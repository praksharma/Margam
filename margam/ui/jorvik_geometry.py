"""Workbench-style document editing and explicit sample updates; no embedded CAD."""
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import sys
from zipfile import BadZipFile
from uuid import uuid4
from PySide6.QtCore import QProcess, QIODevice, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QFormLayout, QLabel, QComboBox, QLineEdit, QMessageBox, QPushButton, QFileDialog
from ..components.jorvik_data import records, inspect_selection
from .graph_model import parameters_for


def document_relative(identifier):
    return 'geometry/' + hashlib.sha256(identifier.encode()).hexdigest()[:24] + '/domain.jorvik'


class DatasetDialog(QDialog):
    def __init__(self, path, previous, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Update from Jorvik · planar 2D samples')
        layout=QVBoxLayout(self);form=QFormLayout();layout.addLayout(form)
        note=QLabel('Choose one interior dataset and named boundaries. Geometry supplies samples, not PDEs or BC types. Normals are used only when verified.');note.setWordWrap(True);layout.addWidget(note)
        self.axes=QComboBox()
        for axes in ((0,1),(0,2),(1,2)):self.axes.addItem('Problem x,y ← source '+','.join('xyz'[axis] for axis in axes),axes)
        self.axes.setCurrentIndex(((0,1),(0,2),(1,2)).index(tuple(previous.get('axes',[0,1]))));form.addRow('Coordinate plane',self.axes)
        self.rows=[]
        for record in records(path):
            metadata=record['metadata'];identity=metadata.get('boundary_region',{}).get('id') or 'label:'+record['label']
            old=previous.get('datasets',{}).get(identity,{})
            choice=QComboBox();choice.addItems(['Ignore','Interior','Boundary'])
            choice.setCurrentText(old.get('role','Ignore').title())
            alias=QLineEdit(old.get('name',record['label']));alias.setMaxLength(64)
            form.addRow(record['label']+' · '+str(metadata.get('count','?'))+' points',choice)
            form.addRow('Boundary name',alias)
            self.rows.append((record['label'],choice,alias))
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)

    def selection(self):
        return {label:(choice.currentText().lower(),name.text().strip()) for label,choice,name in self.rows}


class JorvikGeometryActions:
    def jorvik_document(self,node,create=False):
        reference=node.settings.get('jorvik_files',{}).get('document','margam-owned:'+document_relative(node.identifier))
        path=Path(self.project_workspace.resolve(reference))
        if create and not path.exists():
            from jorvik.project import create_empty_project
            path.parent.mkdir(parents=True,exist_ok=True)
            create_empty_project(path)
        return path

    def jorvik_export(self,node):
        reference=node.settings.get('jorvik_files',{}).get('export')
        return Path(self.project_workspace.resolve(reference)) if reference else self.jorvik_document(node).with_suffix('.nc')

    def browse_jorvik(self,node,kind,path=None):
        if self.is_running() or self.scene.locked:return False
        if kind not in ('document','export'):raise ValueError('Unknown Jorvik file role.')
        if path is None:
            title,pattern=('Choose existing Jorvik project','Jorvik project (*.jorvik *.gsproj)') if kind=='document' else ('Choose Jorvik sample export','NetCDF samples (*.nc)')
            path,_=QFileDialog.getOpenFileName(self,title,str(self.jorvik_document(node).parent),pattern)
        if not path:return False
        copied=None
        try:
            source=Path(path)
            if kind=='document':
                if source.suffix.lower() not in ('.jorvik','.gsproj'):raise ValueError('Choose a Jorvik project file.')
                from jorvik.project import load_project
                load_project(source)
            else:
                if source.suffix.lower()!='.nc':raise ValueError('Choose a NetCDF sample export.')
                records(source)
            relative=Path(document_relative(node.identifier)).parent/'imports'/uuid4().hex/('domain.jorvik' if kind=='document' else 'domain.nc')
            copied=Path(self.project_workspace.resolve('margam-owned:'+relative.as_posix()))
            copied.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,copied)
            # Check the owned copy before changing any block settings.
            if kind=='document':load_project(copied)
            else:records(copied)
            old=deepcopy(node.settings);new=deepcopy(old)
            files=new.setdefault('jorvik_files',{})
            files[kind]='margam-owned:'+relative.as_posix()
            if kind=='document':
                files.pop('export',None);files['needs_update']=True
            def apply(settings):
                node.settings=deepcopy(settings);self.graph_changed();self.selection_changed()
            self.push_edit('Choose Jorvik '+kind,lambda:apply(new),lambda:apply(old))
            self.statusBar().showMessage('Copied '+source.name+' into this project · Update from Jorvik to apply samples')
            return True
        except (OSError,ValueError,TypeError,KeyError,ImportError,BadZipFile) as error:
            if copied is not None:copied.unlink(missing_ok=True)
            QMessageBox.warning(self,'Jorvik file could not be selected',str(error));return False

    def use_adjacent_jorvik_export(self,node):
        if self.is_running() or self.scene.locked:return False
        old=deepcopy(node.settings);new=deepcopy(old)
        new.get('jorvik_files',{}).pop('export',None)
        if old==new:return False
        def apply(settings):
            node.settings=deepcopy(settings);self.graph_changed();self.selection_changed()
        self.push_edit('Use adjacent Jorvik export',lambda:apply(new),lambda:apply(old))
        return True

    def edit_jorvik(self,node):
        if self.is_running() or self.scene.locked:return False
        try:
            path=self.jorvik_document(node)
            if not path.is_file():raise FileNotFoundError('The Jorvik document is missing. Restore the project companion folder before editing.')
        except (OSError,ValueError,ImportError) as error:
            QMessageBox.warning(self,'Jorvik could not be opened',str(error));return False
        process=QProcess(self)
        process.setProgram(sys.executable);process.setArguments(['-m','jorvik',str(path)])
        process.setWorkingDirectory(str(path.parent))
        # File output avoids unbounded console buffering in the GUI process.
        process.setStandardErrorFile(str(path.parent/'editor.log'))
        process.setStandardOutputFile(str(path.parent/'editor.log'),QIODevice.OpenModeFlag.Append)
        self.jorvik_process=process;self.project_workspace.busy=True;self.set_busy(True)
        self.cancel_button.setEnabled(False);self.stop_action.setEnabled(False)
        def finished(*args):
            if getattr(self,'jorvik_process',None) is not process:return
            self.jorvik_process=None;self.project_workspace.busy=False;self.set_busy(False)
            self.undo_stack.resetClean()
            if self.desktop_state is not None:self.desktop_state.queue()
            self.statusBar().showMessage('Jorvik closed · Update from Jorvik to apply exported samples' if not args or args[0]==0 else 'Jorvik exited with an error · check editor.log in its working folder')
            process.deleteLater()
        process.finished.connect(finished)
        def failed(error):
            self.statusBar().showMessage('Jorvik launch failed · check editor.log')
            if error==QProcess.ProcessError.FailedToStart:finished()
        process.errorOccurred.connect(failed);process.start()
        self.statusBar().showMessage('Editing in Jorvik · close its window before saving or updating Margam')
        return True

    def update_jorvik(self,node,selections=None,axes=None,confirm=True):
        if self.is_running() or self.scene.locked:return False
        snapshot=None
        try:
            path=self.jorvik_export(node)
            previous=node.settings.get('jorvik',{})
            if selections is None:
                dialog=DatasetDialog(path,previous,self)
                if dialog.exec()!=QDialog.DialogCode.Accepted:return False
                selections=dialog.selection();axes=dialog.axes.currentData()
            report=inspect_selection(path,selections,axes or (0,1))
            for name,record in report['boundaries'].items():
                for old_identity,old_record in previous.get('datasets',{}).items():
                    if old_record['role']=='boundary' and old_record['name']==name and old_identity!=record['identity']:
                        raise ValueError('Boundary '+name+' now identifies a different region. Choose a new boundary name and assign its physics explicitly.')
            old_boundaries={record['name'] for record in previous.get('datasets',{}).values() if record['role']=='boundary'}
            removed=old_boundaries-set(report['boundaries'])
            text='Apply '+str(len(report['boundaries']))+' boundary sets and one interior dataset? This clears trained results.'
            if removed:text+='\nRemoved or renamed boundary assignments: '+', '.join(sorted(removed))
            if confirm and QMessageBox.question(self,'Apply Jorvik samples?',text,QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)!=QMessageBox.StandardButton.Yes:return False
            from .app import Node, Wire
            sources={}
            for wire in self.scene.wires:
                if wire.source.parentItem() is node:
                    target=wire.target.parentItem()
                    if target.kind not in ('interior_data','boundary_data'):
                        raise ValueError('Jorvik geometry connects to Interior data and Boundary data, not built-in samplers.')
                    if target.kind in sources:raise ValueError('Use one Interior data and one Boundary data block per Jorvik geometry.')
                    sources[target.kind]=target
            created=[]
            for kind,title,dy in [('interior_data','Interior data',-80),('boundary_data','Boundary data',100)]:
                if kind not in sources:
                    source=Node(title,'',node.pos().x()+250,node.pos().y()+dy);sources[kind]=source;created.append(source)
            snapshot=path.parent/'snapshots'/(uuid4().hex+'.nc');snapshot.parent.mkdir(exist_ok=True)
            shutil.copyfile(path,snapshot)
            # Validate the copied bytes, not a potentially changing editor export.
            copied=inspect_selection(snapshot,selections,axes or (0,1))
            if copied['sha256']!=report['sha256']:raise ValueError('The export changed during import. Retry the update.')
            reference='margam-owned:'+snapshot.relative_to(self.project_workspace.directory).as_posix()
            def mapping(source):return dict(source,path=reference)
            physics_nodes=[wire.target.parentItem() for wire in self.scene.wires if wire.source.parentItem() is sources['boundary_data'] and wire.target.parentItem().kind=='bc']
            affected=[node,*sources.values(),*physics_nodes]
            old={item.identifier:deepcopy(item.settings) for item in affected}
            new=deepcopy(old)
            lower,upper=report['bounds'];parameters=new[node.identifier]['parameters']
            parameters.update(shape='Prepared2D',x_min=lower[0],x_max=upper[0],y_min=lower[1],y_max=upper[1])
            datasets={}
            for record in records(snapshot):
                if selections.get(record['label'],('ignore',''))[0]=='interior':
                    identity=record['metadata'].get('boundary_region',{}).get('id') or 'label:'+record['label']
                    datasets[identity]=dict(role='interior',name=record['label'])
            for name,record in report['boundaries'].items():datasets[record['identity']]=dict(role='boundary',name=name)
            new[node.identifier]['jorvik']=dict(axes=report['axes'],datasets=datasets,sha256=report['sha256'])
            if 'jorvik_files' in new[node.identifier]:new[node.identifier]['jorvik_files']['needs_update']=False
            interior=new[sources['interior_data'].identifier]['parameters'];interior.update(interior_data=mapping(report['interior']),adaptive_sampling={})
            boundary=new[sources['boundary_data'].identifier]['parameters'];boundary['boundary_data']={name:mapping(record['mapping']) for name,record in report['boundaries'].items()}
            for physics_node in physics_nodes:
                parameters=new[physics_node.identifier]['parameters']
                for name in removed:
                    for key in ('bc_types','bc_values','bc_targets','bc_operators'):
                        parameters[key].pop(name,None)
                    if name in ('left','right','bottom','top'):
                        parameters['bc_types'][name]='Dirichlet';parameters['bc_values'][name]=0.
            wires=[Wire(node.output,source.input) for source in created]
            def apply(settings,add):
                if add:
                    for source in created:self.scene.addItem(source)
                    for wire in wires:self.scene.addItem(wire);self.scene.wires.append(wire);wire.update_path()
                else:
                    for wire in wires:self.scene.wires.remove(wire);self.scene.removeItem(wire)
                    for source in created:self.scene.removeItem(source)
                for item in affected:item.settings=deepcopy(settings[item.identifier])
                self.graph_changed();self.selection_changed()
            def redo():apply(new,True)
            self.push_edit('Update from Jorvik',redo,lambda:apply(old,False))
            self.statusBar().showMessage('Applied Jorvik snapshot · assign BC physics and connect supplied evaluation points')
            return True
        except (OSError,ValueError,TypeError,KeyError,ImportError) as error:
            if snapshot is not None:snapshot.unlink(missing_ok=True)
            QMessageBox.warning(self,'Jorvik update could not be applied',str(error));return False

    def jorvik_fields(self,node):
        path=self.jorvik_document(node)
        relative=node.settings.get('jorvik_files',{}).get('document','margam-owned:'+document_relative(node.identifier)).removeprefix('margam-owned:')
        label=QLabel(relative);label.setToolTip(str(path));label.setWordWrap(True);self.form.addRow('Geometry document',label)
        browse=QPushButton('Browse existing project… (optional)');browse.clicked.connect(lambda:self.browse_jorvik(node,'document'));self.form.addRow(browse)
        export=self.jorvik_export(node)
        label=QLabel(export.name+(' · selected copy' if node.settings.get('jorvik_files',{}).get('export') else ' · beside geometry document'));label.setToolTip(str(export));self.form.addRow('Sample export',label)
        browse=QPushButton('Browse sample export… (optional)');browse.clicked.connect(lambda:self.browse_jorvik(node,'export'));self.form.addRow(browse)
        if node.settings.get('jorvik_files',{}).get('export'):
            reset=QPushButton('Use export beside geometry document');reset.clicked.connect(lambda:self.use_adjacent_jorvik_export(node));self.form.addRow(reset)
        folder=QPushButton('Open working folder');folder.clicked.connect(lambda:QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent))));self.form.addRow(folder)
        note=QLabel('In Jorvik, save the document and export samples as domain.nc in the same folder. Close Jorvik, then Update. Choose Interior / Boundary datasets explicitly. Supported: axis-aligned planar 2D scalar problems; evaluation points remain separate.');note.setWordWrap(True);self.form.addRow(note)
        for text,action in [('Edit in Jorvik',self.edit_jorvik),('Update from Jorvik',self.update_jorvik)]:
            button=QPushButton(text);button.clicked.connect(lambda checked=False,callback=action:callback(node));self.form.addRow(button)
