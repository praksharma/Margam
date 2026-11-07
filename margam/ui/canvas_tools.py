"""Concrete presentation tools: notes, blank drafts and dependency navigation."""
from html import escape
from dataclasses import asdict
from PySide6.QtCore import Qt
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import QMenu,QPlainTextEdit
from shiboken6 import isValid
from ..templates.poisson import PoissonConfig
from .model_document import capture_model


def dependency_paths(seeds,edges,mode):
    """Return reachable IDs and edge indices; handle branching and cycles."""
    found=set(seeds);selected_edges=set()
    for reverse in ([True,False] if mode=='both' else [mode=='upstream'] if mode!='off' else []):
        adjacency={}
        for index,(source,target) in enumerate(edges):
            a,b=(target,source) if reverse else (source,target)
            adjacency.setdefault(a,[]).append((b,index))
        seen=set(seeds);pending=list(seeds)
        while pending:
            for target,index in adjacency.get(pending.pop(),[]):
                selected_edges.add(index);found.add(target)
                if target not in seen:seen.add(target);pending.append(target)
    return (found,selected_edges) if mode!='off' else (set(),set())


class CanvasTools:
    def setup_canvas_tools(self):
        self.show_grid_action=self.view_menu.addAction('Show grid');self.show_grid_action.setCheckable(True)
        self.snap_grid_action=self.view_menu.addAction('Snap to grid');self.snap_grid_action.setCheckable(True)
        self.show_grid_action.setToolTip('Show a 20-unit canvas grid; visual preference only')
        self.snap_grid_action.setToolTip('Snap new items and drag release to the 20-unit grid; group offsets stay unchanged')
        self.show_grid_action.toggled.connect(lambda checked:self.set_canvas_preference('show_grid',checked))
        self.snap_grid_action.toggled.connect(lambda checked:self.set_canvas_preference('snap_grid',checked))
        menu=self.view_menu.addMenu('Highlight dependencies');group=QActionGroup(self);group.setExclusive(True)
        self.dependency_actions={};self.dependency_mode='off'
        for key,label in [('off','Off'),('upstream','Upstream'),('downstream','Downstream'),('both','Both directions')]:
            action=menu.addAction(label);action.setCheckable(True);action.setChecked(key=='off');group.addAction(action)
            action.triggered.connect(lambda checked=False,k=key:self.set_dependency_mode(k));self.dependency_actions[key]=action
        self.scene.selectionChanged.connect(self.refresh_dependencies)
        self.scene.presentationChanged.connect(self.presentation_changed)

    def set_canvas_preference(self,key,value):
        setattr(self.scene,key,bool(value));self.scene.update()
        state=getattr(self,'desktop_state',None)
        if state is not None:state.settings.setValue(key,bool(value))

    def set_dependency_mode(self,mode):
        if mode not in self.dependency_actions:return
        self.dependency_mode=mode;self.dependency_actions[mode].setChecked(True)
        state=getattr(self,'desktop_state',None)
        if state is not None:state.settings.setValue('dependency_mode',mode)
        self.refresh_dependencies()

    def refresh_dependencies(self):
        if not isValid(self.scene):return
        seeds={n.identifier for n in self.scene.nodes() if n.isSelected() and n.kind!='annotation'}
        edges=[(w.source.parentItem().identifier,w.target.parentItem().identifier) for w in self.scene.wires]
        nodes,wires=dependency_paths(seeds,edges,getattr(self,'dependency_mode','off'))
        for node in self.scene.nodes():node.dependency_highlight=node.identifier in nodes;node.update()
        for index,wire in enumerate(self.scene.wires):wire.dependency_highlight=index in wires;wire.update()

    def presentation_changed(self):
        # Notes affect the editable document/recovery, never the numerical state.
        self.refresh_dependencies();self.update_document_title();self.update_clipboard_actions();self.update_layout_actions()
        for node in self.scene.nodes():
            if node.kind=='annotation':self.refresh_note(node)

    def refresh_note(self,node):
        from .block_labels import TYPE_NAMES,block_tooltip
        node.type_title=TYPE_NAMES[node.kind]
        if node.settings.get('label'):node.title=node.settings['label']
        node.detail=node.settings.get('annotation_text','');node.setToolTip(block_tooltip(node,node.title+'\n'+node.detail));node.update()
        if node.isSelected():
            self.detail.setText('Canvas note · no numerical effect')
            body=self.property_scroll.widget()
            for edit in body.findChildren(QPlainTextEdit) if body is not None else ():
                if edit.property('search_key')=='annotation_text' and edit.toPlainText()!=node.detail:
                    position=edit.textCursor().position();blocked=edit.blockSignals(True);edit.setPlainText(node.detail)
                    cursor=edit.textCursor();cursor.setPosition(min(position,len(node.detail)));edit.setTextCursor(cursor);edit.blockSignals(blocked)

    def change_note(self,node,text,token=None):
        if self.is_running() or self.scene.locked or node.scene() is not self.scene or node.kind!='annotation':return False
        if not isinstance(text,str) or len(text)>4096 or '\x00' in text:return False
        old=node.settings.get('annotation_text','')
        if old==text:return False
        def apply(value):node.settings['annotation_text']=value;self.refresh_note(node)
        key=(node.identifier,'annotation_text',token) if token is not None else None
        self.push_edit('Edit note',lambda:apply(text),lambda:apply(old),key)
        return True

    def new_blank_model(self):
        if self.is_running() or self.scene.locked or not self.confirm_discard():return False
        document=capture_model(self);document.update(config=asdict(PoissonConfig()),nodes=[],wires=[])
        self.replace_model(document)
        self.statusBar().showMessage('Blank model · add components from the library or start with File → Examples')
        return True

    def create_group_menu(self):
        menu=QMenu(self);self.update_layout_actions()
        menu.addAction(self.fit_selection_action)
        align=menu.addMenu('Align');spacing=menu.addMenu('Distribute')
        for key,action in self.layout_actions.items():
            if key=='auto':continue
            (spacing if key in ('horizontal','vertical') else align).addAction(action)
        menu.addSeparator()
        for action in (self.copy_action,self.cut_action,self.paste_action,self.delete_action,self.undo_action,self.redo_action):menu.addAction(action)
        return menu
