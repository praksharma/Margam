"""Presentation labels independent of typed identity and numerical settings."""
from html import escape
from .block_catalogue import KINDS

TYPE_NAMES={kind:title for title,kind in KINDS.items()}


def valid_label(text):
    return isinstance(text,str) and text==text.strip() and 1<=len(text)<=256 and len(text.splitlines())==1 and all(ord(c)>=32 and ord(c)!=127 for c in text)


def literal_tooltip(text):
    return '<qt>'+escape(text).replace('\n','<br>')+'</qt>'


def block_tooltip(node,text):
    return literal_tooltip(text+'\n\nType: '+getattr(node,'type_title',TYPE_NAMES[node.kind])+
        ' ['+node.kind+']\nBlock ID: '+node.identifier)


class BlockLabels:
    def rename_block(self,node,text):
        from shiboken6 import isValid
        if not isValid(self) or getattr(self,'_labels_closing',False) or self.is_running() or self.scene.locked or not isValid(node) or node.scene() is not self.scene:return False
        if not isinstance(text,str):return False
        label=text.strip() or None
        if label is not None and not valid_label(label):
            self.statusBar().showMessage('Use a single-line block label of up to 256 characters.');return False
        previous=node.settings.get('label');old_title=node.title
        default=getattr(node,'type_title',TYPE_NAMES[node.kind])
        if previous==label and (label is not None or old_title==default):return False
        def apply(value,title):
            if value is None:node.settings.pop('label',None)
            else:node.settings['label']=value
            node.title=title
            self.refresh_descriptions()
            entry=self.output_views.entries.get(node.identifier)
            if entry:entry[0].setWindowTitle(node.title)
            body=self.property_scroll.widget()
            if node.isSelected():self.name.setText(node.title)
            if node.isSelected() and body is not None:
                from PySide6.QtWidgets import QLineEdit,QLabel
                for heading in body.findChildren(QLabel):
                    if heading.objectName()=='block_heading':heading.setText(node.title)
                for edit in body.findChildren(QLineEdit):
                    if edit.property('search_key')=='block_label':
                        blocked=edit.blockSignals(True);edit.setText(node.settings.get('label',node.title if node.kind in ('monitor','solution','comparison','export','paraview') else ''));edit.blockSignals(blocked)
            self.presentation_changed()
        self.push_edit('Rename block',lambda:apply(label,label or default),lambda:apply(previous,old_title))
        return True

    def request_block_label(self,node):
        from shiboken6 import isValid
        from PySide6.QtWidgets import QInputDialog
        if not isValid(self) or getattr(self,'_labels_closing',False) or self.is_running() or self.scene.locked or not isValid(node) or node.scene() is not self.scene:return False
        text,accepted=QInputDialog.getText(self,'Block label','Label (empty restores automatic type name):',text=node.settings.get('label',node.title))
        return self.rename_block(node,text) if accepted else False

    def label_fields(self,node):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QLineEdit,QLabel
        text=node.settings.get('label',node.title if node.kind in ('monitor','solution','comparison','export','paraview') else '')
        edit=QLineEdit(text);edit.setMaxLength(256);edit.setPlaceholderText(getattr(node,'type_title',TYPE_NAMES[node.kind]))
        edit.setProperty('search_key','block_label');edit.setProperty('help_key','block_label');last=[text]
        def finish():
            from shiboken6 import isValid
            if not isValid(self) or getattr(self,'_labels_closing',False) or not isValid(edit) or not isValid(node) or node.scene() is not self.scene:return
            if edit.text()!=last[0]:
                self.rename_block(node,edit.text());last[0]=node.settings.get('label',node.title if node.kind in ('monitor','solution','comparison','export','paraview') else '')
                edit.setText(last[0])
        edit.editingFinished.connect(finish);self.form.addRow('Block label',edit)
        kind=QLabel(getattr(node,'type_title',TYPE_NAMES[node.kind])+' ['+node.kind+']');kind.setTextFormat(Qt.TextFormat.PlainText);kind.setWordWrap(True)
        self.form.addRow('Block type',kind)
        identity=QLineEdit(node.identifier);identity.setReadOnly(True);self.form.addRow('Block ID',identity)
