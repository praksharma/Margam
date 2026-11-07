"""Reversible editor operations; numerical run data and view layouts are excluded."""
from PySide6.QtGui import QUndoCommand


class EditCommand(QUndoCommand):
    def __init__(self, text, redo, undo, refresh, merge_key=None, resource_refs=None, resource_collector=None):
        super().__init__(text)
        self.apply = redo
        self.revert = undo
        self.refresh = refresh
        self.merge_key = merge_key
        self.started = False
        self.resource_refs=None if resource_refs is None else set(resource_refs)
        self.resource_collector=resource_collector

    def redo(self):
        self.apply()
        if self.resource_collector is not None:
            try:refs=self.resource_collector()
            except (ValueError,OSError):refs=None
            if refs is None:self.resource_refs=None
            elif self.resource_refs is not None:self.resource_refs.update(refs)
        if self.started:
            self.refresh()
        self.started = True

    def undo(self):
        self.revert()
        self.refresh()

    def id(self):
        return 1 if self.merge_key is not None else -1

    def mergeWith(self, other):
        if not isinstance(other, EditCommand) or self.merge_key != other.merge_key:
            return False
        self.apply = other.apply
        if self.resource_refs is None or other.resource_refs is None:self.resource_refs=None
        else:self.resource_refs.update(other.resource_refs)
        return True
