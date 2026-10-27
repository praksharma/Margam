"""Reversible editor operations; numerical run data and view layouts are excluded."""
from PySide6.QtGui import QUndoCommand


class EditCommand(QUndoCommand):
    def __init__(self, text, redo, undo, refresh, merge_key=None):
        super().__init__(text)
        self.apply = redo
        self.revert = undo
        self.refresh = refresh
        self.merge_key = merge_key
        self.started = False

    def redo(self):
        self.apply()
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
        return True
