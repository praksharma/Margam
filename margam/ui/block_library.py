"""Collapsible library of the blocks currently supported by the builder."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QTreeWidget, QTreeWidgetItem


BLOCK_GROUPS = {
    'Geometry': ('Rectangle',),
    'Sampling': ('Interior sampler', 'Boundary points'),
    'Neural networks': ('Neural network',),
    'Loss / Physics': ('PDE residual', 'Boundary condition'),
    'Optimisation / PINN': ('PINN method',),
    'Results': ('Solution Plot', 'Training Monitor', 'Reference', 'Comparison'),
}


class LibraryTree(QTreeWidget):
    enterPressed = Signal(object)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.currentItem():
            self.enterPressed.emit(self.currentItem())
            event.accept()
        else:
            super().keyPressEvent(event)


class BlockLibrary(QWidget):
    blockActivated = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.search = QLineEdit()
        self.search.setPlaceholderText('Search blocks…')
        self.search.setClearButtonEnabled(True)
        self.search.setAccessibleName('Search block library')
        layout.addWidget(self.search)
        self.tree = LibraryTree()
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(1)
        self.tree.setExpandsOnDoubleClick(False)
        self.tree.setAccessibleName('Block categories')
        layout.addWidget(self.tree)
        for name, blocks in BLOCK_GROUPS.items():
            group = QTreeWidgetItem(self.tree, [name])
            for title in blocks:
                item = QTreeWidgetItem(group, [title])
                item.setData(0, Qt.ItemDataRole.UserRole, title)
                item.setToolTip(0, 'Double-click or press Enter to add ' + title)
            group.setExpanded(True)
        self._expanded_before_search = None
        self.search.textChanged.connect(self.filter_blocks)
        self.tree.itemDoubleClicked.connect(self.activate_item)
        self.tree.enterPressed.connect(self.activate_item)

    def activate_item(self, item, column=0):
        title = item.data(0, Qt.ItemDataRole.UserRole)
        if title:
            self.blockActivated.emit(title)
        else:
            item.setExpanded(not item.isExpanded())

    def filter_blocks(self, text):
        query = text.strip().casefold()
        groups = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        if query and self._expanded_before_search is None:
            self._expanded_before_search = [group.isExpanded() for group in groups]
        for group in groups:
            visible = False
            for index in range(group.childCount()):
                item = group.child(index)
                match = not query or query in item.text(0).casefold() or query in group.text(0).casefold()
                item.setHidden(not match)
                visible |= match
            group.setHidden(not visible)
            if query and visible:
                group.setExpanded(True)
        if not query and self._expanded_before_search is not None:
            for group, expanded in zip(groups, self._expanded_before_search):
                group.setExpanded(expanded)
            self._expanded_before_search = None
