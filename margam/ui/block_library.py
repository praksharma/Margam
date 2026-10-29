"""Collapsible library of the blocks currently supported by the builder."""
from PySide6.QtCore import Qt, Signal, QMimeData, QEvent
from PySide6.QtGui import QDrag
from .icons import icon
from .setting_help import show_information
from ..components.network_config import NETWORK_NAMES
from .block_catalogue import BLOCK_GROUPS, BLOCK_MIME, KINDS, describe_block
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QTreeWidget, QTreeWidgetItem, QDialog, QLabel, QHBoxLayout, QToolButton


class LibraryTree(QTreeWidget):
    enterPressed = Signal(object)

    def mimeTypes(self):
        return [BLOCK_MIME]

    def mimeData(self, items):
        mime = QMimeData()
        if len(items) == 1:
            title = items[0].data(0, Qt.ItemDataRole.UserRole)
            if title:
                mime.setData(BLOCK_MIME, title.encode('utf-8'))
        return mime

    def startDrag(self, supported_actions):
        mime = self.mimeData(self.selectedItems())
        if not self.isEnabled() or not mime.hasFormat(BLOCK_MIME):
            return
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)

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
        self.tree.setDragEnabled(True)
        self.tree.setDragDropMode(QTreeWidget.DragDropMode.DragOnly)
        self.tree.setExpandsOnDoubleClick(False)
        self.tree.setAccessibleName('Block categories')
        layout.addWidget(self.tree)
        self.description=QLabel('Select a block to see supported settings and limits.');self.description.setWordWrap(True)
        row=QHBoxLayout();row.addWidget(self.description,1)
        self.info_button=QToolButton();self.info_button.setText('i');self.info_button.setAutoRaise(True);self.info_button.setAccessibleName('Explain selected library block');self.info_button.setToolTip('Show full block description');self.info_button.setEnabled(False)
        self.info_button.clicked.connect(self.show_description);row.addWidget(self.info_button,0,Qt.AlignmentFlag.AlignTop);layout.addLayout(row)
        self.tree.currentItemChanged.connect(self.describe_item)
        for name, blocks in BLOCK_GROUPS.items():
            group = QTreeWidgetItem(self.tree, [name])
            group.setIcon(0,icon(KINDS[blocks[0]]))
            group.setFlags(group.flags() & ~Qt.ItemFlag.ItemIsDragEnabled)
            for title in blocks:
                item = QTreeWidgetItem(group, [title])
                item.setIcon(0,icon(KINDS[title]))
                item.setData(0, Qt.ItemDataRole.UserRole, title)
                item.setToolTip(0, 'Drag, double-click or press Enter to add '+title+'\n'+describe_block(KINDS[title]))
            group.setExpanded(True)
        self._expanded_before_search = None
        self.search.textChanged.connect(self.filter_blocks)
        self.tree.itemDoubleClicked.connect(self.activate_item)
        self.tree.enterPressed.connect(self.activate_item)

    def describe_item(self,item,previous=None):
        title=item.data(0,Qt.ItemDataRole.UserRole) if item else None
        self.description.setText(describe_block(KINDS[title]) if title else 'Choose a supported component from this category.')
        self.info_button.setEnabled(bool(title))

    def show_description(self):
        item=self.tree.currentItem();title=item.data(0,Qt.ItemDataRole.UserRole) if item else None
        if title:return show_information(self.window(),title,describe_block(KINDS[title]))

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
                if item.data(0,Qt.ItemDataRole.UserRole)=='Neural network':match |= query in ' '.join(NETWORK_NAMES.values()).casefold()
                item.setHidden(not match)
                visible |= match
            group.setHidden(not visible)
            if query and visible:
                group.setExpanded(True)
        if not query and self._expanded_before_search is not None:
            for group, expanded in zip(groups, self._expanded_before_search):
                group.setExpanded(expanded)
            self._expanded_before_search = None


class QuickInsert(QDialog):
    """Small canvas popup using the same supported catalogue as the library."""
    blockChosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle('Add block')
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Add block · search, then Enter · Esc to cancel'))
        self.library = BlockLibrary()
        # Drag insertion belongs to the dock, not a popup that owns focus/grabs.
        self.library.tree.setDragEnabled(False)
        layout.addWidget(self.library)
        self.resize(350, 430)
        self.library.blockActivated.connect(self.choose)
        self.library.search.returnPressed.connect(self.choose_first)
        self.library.search.installEventFilter(self)

    def first_match(self):
        tree = self.library.tree
        for index in range(tree.topLevelItemCount()):
            group = tree.topLevelItem(index)
            if group.isHidden():
                continue
            for child in range(group.childCount()):
                item = group.child(child)
                if not item.isHidden():
                    return item
        return None

    def choose(self, title):
        self.blockChosen.emit(title)
        self.accept()

    def choose_first(self):
        item = self.first_match()
        if item is not None:
            self.choose(item.data(0, Qt.ItemDataRole.UserRole))

    def eventFilter(self, watched, event):
        if watched is self.library.search and event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Down:
            item = self.first_match()
            if item is not None:
                self.library.tree.setCurrentItem(item)
                self.library.tree.setFocus()
            return True
        return super().eventFilter(watched, event)
