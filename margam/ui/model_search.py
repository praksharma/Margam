"""Find existing instances without inserting blocks or changing model settings."""
from html import escape
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QLineEdit,QListWidget,QListWidgetItem,QPushButton,QLabel


class ModelSearch(QDialog):
    def __init__(self,window):
        super().__init__(window)
        self.window=window
        self.setWindowTitle('Find in model');self.resize(540,400)
        layout=QVBoxLayout(self)
        layout.addWidget(QLabel('Search existing blocks by name, type, displayed settings or ID. Refresh after editing.'))
        self.search=QLineEdit();self.search.setPlaceholderText('Find a block…');self.search.setClearButtonEnabled(True)
        layout.addWidget(self.search)
        self.results=QListWidget();layout.addWidget(self.results)
        self.status=QLabel();self.status.setTextFormat(Qt.TextFormat.PlainText);layout.addWidget(self.status)
        row=QHBoxLayout();layout.addLayout(row)
        refresh=QPushButton('Refresh');refresh.clicked.connect(self.refresh);row.addWidget(refresh)
        locate=QPushButton('Select block');locate.clicked.connect(self.locate);row.addWidget(locate)
        close=QPushButton('Close');close.clicked.connect(self.close);row.addWidget(close)
        self.search.textChanged.connect(self.refresh)
        self.search.returnPressed.connect(self.locate)
        self.results.itemActivated.connect(lambda item:self.locate())

    def refresh(self):
        query=self.search.text().strip().casefold()
        self.results.clear()
        for node in sorted(self.window.scene.nodes(),key=lambda n:(n.title,n.identifier)):
            text=f'{node.title} · {node.detail} · {node.kind} · {node.identifier}'
            if query and query not in text.casefold():continue
            item=QListWidgetItem(f'{node.title} · {node.detail} [{node.identifier[:8]}]')
            item.setData(Qt.ItemDataRole.UserRole,node.identifier)
            item.setToolTip("<qt>"+escape(text).replace("\n","<br>")+"</qt>");self.results.addItem(item)
        if self.results.count():self.results.setCurrentRow(0)
        self.status.setText(f'{self.results.count()} matching block(s)')

    def locate(self):
        item=self.results.currentItem()
        if item is None:return
        identifier=item.data(Qt.ItemDataRole.UserRole)
        node=next((n for n in self.window.scene.nodes() if n.identifier==identifier),None)
        if node is None:
            self.refresh();self.status.setText('Block no longer exists. Results refreshed.');return
        self.window.scene.clearSelection();node.setSelected(True)
        self.window.view.centerOn(node)
        self.status.setText('Selected '+node.title)
