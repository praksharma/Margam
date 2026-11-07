"""Native explicit review/removal of unreferenced managed dataset snapshots."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QLabel,QTableWidget,QTableWidgetItem,QHeaderView,QPushButton
from .dataset_cleanup import protected_references,scan_snapshots,remove_snapshots


class DatasetCleanupDialog(QDialog):
    def __init__(self,window):
        super().__init__(window);self.window=window;self.workspace=None;self.rows=[]
        self.setWindowTitle('Clean up unused imports');self.resize(920,550)
        layout=QVBoxLayout(self)
        note=QLabel('Review project-owned CSV/NPZ snapshots. Current/saved mappings, cached runs and Undo/Redo are kept. Select unused rows to remove them permanently; removal cannot be undone. Linked originals, geometry and other files are not removed.')
        note.setWordWrap(True);note.setTextFormat(Qt.TextFormat.PlainText);layout.addWidget(note)
        self.status=QLabel();self.status.setWordWrap(True);self.status.setTextFormat(Qt.TextFormat.PlainText);layout.addWidget(self.status)
        self.table=QTableWidget(0,4);self.table.setHorizontalHeaderLabels(['Remove','Project-relative file','Bytes','Kept for / status'])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(self.update_buttons);layout.addWidget(self.table)
        row=QHBoxLayout();layout.addLayout(row)
        refresh=QPushButton('Refresh');refresh.clicked.connect(self.refresh);row.addWidget(refresh)
        select=QPushButton('Select unused');select.clicked.connect(self.select_unused);row.addWidget(select)
        self.remove_button=QPushButton('Remove selected snapshots');self.remove_button.clicked.connect(self.remove_selected);row.addWidget(self.remove_button)
        row.addStretch();close=QPushButton('Close');close.clicked.connect(self.close);row.addWidget(close)

    def busy(self):return self.window.is_running() or self.window.scene.locked or self.window.project_workspace.busy

    def selected(self):
        return [row for index,row in enumerate(self.rows) if self.table.item(index,0).checkState()==Qt.CheckState.Checked]

    def update_buttons(self,*args):
        self.remove_button.setEnabled(not self.busy() and self.workspace is self.window.project_workspace and bool(self.selected()))

    def refresh(self):
        self.workspace=self.window.project_workspace;self.rows=[]
        self.table.blockSignals(True);self.table.setRowCount(0)
        try:
            self.rows=scan_snapshots(self.workspace,protected_references(self.window))
            self.table.setRowCount(len(self.rows))
            for index,row in enumerate(self.rows):
                check=QTableWidgetItem();check.setCheckState(Qt.CheckState.Unchecked)
                if row.reason=='Unused':check.setFlags(Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsUserCheckable)
                else:check.setFlags(Qt.ItemFlag.NoItemFlags)
                self.table.setItem(index,0,check)
                for column,value in enumerate((row.relative,str(row.size),row.reason),1):self.table.setItem(index,column,QTableWidgetItem(value))
            unused=[row for row in self.rows if row.reason=='Unused']
            self.status.setText(f'{len(self.rows)} recognised snapshot(s); {len(unused)} unused ({sum(row.size for row in unused)} bytes). Other folder layouts and symbolic/hard links are skipped. Refresh after edits.')
        except (OSError,ValueError,TypeError,KeyError) as error:
            self.rows=[];self.table.setRowCount(0);self.status.setText('Cleanup unavailable: '+str(error))
        finally:self.table.blockSignals(False)
        self.update_buttons()

    def select_unused(self):
        for index,row in enumerate(self.rows):
            if row.reason=='Unused':self.table.item(index,0).setCheckState(Qt.CheckState.Checked)
        self.update_buttons()

    def remove_selected(self):
        if self.busy():self.status.setText('Wait for training or the geometry editor to finish.');self.update_buttons();return False
        if self.workspace is not self.window.project_workspace:
            self.status.setText('The project changed. Refresh before cleanup.');self.update_buttons();return False
        selected=self.selected()
        if not selected:return False
        try:removed,kept=remove_snapshots(self.workspace,selected,protected_references(self.window))
        except (OSError,ValueError,TypeError,KeyError) as error:
            self.status.setText('No further cleanup: '+str(error));return False
        self.refresh()
        self.status.setText(f'Removed {len(removed)} snapshot(s); kept {len(kept)} changed, protected or unavailable snapshot(s). Removal is permanent.')
        return bool(removed)
