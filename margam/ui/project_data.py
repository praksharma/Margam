"""Read-only inventory of the concrete file mappings held by desktop blocks."""
from copy import deepcopy
import json
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from shiboken6 import isValid
from PySide6.QtWidgets import (QApplication,QDialog,QVBoxLayout,QHBoxLayout,QLabel,
    QTableWidget,QTableWidgetItem,QHeaderView,QPushButton,QTextBrowser)
from .graph_model import overlay,role
from .source_fingerprints import source_slot
from ..components.prepared_data import LIMIT,read_points
from ..components.evaluation_data import read_point_data


def collect_sources(window):
    """Read instance settings, including disconnected/unconfigured source blocks."""
    rows=[]
    base=deepcopy(window.config)
    domains=[node for node in window.scene.nodes() if role(node.kind)=='domain']
    pdes=[node for node in window.scene.nodes() if role(node.kind)=='pde']
    for candidates in (domains,pdes):
        if len(candidates)==1:
            node=candidates[0]
            try:base=overlay(base,node.kind,node.settings['parameters'])
            except (KeyError,ValueError):pass
    incoming={(wire.target.parentItem().identifier,wire.target.name):wire.source.parentItem() for wire in window.scene.wires}
    for node in sorted(window.scene.nodes(),key=lambda node:(node.title,node.identifier)):
        config=deepcopy(base)
        parameters=node.settings.get('parameters',{})
        geometry=incoming.get((node.identifier,'input'))
        if geometry is not None and role(geometry.kind)=='domain':
            try:config=overlay(config,geometry.kind,geometry.settings['parameters'])
            except (KeyError,ValueError):pass
        expected=geometry.settings.get('jorvik',{}).get('sha256') if geometry is not None and geometry.kind=='jorvik_geometry' else None
        def add(role,source,reader='points',boundary=None):
            logical=window.project_workspace.portable_document(deepcopy(source))
            # Only an exact mapping binding can supply a persisted import baseline.
            slot=source_slot(node.kind,boundary)
            stored=node.settings.get('data_fingerprints',{}).get(slot,{})
            imported=stored.get('sha256') if window.project_workspace.portable_document(stored.get('source',{}))==logical else None
            baseline_kind='Jorvik' if expected else 'Import' if imported else 'Session'
            rows.append(dict(node_id=node.identifier,block=node.title,role=role,boundary=boundary,
                             source=logical,reader=reader,config=deepcopy(config),expected=expected or imported,baseline_kind=baseline_kind,
                             geometry_id=geometry.identifier if geometry is not None and geometry.kind=='jorvik_geometry' else None,
                             key=(node.identifier,role,json.dumps(logical,sort_keys=True))))
        if node.kind in ('interior','interior_data','rar','importance','observations'):
            source=parameters.get('interior_data',{})
            if source or node.kind in ('interior_data','observations'):add('Observations' if node.kind=='observations' else 'Interior',source)
        if node.kind in ('boundary','boundary_data'):
            mappings=parameters.get('boundary_data',{})
            for name,source in mappings.items():add('Boundary · '+name,source,boundary=name)
            if not mappings and node.kind=='boundary_data':add('Boundary',{})
        if node.kind in ('evaluation_points','reference_data'):
            add('Reference' if node.kind=='reference_data' else 'Evaluation',node.settings.get('source_data',{}),node.kind)
        if node.kind=='reference' and node.settings.get('reference_data'):
            add('Reference',node.settings['reference_data'])
    return rows


def file_digest(path,cache):
    from ..components.prepared_data import file_fingerprint
    key=str(path.absolute())
    if key not in cache:cache[key]=file_fingerprint(path)
    return cache[key]


def inspect_source(record,workspace,baselines,cache):
    source=record['source'];config=record['config']
    result=dict(status='Not mapped',rows=None,storage='—',location='',error='',changed=False)
    if not source:return result
    result['storage']='Project copy' if source['path'].startswith('margam-owned:') else 'Linked file'
    result['location']=source['path'].removeprefix('margam-owned:')
    try:
        resolved=workspace.resolve_source(source,create=False);path=Path(resolved['path'])
        digest,signature=file_digest(path,cache)
        first=baselines.setdefault(record['key'],digest)
        result['changed']=digest!=(record['expected'] or first)
        result['status']=('Changed from applied Jorvik snapshot' if record.get('baseline_kind')=='Jorvik' else 'Changed since import' if record.get('baseline_kind')=='Import' else 'Changed since first check') if result['changed'] else 'Available'
        if record['reader'] in ('evaluation_points','reference_data'):
            data=read_point_data(resolved,config,reference=record['reader']=='reference_data');result['rows']=len(data.points)
        else:
            points,normals,values=read_points(resolved,len(config.bounds[0]),len(config.field_names) if record['role']=='Reference' else 1)
            if record['role'] in ('Reference','Observations') and values is None:raise ValueError('Reference values are not mapped.')
            result['rows']=len(points)
        after=path.stat()
        if signature!=(after.st_size,after.st_mtime_ns):raise ValueError('File changed during the check; refresh to retry.')
    except FileNotFoundError as error:result.update(status='Missing',error=str(error))
    except PermissionError as error:result.update(status='Unreadable',error=str(error))
    except Exception as error:result.update(status='Invalid / unavailable',error=str(error))
    return result


class ProjectDataDialog(QDialog):
    def __init__(self,window):
        super().__init__(window);self.window=window;self.workspace=window.project_workspace
        self.baselines={};self.records=[];self.results=[];self.stale=False
        self.setWindowTitle('Project data overview');self.setObjectName('project_data_overview');self.resize(1120,670)
        layout=QVBoxLayout(self)
        note=QLabel('Read-only file and mapping checks, not full model validation. Refresh to recheck. New imports use a saved fingerprint. Older mappings without one use a session-only first check. Jorvik uses its applied snapshot fingerprint.')
        note.setWordWrap(True);layout.addWidget(note)
        self.summary=QLabel();self.summary.setWordWrap(True);layout.addWidget(self.summary)
        self.table=QTableWidget(0,6);self.table.setObjectName('project_data_table')
        self.table.setHorizontalHeaderLabels(['Block','Role / set','Storage','Rows','Status','File'])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True);layout.addWidget(self.table,2)
        self.details=QTextBrowser();self.details.setObjectName('project_data_details');layout.addWidget(self.details,1)
        buttons=QHBoxLayout();self.refresh_button=QPushButton('Refresh')
        self.select_button=QPushButton('Select block');self.map_button=QPushButton('Map data…')
        close=QPushButton('Close')
        for button in (self.refresh_button,self.select_button,self.map_button):buttons.addWidget(button)
        buttons.addStretch();buttons.addWidget(close);layout.addLayout(buttons)
        self.refresh_button.clicked.connect(self.refresh);self.select_button.clicked.connect(self.select_block)
        self.map_button.clicked.connect(self.map_source);close.clicked.connect(self.close)
        self.table.itemSelectionChanged.connect(self.show_details)
        self.table.cellDoubleClicked.connect(lambda *args:self.select_block())
        self.window.scene.graphChanged.connect(self.mark_stale)
        self.window.undo_stack.indexChanged.connect(self.mark_stale)
        self.refresh()

    def detach(self):
        if getattr(self,'detached',False):return
        self.detached=True
        self.window.scene.graphChanged.disconnect(self.mark_stale)
        self.window.undo_stack.indexChanged.disconnect(self.mark_stale)

    def mark_stale(self,*args):
        if not isValid(self) or not isValid(self.window.scene):return
        self.stale=True
        if self.isVisible():self.summary.setText('The graph/settings changed. Refresh to inspect the current sources.')
        self.update_buttons()

    def current(self):
        index=self.table.currentRow()
        return self.records[index] if 0<=index<len(self.records) else None

    def owning_node(self,record):
        if not isValid(self.window.scene):return None
        return next((node for node in self.window.scene.nodes() if record and node.identifier==record['node_id']),None)

    def update_buttons(self):
        record=self.current();node=self.owning_node(record)
        self.select_button.setEnabled(node is not None)
        self.map_button.setEnabled(node is not None and not self.stale and not self.window.is_running() and not self.window.scene.locked)
        jorvik=record is not None and record['geometry_id'] and record['source'].get('path','').lower().endswith('.nc')
        self.map_button.setText('Review Jorvik datasets…' if jorvik else 'Map data…')

    def refresh(self):
        if self.workspace is not self.window.project_workspace:
            self.baselines.clear();self.workspace=self.window.project_workspace
        selected=self.current();key=selected['key'] if selected else None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.stale=False
            self.records=collect_sources(self.window);cache={}
            self.results=[inspect_source(record,self.workspace,self.baselines,cache) for record in self.records]
            self.table.setRowCount(len(self.records))
            for index,(record,result) in enumerate(zip(self.records,self.results)):
                cells=[record['block'],record['role'],result['storage'],str(result['rows']) if result['rows'] is not None else '—',result['status'],Path(record['source'].get('path','')).name if record['source'] else '—']
                for column,text in enumerate(cells):
                    item=QTableWidgetItem(text);item.setToolTip(text);self.table.setItem(index,column,item)
                    if result['error'] or result['changed'] or not record['source']:item.setForeground(QColor('#be3e38'))
            warning=sum(bool(result['error'] or result['changed'] or not record['source']) for record,result in zip(self.records,self.results))
            self.summary.setText(f'{len(self.records)} data source entries · {warning} need attention. Built-in generated samples and geometry documents are not file datasets.' if self.records else 'No imported datasets. Built-in samplers generate their own points.')
            index=next((i for i,record in enumerate(self.records) if record['key']==key),0)
            if self.records:self.table.selectRow(index)
            else:self.details.clear()
            self.show_details()
        finally:QApplication.restoreOverrideCursor()

    def show_details(self):
        record=self.current()
        if record is None:self.update_buttons();return
        result=self.results[self.table.currentRow()];source=record['source']
        lines=[record['block']+' · '+record['role'],result['status'],result['storage']+': '+result['location']]
        for key,label in [('coordinates','Coordinates'),('normals','Normals'),('values','Values'),('ids','Point IDs')]:
            if key in source:lines.append(label+': '+(source[key] or 'Not mapped'))
        if source.get('values'):lines.append('Fields: '+', '.join(record['config'].field_names))
        lines.append('Baseline: '+{'Import':'accepted import (saved with project)','Jorvik':'applied Jorvik snapshot (saved with project)','Session':'first check in this session; no saved import fingerprint'}.get(record.get('baseline_kind'),'first check in this session'))
        if result['changed']:lines.append('File bytes differ. No data, mapping, checkpoint or baseline was replaced by this check.')
        if result['error']:lines.append(result['error'])
        lines.append('Use Validate for graph compatibility, BC operators and reference alignment.')
        self.details.setPlainText('\n'.join(lines));self.update_buttons()

    def select_block(self):
        record=self.current();node=self.owning_node(record)
        if node is None:self.refresh();return None
        if record['boundary']:self.window.boundary_side=record['boundary']
        self.window.show_block_properties(node);self.window.view.centerOn(node)
        return node

    def map_source(self):
        if self.window.is_running() or self.window.scene.locked:return
        if self.stale:self.refresh();return
        record=self.current();node=self.select_block()
        if node is None:return
        if record['geometry_id'] and record['source'].get('path','').lower().endswith('.nc'):
            geometry=next((n for n in self.window.scene.nodes() if n.identifier==record['geometry_id']),None)
            if geometry is not None:self.window.update_jorvik(geometry)
        elif record['reader'] in ('evaluation_points','reference_data'):self.window.edit_evaluation_data(node)
        else:self.window.edit_point_data(record['boundary'],reference=node if record['role']=='Reference' else None)
        self.refresh()
