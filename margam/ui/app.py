"""Block-based baseline Poisson workflow with background training."""
from uuid import uuid4
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
from ..recipes import baseline, self_adaptive, learning_rate_annealing, COMPACT_RECIPES, TRAINER_RECIPES
from ..templates.poisson import PoissonConfig, make_poisson_reference
from ..templates.predefined_pdes import PREDEFINED_PDES
from ..templates.examples import EXAMPLE_PROBLEMS,example_config
from ..templates.heat import heat_demo_config
from ..templates.incompressible_flow import flow_demo_config
from ..components.outputs import OutputRequest, ReferenceSource
from ..components.expressions import Expression
from ..components.prepared_data import reference_values
from ..components.export import export_csv
from ..components.sampling import SAMPLERS
from ..components.network_config import NETWORK_NAMES, OPTION_SCHEMAS, network_options
from .plots import RectanglePreview
from .output_views import OutputViews, OUTPUT_KINDS
from .training import TrainingWorker
from .canvas_view import CanvasView
from .block_library import BlockLibrary, QuickInsert
from .block_catalogue import KINDS, describe_block
from .ports import INPUTS, OUTPUTS
from .edit_history import EditCommand
from .desktop_state import DesktopState
from .data_mapping import DataMappingDialog
from .graph_model import role, PARAMETERS, OWNERS, parameters_for, overlay, assemble, encode_parameter
from .run_document import capture_run, save_run, load_run
from .model_document import capture_model, decode_document, read_model, write_model, unique_keys, MAX_BYTES
from .theme import colors, configure_theme, theme_name
from .icons import icon
from PySide6.QtCore import Qt, QPointF, QTimer, Signal, Slot, QMimeData, QSize, QUrl
from shiboken6 import isValid
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QActionGroup, QUndoStack, QAction, QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QGraphicsView,
    QGraphicsScene, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPathItem,
    QGraphicsItem, QWidget, QVBoxLayout, QLabel, QFormLayout, QLineEdit,
    QMenu, QPlainTextEdit, QScrollArea, QComboBox, QDoubleSpinBox,
    QSpinBox, QPushButton, QProgressBar, QHBoxLayout, QCheckBox, QMessageBox, QFileDialog, QInputDialog,
)


class Port(QGraphicsEllipseItem):
    def __init__(self, node, output, spec, y=40):
        super().__init__(-6, -6, 12, 12, node)
        self.output = output
        self.name = spec.name
        self.data_type = spec.data_type
        self.label = spec.label
        self.setPos(170 if output else 0, y)
        self.setBrush(QColor(colors()['highlight']))
        self.setPen(QPen(QColor(colors()['base']), 2))
        self.setToolTip(('Output: ' if output else 'Input: ') + spec.label + '\n' + spec.data_type.replace('_',' ') + (' · drag to a compatible input' if output else ' · accepts one source'))
        self.setZValue(2)

    def mousePressEvent(self, event):
        if self.output and not self.scene().locked:
            self.scene().begin_wire(self)
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if self.scene().pending:
            self.scene().pending.update_path(event.scenePos())

    def mouseReleaseEvent(self, event):
        self.scene().finish_wire(event.scenePos())


class Wire(QGraphicsPathItem):
    def __init__(self, source, target=None):
        super().__init__()
        self.source, self.target = source, target
        self.setPen(QPen(QColor(colors()['highlight']), 2))
        self.setZValue(-1)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.update_path()

    def update_path(self, end=None):
        a = self.source.scenePos()
        b = self.target.scenePos() if self.target else (end or a)
        bend = max(60, abs(b.x() - a.x()) / 2)
        path = QPainterPath(a)
        path.cubicTo(a + QPointF(bend, 0), b - QPointF(bend, 0), b)
        self.setPath(path)


REQUIRED_EDGES = {("domain","interior"),("domain","boundary"),
                  ("interior","pde"),("boundary","bc"),("pde","method"),
                  ("bc","method"),("network","method")}

CORE_KINDS = {"domain", "interior", "boundary", "pde", "bc", "network", "method"}


class Node(QGraphicsRectItem):
    def __init__(self, title, detail, x, y):
        super().__init__(0, 0, 170, 80)
        self.title, self.detail = title, detail
        self.kind = KINDS[title]
        self.identifier = uuid4().hex
        self.settings = {"resolution": 101, "history_limit": 10000}
        if self.kind in PARAMETERS:self.settings["parameters"]=parameters_for(self.kind)
        self.setBrush(QColor(colors()['base']))
        self.setPen(QPen(QColor(colors()['border']), 1.5))
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
                      QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
                      QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        specs=INPUTS[self.kind]
        self.inputs={spec.name:Port(self,False,spec,80*(index+1)/(len(specs)+1))
                     for index,spec in enumerate(specs)}
        self.input=next(iter(self.inputs.values()),None)
        spec=OUTPUTS[self.kind]
        self.output=Port(self,True,spec) if spec else None
        self.setPos(x, y)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        painter.setPen(QColor(colors()['text']))
        painter.drawText(14, 28, painter.fontMetrics().elidedText(self.title,Qt.TextElideMode.ElideRight,145))
        painter.setPen(QColor(colors()['muted']))
        painter.drawText(14, 55, painter.fontMetrics().elidedText(self.detail, Qt.TextElideMode.ElideRight, 145))

    def contextMenuEvent(self, event):
        self.scene().clearSelection()
        self.setSelected(True)
        self.scene().blockContextRequested.emit(self, event.screenPos())
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.scene().clearSelection()
        self.setSelected(True)
        self.scene().blockOpened.emit(self)
        event.accept()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and self.scene():
            for wire in self.scene().wires:
                wire.update_path()
        return super().itemChange(change, value)


class Canvas(QGraphicsScene):
    graphChanged = Signal()
    blockOpened = Signal(object)
    blockContextRequested = Signal(object, object)
    connectionRejected = Signal(str)

    def __init__(self):
        super().__init__()
        self.wires, self.pending = [], None
        self.locked = False
        self.edit_handler = None
        self._move_start = {}
        self.setSceneRect(-5000, -5000, 10000, 10000)
        self.setBackgroundBrush(QColor(colors()['canvas']))

    def edit(self,text,redo,undo):
        if self.edit_handler is None:redo()
        else:self.edit_handler(text,redo,undo)

    def mousePressEvent(self,event):
        self._move_start={node:QPointF(node.pos()) for node in self.nodes()} if event.button()==Qt.MouseButton.LeftButton and not self.locked else {}
        super().mousePressEvent(event)

    def mouseReleaseEvent(self,event):
        super().mouseReleaseEvent(event)
        moves=[(node,old,QPointF(node.pos())) for node,old in self._move_start.items()
               if node.scene() is self and node.pos()!=old]
        self._move_start={}
        if moves and not self.locked:
            def place(use_new):
                for node,old,new in moves:node.setPos(new if use_new else old)
            self.edit('Move blocks',lambda:place(True),lambda:place(False))

    def cancel_wire(self):
        if self.pending is not None:
            self.removeItem(self.pending)
            self.pending=None

    def begin_wire(self, source):
        if self.locked or source is None or not source.output or source.scene() is not self:
            return
        self.cancel_wire()
        self.pending = Wire(source)
        self.addItem(self.pending)

    def connection_error(self, source, target, ignore=None):
        if source is None or target is None:
            return 'Choose an output and a compatible input.'
        if source.scene() is not self or target.scene() is not self:
            return 'Both ports must belong to the current model.'
        if not source.output or target.output:
            return 'Connect an output to an input.'
        if source.parentItem() is target.parentItem():
            return 'A block cannot connect to itself.'
        if source.data_type != target.data_type:
            return f'{source.label} cannot feed {target.label}. Choose a compatible input.'
        for wire in self.wires:
            if wire is ignore:continue
            if wire.target is target:
                return f'{target.label} already has a source. Delete its wire before reconnecting.'
        return None

    def connect_ports(self, source, target):
        if self.locked:raise ValueError('Connections are locked during training.')
        error=self.connection_error(source,target)
        if error:raise ValueError(error)
        wire=Wire(source,target)
        def add():
            self.addItem(wire);self.wires.append(wire);wire.update_path()
            self.graphChanged.emit()
        def remove():
            self.wires.remove(wire);self.removeItem(wire)
            self.graphChanged.emit()
        self.edit('Connect blocks',add,remove)
        return wire

    def finish_wire(self, position):
        if self.pending is None:return
        source=self.pending.source
        targets=[item for item in self.items(position) if isinstance(item,Port) and not item.output]
        self.cancel_wire()
        if self.locked or not targets:return
        try:self.connect_ports(source,targets[0])
        except ValueError as error:self.connectionRejected.emit(str(error))

    def connect_nodes(self, source, target, input_name=None):
        if input_name is None and source.output is not None:
            matches=[port for port in target.inputs.values() if port.data_type==source.output.data_type]
            if len(matches)==1:input_name=matches[0].name
        port=target.inputs.get(input_name)
        return self.connect_ports(source.output,port)

    def nodes(self):
        return [item for item in self.items() if isinstance(item, Node)]

    def validate_graph(self):
        nodes = self.nodes()
        compact=any(n.kind in COMPACT_RECIPES for n in nodes)
        expected=CORE_KINDS-{'network'} if compact else CORE_KINDS
        required=REQUIRED_EDGES-{('network','method')} if compact else REQUIRED_EDGES
        if compact and any(n.kind=='network' for n in nodes):raise ValueError('A compact recipe owns its network. Expand it before replacing the network.')
        for kind in expected:
            if sum(role(n.kind) == kind for n in nodes) != 1:
                raise ValueError(f'The graph needs exactly one {kind} block.')
        core_edges = set()
        destinations = set()
        for wire in self.wires:
            error=self.connection_error(wire.source,wire.target,ignore=wire)
            if error:raise ValueError(error)
            source,target = wire.source.parentItem(),wire.target.parentItem()
            edge = (role(source.kind),role(target.kind))
            if role(source.kind) in CORE_KINDS and role(target.kind) in CORE_KINDS:
                if edge not in required or edge in core_edges:
                    raise ValueError('Unsupported or duplicate core connection.')
                core_edges.add(edge)
            else:
                allowed = role(source.kind) == 'method' and target.kind in OUTPUT_KINDS
                allowed = allowed and wire.target.name == ('model' if target.kind=='comparison' else 'input')
                allowed = allowed or (source.kind=='reference' and target.kind=='comparison' and wire.target.name=='reference')
                if not allowed:raise ValueError('Unsupported output connection: '+source.title+' → '+target.title)
                key=(target.identifier,wire.target.name)
                if key in destinations:raise ValueError('Each output input accepts only one source.')
                destinations.add(key)
        missing=required-core_edges
        if missing:
            raise ValueError('Missing connections: '+', '.join(f'{a} → {b}' for a,b in sorted(missing)))

    def compile_config(self):
        self.validate_graph()
        return assemble(self.nodes(),self.wires)

    def output_requests(self, config):
        self.validate_graph()
        requests=[]
        for node in self.nodes():
            if node.kind not in OUTPUT_KINDS:continue
            incoming=[w for w in self.wires if w.target.parentItem() is node]
            if not incoming:continue
            reference=None
            if node.kind=='comparison':
                if {w.target.name for w in incoming} != {'model','reference'}:
                    raise ValueError('Comparison needs both a trained-model connection and a Reference connection.')
                source=next(w.source.parentItem() for w in incoming if w.target.name=='reference')
                text=source.settings.get('reference_expression','')
                data=source.settings.get('reference_data',{})
                if text and data:raise ValueError('Choose an expression or imported reference data, not both.')
                if text:
                    texts=[part.strip() for part in text.split(';')]
                    if len(texts)!=len(config.field_names):raise ValueError('Provide one reference expression per field, separated by semicolons.')
                    expressions=[Expression(part,len(config.bounds[0]),target=True,fields=config.field_names) for part in texts]
                    def evaluate(points,expressions=expressions):
                        import torch
                        values=torch.cat([expr(torch.tensor(points),source=config.source,viscosity=config.viscosity) for expr in expressions],dim=1)
                        return values[:,0].tolist() if len(expressions)==1 else values.tolist()
                    reference=ReferenceSource('User reference expression',evaluate)
                elif data:
                    from functools import partial
                    reference=ReferenceSource('Imported reference fields',partial(reference_values,deepcopy(data),components=len(config.field_names)))
                else:reference=make_poisson_reference(config)
            if node.kind=='monitor':
                if not 25 <= node.settings['history_limit'] <= 100000:
                    raise ValueError('Monitor history limit must be between 25 and 100000.')
            elif not 2 <= node.settings['resolution'] <= 201:
                raise ValueError('Evaluation resolution must be between 2 and 201.')
            requests.append(OutputRequest(node.identifier,node.kind,node.settings['resolution'],reference))
        return requests

    def delete_selection(self,text='Delete selection'):
        if self.locked:return
        self.cancel_wire()
        selected = set(self.selectedItems())
        nodes=[item for item in selected if isinstance(item,Node)]
        wires=[wire for wire in self.wires if wire in selected or
               wire.source.parentItem() in selected or wire.target.parentItem() in selected]
        if not nodes and not wires:return
        def remove():
            self.cancel_wire()
            for wire in wires:self.wires.remove(wire);self.removeItem(wire)
            for node in nodes:self.removeItem(node)
            self.graphChanged.emit()
        def restore():
            self.clearSelection()
            for node in nodes:self.addItem(node);node.setSelected(True)
            for wire in wires:
                self.addItem(wire);self.wires.append(wire);wire.update_path()
                wire.setSelected(wire in selected)
            self.graphChanged.emit()
        self.edit(text,remove,restore)



class Window(QMainWindow):
    def __init__(self,state_directory=None):
        super().__init__()
        self.undo_stack=QUndoStack(self)
        self.undo_stack.setUndoLimit(100)
        self._resetting=False
        self._history_replaying=False
        self._history_epoch=0
        self.config = PoissonConfig()
        self.worker = None
        self.loaded_checkpoint = None
        self.result = None
        self.monitor_records = {}
        self.active_monitors = set()
        self.last_run = None
        self.rectangle_previews = {}
        self.quick_insert = None
        self._closing = False
        self.model_path = None
        self._paste_serial = 0
        self.setWindowTitle('Margam · baseline PINN builder')
        self.resize(1480,950)
        configure_theme(QApplication.instance(),'light')
        toolbar=self.addToolBar('Model');self.model_toolbar=toolbar
        toolbar.setIconSize(QSize(20,20));toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.load_action=QAction('2D Poisson — rectangle (reset)',self)
        self.load_action.triggered.connect(lambda:self.request_template(False))
        self.heat_action=QAction('2D steady heat conduction',self)
        self.heat_action.triggered.connect(lambda:self.request_template(True))
        self.zoom_in_action=toolbar.addAction('Zoom in')
        self.zoom_in_action.triggered.connect(lambda:self.view.zoom_by(1.2))
        self.zoom_out_action=toolbar.addAction('Zoom out')
        self.zoom_out_action.triggered.connect(lambda:self.view.zoom_by(1/1.2))
        self.pan_action=toolbar.addAction('Pan');self.pan_action.setCheckable(True)
        self.pan_action.setToolTip('Pan canvas with left drag · click again or press Esc to return to selection')
        self.pan_action.toggled.connect(lambda enabled:self.view.set_pan_mode(enabled))
        self.fit_action=toolbar.addAction('Fit model')
        self.fit_action.triggered.connect(self.fit_model)
        self.actual_size_action=toolbar.addAction('100%')
        self.actual_size_action.triggered.connect(lambda:self.view.actual_size())
        self.zoom_label=QLabel('100%');toolbar.addWidget(self.zoom_label)
        self.delete_action=toolbar.addAction('Delete selected')
        self.delete_action.setShortcut('Delete')
        self.delete_action.triggered.connect(self.delete_selected)
        toolbar.addSeparator()
        self.validate_button=QPushButton('Validate')
        self.validate_button.clicked.connect(self.validate_model)
        self.train_button=QPushButton('Train PINN')
        self.train_button.clicked.connect(lambda:self.start_training())
        self.cancel_button=QPushButton('Stop')
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_training)
        for button in [self.validate_button,self.train_button,self.cancel_button]:toolbar.addWidget(button)
        self.progress_bar=QProgressBar();self.progress_bar.setMaximumWidth(210)
        self.progress_bar.setRange(0,self.config.iterations);self.progress_bar.setValue(0)
        toolbar.addWidget(self.progress_bar)
        self.scene=Canvas()
        self.view=CanvasView(self.scene)
        self.view.panModeChanged.connect(self.pan_action.setChecked)
        self.view.blockInsertRequested.connect(self.add_node)
        self.view.quickInsertRequested.connect(self.show_quick_insert)
        self.view.zoomChanged.connect(lambda value:self.zoom_label.setText(f"{value*100:.0f}%"))
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setCentralWidget(self.view)
        self.library=BlockLibrary()
        self.library.blockActivated.connect(self.add_node)
        self.library_dock=self.dock('Block library',self.library,Qt.DockWidgetArea.LeftDockWidgetArea)
        properties=QWidget();layout=QVBoxLayout(properties)
        self.name=QLineEdit();self.name.setReadOnly(True)
        self.detail=QLineEdit();self.detail.setReadOnly(True)
        layout.addWidget(QLabel('Selected block'));layout.addWidget(self.name);layout.addWidget(self.detail)
        self.property_scroll=QScrollArea();self.property_scroll.setWidgetResizable(True)
        self.property_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self.property_scroll)
        self.validation=QLabel('');self.validation.setWordWrap(True);layout.addWidget(self.validation)
        self.properties_dock=self.dock('Properties',properties,Qt.DockWidgetArea.RightDockWidgetArea)
        self.properties_dock.setMinimumWidth(340)
        self.log=QPlainTextEdit();self.log.setReadOnly(True)
        self.messages_dock=self.dock('Messages',self.log,Qt.DockWidgetArea.BottomDockWidgetArea)
        self.messages_dock.hide()
        self.output_views=OutputViews(self)
        self.setup_menus()
        self.setup_icons()
        self.scene.blockOpened.connect(self.open_block)
        self.scene.connectionRejected.connect(self.connection_rejected)
        self.scene.blockContextRequested.connect(self.show_block_menu)
        self.scene.selectionChanged.connect(self.selection_changed)
        self.populate_template()
        self.scene.graphChanged.connect(self.graph_changed)
        self.scene.edit_handler=self.push_edit
        self.scene.selectionChanged.emit()
        self.log.appendPlainText('Poisson template: Δu + c = 0. Select a block to configure it, validate, then train. Plot views are optional; add and connect an output block, then double-click it to open its view.\nBoundary sets are sampled first; assign a constant condition to each side in the BC block.')
        self.undo_stack.cleanChanged.connect(self.update_document_title)
        self.update_document_title()
        self.statusBar().showMessage('Ready · rectangle Poisson template · select blocks to edit settings')
        QTimer.singleShot(0,self.fit_model)
        self.desktop_state=None
        if state_directory is not False:
            try:
                self.desktop_state=DesktopState(self,state_directory)
                self.desktop_state.restore()
                QTimer.singleShot(0,self.desktop_state.offer_recovery)
            except OSError as error:self.statusBar().showMessage('Local preferences unavailable: '+str(error))

    def setup_icons(self):
        actions={
            'open_model_action':'open','save_model_action':'save','save_as_action':'save',
            'open_run_action':'open','save_run_action':'save','load_action':'examples',
            'heat_action':'examples','flow_action':'examples','zoom_in_action':'zoom_in',
            'zoom_out_action':'zoom_out','pan_action':'pan','fit_action':'fit',
            'actual_size_action':'actual','delete_action':'delete','undo_action':'undo',
            'redo_action':'redo','copy_action':'copy','cut_action':'cut','paste_action':'paste',
            'check_action':'check','run_action':'play','stop_action':'stop',
            'open_output_action':'solution',
        }
        for attribute,glyph in actions.items():
            action=getattr(self,attribute);action.setIcon(icon(glyph));action.setIconVisibleInMenu(True)
            if not action.toolTip():action.setToolTip(action.text())
        for action in self.example_actions.values():action.setIcon(icon('examples'))
        self.examples_menu.setIcon(icon('examples'))
        for button,glyph in ((self.validate_button,'check'),(self.train_button,'play'),(self.cancel_button,'stop')):
            button.setIcon(icon(glyph))
        for menu in self.menuBar().actions():
            for action in menu.menu().actions():
                glyph={'Quit':'quit','Help':'help','Block capabilities':'help','Select all blocks':'domain','Reset panel layout':'fit'}.get(action.text())
                if glyph:action.setIcon(icon(glyph))
                elif menu.text()=='Help':action.setIcon(icon('help'))

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, '_initial_layout_fitted', False):
            self._initial_layout_fitted = True
            QTimer.singleShot(0, self.fit_model)

    def setup_menus(self):
        file_menu=self.menuBar().addMenu('File')
        self.open_model_action=file_menu.addAction('Open model…')
        self.open_model_action.setShortcut('Ctrl+O');self.open_model_action.triggered.connect(self.open_model)
        self.save_model_action=file_menu.addAction('Save model')
        self.save_model_action.setShortcut('Ctrl+S');self.save_model_action.triggered.connect(lambda:self.save_model())
        self.save_as_action=file_menu.addAction('Save model as…')
        self.save_as_action.setShortcut('Ctrl+Shift+S');self.save_as_action.triggered.connect(self.save_model_as)
        self.open_run_action=file_menu.addAction('Open saved run…')
        self.open_run_action.triggered.connect(lambda:self.open_saved_run())
        self.save_run_action=file_menu.addAction('Save run and checkpoint…')
        self.save_run_action.triggered.connect(lambda:self.save_current_run())
        self.save_run_action.setEnabled(False)
        file_menu.addSeparator()
        self.load_action.setShortcut('Ctrl+N')
        self.examples_menu=file_menu.addMenu('Examples')
        self.examples_menu.addAction(self.load_action)
        self.examples_menu.addAction(self.heat_action)
        self.flow_action=self.examples_menu.addAction('2D incompressible flow');self.flow_action.triggered.connect(lambda:self.request_template('flow'))
        self.examples_menu.addSeparator()
        self.examples_1d_menu=self.examples_menu.addMenu('1D examples');self.examples_1d_menu.setIcon(icon('examples'))
        self.example_actions={}
        for identifier,example in EXAMPLE_PROBLEMS.items():
            menu=self.examples_1d_menu if example_config(identifier).shape=='Interval' else self.examples_menu
            action=menu.addAction(example.label);action.setToolTip(example.description)
            action.triggered.connect(lambda checked=False,key=identifier:self.request_example(key));self.example_actions[identifier]=action
        recipes=file_menu.addMenu('Insert PINN recipe');recipes.setIcon(icon('recipe'))
        for recipe in COMPACT_RECIPES.values():
            recipes.addAction(recipe.NAME+' · compact').triggered.connect(lambda checked=False,r=recipe:self.add_node(r.NAME))
            recipes.addAction(recipe.NAME+' · expanded').triggered.connect(lambda checked=False,r=recipe:self.insert_expanded_recipe(r))
        self.recipe_menu=recipes
        file_menu.addSeparator()
        file_menu.addAction('Quit').triggered.connect(self.close)
        file_menu.actions()[-1].setShortcut('Ctrl+Q')
        edit=self.menuBar().addMenu('Edit')
        self.undo_action=edit.addAction('Undo');self.undo_action.setShortcut('Ctrl+Z')
        self.redo_action=edit.addAction('Redo');self.redo_action.setShortcuts(['Ctrl+Y','Ctrl+Shift+Z'])
        for action,callback in [(self.undo_action,self.undo_edit),(self.redo_action,self.redo_edit)]:
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            self.view.addAction(action);action.triggered.connect(callback)
        self.undo_stack.indexChanged.connect(self.update_history_actions)
        self.undo_stack.undoTextChanged.connect(self.update_history_actions)
        self.undo_stack.redoTextChanged.connect(self.update_history_actions)
        self.update_history_actions()
        edit.addSeparator()
        self.copy_action=edit.addAction('Copy blocks');self.copy_action.setShortcut('Ctrl+C')
        self.cut_action=edit.addAction('Cut blocks');self.cut_action.setShortcut('Ctrl+X')
        self.paste_action=edit.addAction('Paste blocks');self.paste_action.setShortcut('Ctrl+V')
        for action,callback in [(self.copy_action,self.copy_blocks),(self.cut_action,self.cut_blocks),(self.paste_action,self.paste_blocks)]:
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            self.view.addAction(action);action.triggered.connect(callback)
        QApplication.clipboard().dataChanged.connect(self.update_clipboard_actions)
        edit.addSeparator()
        edit.addAction(self.delete_action)
        self.delete_action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.view.addAction(self.delete_action)
        select_all=edit.addAction('Select all blocks')
        select_all.setShortcut('Ctrl+A')
        select_all.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.view.addAction(select_all)
        select_all.triggered.connect(self.select_all_blocks)
        self.view_menu=self.menuBar().addMenu('View')
        for dock in [self.library_dock,self.properties_dock,self.messages_dock]:
            self.view_menu.addAction(dock.toggleViewAction())
        self.view_menu.addSeparator()
        for action,shortcuts in [(self.zoom_in_action,['Ctrl++','Ctrl+=']),(self.zoom_out_action,['Ctrl+-']),(self.actual_size_action,['Ctrl+0'])]:
            action.setShortcuts(shortcuts)
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            self.view.addAction(action);self.view_menu.addAction(action)
        self.view_menu.addAction(self.pan_action)
        self.view_menu.addAction(self.fit_action)
        self.view_menu.addAction('Reset panel layout').triggered.connect(self.reset_panel_layout)
        self.open_output_action=self.view_menu.addAction('Open selected output')
        self.open_output_action.setEnabled(False)
        self.open_output_action.triggered.connect(self.open_selected_output)
        themes=self.view_menu.addMenu('Theme')
        self.theme_actions={}
        group=QActionGroup(self);group.setExclusive(True)
        for name in ['light','dark']:
            action=themes.addAction(name.capitalize())
            action.setCheckable(True);action.setChecked(name==theme_name())
            action.triggered.connect(lambda checked=False,n=name:configure_theme(QApplication.instance(),n))
            group.addAction(action);self.theme_actions[name]=action
        run=self.menuBar().addMenu('Run')
        self.check_action=run.addAction('Validate model')
        self.check_action.setShortcut('Ctrl+D')
        self.check_action.triggered.connect(self.validate_button.click)
        self.run_action=run.addAction('Train PINN')
        self.run_action.setShortcut('F5');self.run_action.triggered.connect(self.train_button.click)
        self.stop_action=run.addAction('Stop')
        self.stop_action.setShortcut('Ctrl+Shift+T');self.stop_action.setEnabled(False)
        self.stop_action.triggered.connect(self.cancel_button.click)
        self.resume_action=run.addAction('Resume checkpoint…');self.resume_action.setEnabled(False)
        self.resume_action.triggered.connect(lambda:self.resume_training())
        help_menu=self.menuBar().addMenu('Help')
        self.manual_action=help_menu.addAction('User manual');self.manual_action.setShortcut('F1')
        self.manual_action.triggered.connect(self.open_manual)
        help_menu.addAction('Quick start').triggered.connect(lambda:QMessageBox.information(
            self,'Build and train a PINN',
            'Press F1 for the illustrated user manual.\n\n'
            'Start with File → Examples → 1D examples → Poisson — baseline. '
            'Select a block to edit its settings, then Validate and Train PINN (F5).\n\n'
            'Open a connected Training Monitor or Solution Plot explicitly to see results. '
            'Comparison also requires a Reference block.\n\n'
            'Use the mouse wheel to zoom, the Pan button or Space+drag to pan, and Fit model to frame the graph. '
            'Add CSV Export or ParaView Export when you want to export predictions.'))
        help_menu.addAction('About Margam').triggered.connect(lambda:QMessageBox.about(
            self,'Margam','A visual builder for physics-informed neural networks.\n'
            'Baseline, hard Dirichlet, SA-PINN and LRA; scalar problems and steady 2D incompressible flow.\n'
            'Reusable recipes, pointwise networks, optional results and explicit CSV/ParaView export.'))

    def open_manual(self):
        path=Path(__file__).resolve().parents[2]/'build/manual/site/index.html'
        if not path.is_file():
            QMessageBox.information(self,'User manual',
                'The illustrated manual has not been built in this checkout.\n\n'
                'From the repository root, install the optional documentation tools and build once:\n\n'
                'uv pip install --python .venv/bin/python -e ".[docs]"\n'
                '.venv/bin/python -m examples.documentation build\n\n'
                'Then press F1 again. The Markdown guides are also available in docs/.')
            return False
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            QMessageBox.warning(self,'User manual','The browser could not open the manual. Open this file directly:\n'+str(path));return False
        return True

    def update_clipboard_actions(self):
        if not hasattr(self,'copy_action'):return
        selected=any(node.isSelected() for node in self.scene.nodes())
        locked=self.is_running() or self.scene.locked
        self.copy_action.setEnabled(selected)
        self.cut_action.setEnabled(selected and not locked)
        mime=QApplication.clipboard().mimeData()
        self.paste_action.setEnabled(not locked and mime is not None and mime.hasFormat('application/x-margam-model-fragment'))

    def copy_blocks(self):
        identifiers={node.identifier for node in self.scene.nodes() if node.isSelected()}
        if not identifiers:return False
        document=capture_model(self)
        document['nodes']=[n for n in document['nodes'] if n['id'] in identifiers]
        document['wires']=[wire for wire in document['wires'] if wire['source'] in identifiers and wire['target'] in identifiers]
        content=json.dumps(document,allow_nan=False).encode('utf-8')
        if len(content)>MAX_BYTES:return False
        mime=QMimeData();mime.setData('application/x-margam-model-fragment',content)
        QApplication.clipboard().setMimeData(mime);self._paste_serial=0
        return True

    def cut_blocks(self):
        if self.is_running() or self.scene.locked:return False
        if not self.copy_blocks():return False
        self.scene.delete_selection('Cut blocks');return True

    def paste_blocks(self):
        if self.is_running() or self.scene.locked:return False
        mime=QApplication.clipboard().mimeData()
        if mime is None or not mime.hasFormat('application/x-margam-model-fragment'):return False
        try:
            content=bytes(mime.data('application/x-margam-model-fragment'))
            if len(content)>MAX_BYTES:raise ValueError('Copied model exceeds 4 MiB.')
            data=json.loads(content,object_pairs_hook=unique_keys)
            _,records,connections=decode_document(data)
            if not records:return False
        except (ValueError,TypeError,KeyError,RecursionError,UnicodeError,OverflowError) as error:
            self.statusBar().showMessage('Cannot paste blocks: '+str(error));return False
        self._paste_serial+=1
        center=self.view.mapToScene(self.view.viewport().rect().center())+QPointF(20,20)*(self._paste_serial-1)
        origin=QPointF((min(n['position'][0] for n in records)+max(n['position'][0] for n in records)+170)/2,
                       (min(n['position'][1] for n in records)+max(n['position'][1] for n in records)+80)/2)
        titles={kind:title for title,kind in KINDS.items()};nodes={};staged=Canvas()
        for record in records:
            point=QPointF(*record['position'])+center-origin
            node=Node(titles[record['kind']],'',point.x(),point.y())
            node.title=record['title'];node.settings=deepcopy(record['settings'])
            if node.kind in PARAMETERS:node.settings.setdefault('parameters',parameters_for(node.kind,decode_document(data)[0]))
            staged.addItem(node);nodes[record['id']]=node
        for wire in connections:staged.connect_nodes(nodes[wire['source']],nodes[wire['target']],wire['input'])
        wires=list(staged.wires);staged.wires.clear()
        for wire in wires:staged.removeItem(wire)
        for node in nodes.values():staged.removeItem(node)
        def add():
            self.scene.clearSelection()
            for node in nodes.values():self.scene.addItem(node);node.setSelected(True)
            for wire in wires:self.scene.addItem(wire);self.scene.wires.append(wire);wire.update_path()
            self.graph_changed()
        def remove():
            for wire in wires:self.scene.wires.remove(wire);self.scene.removeItem(wire)
            for node in nodes.values():self.scene.removeItem(node)
            self.graph_changed()
        self.push_edit('Paste blocks',add,remove)
        self.statusBar().showMessage('Pasted blocks · core blocks still use the current template settings')
        return True

    def update_document_title(self,*args):
        name=Path(self.model_path).name if self.model_path else 'Untitled'
        self.setWindowTitle('Margam · '+name+'[*] · PINN builder')
        self.setWindowModified(not self.undo_stack.isClean())

    def commit_inspector(self):
        focus=QApplication.focusWidget()
        if focus is not None and self.property_scroll.isAncestorOf(focus):focus.clearFocus()

    def confirm_discard(self):
        self.commit_inspector()
        if self.undo_stack.isClean():return True
        answer=QMessageBox.question(self,'Unsaved model',
            'Save changes to the current model before continuing?',
            QMessageBox.StandardButton.Save|QMessageBox.StandardButton.Discard|QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if answer==QMessageBox.StandardButton.Save:return self.save_model()
        return answer==QMessageBox.StandardButton.Discard

    def request_template(self,heat=False):
        if self.is_running() or self.scene.locked or not self.confirm_discard():return False
        if heat=='flow':self.load_flow_template()
        elif heat:self.load_heat_template()
        else:self.reset_template()
        return True

    def save_model_as(self):
        if self.is_running() or self.scene.locked:return False
        path,_=QFileDialog.getSaveFileName(self,'Save Margam model',self.model_path or 'model.margam.json',
                                         'Margam model (*.margam.json *.json)')
        if not path:return False
        if not path.lower().endswith('.json'):
            path+='.margam.json'
            if Path(path).exists() and QMessageBox.question(self,'Replace model?',
                    'Replace '+Path(path).name+'?',QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No)!=QMessageBox.StandardButton.Yes:return False
        return self.save_model(path)

    def save_model(self,path=None):
        if self.is_running() or self.scene.locked:return False
        self.commit_inspector()
        path=path or self.model_path
        if not path:return self.save_model_as()
        try:write_model(path,capture_model(self))
        except (OSError,ValueError,TypeError) as error:
            QMessageBox.warning(self,'Model could not be saved',str(error));return False
        self.model_path=str(Path(path).absolute())
        self.undo_stack.setClean();self._history_epoch+=1;self.update_document_title()
        self.statusBar().showMessage('Saved model · '+Path(path).name)
        return True

    def open_model(self):
        if self.is_running() or self.scene.locked:return False
        path,_=QFileDialog.getOpenFileName(self,'Open Margam model',str(Path(self.model_path).parent) if self.model_path else '',
                                         'Margam model (*.margam.json *.json)')
        return self.load_model(path) if path else False

    def load_model(self,path):
        if self.is_running() or self.scene.locked:return False
        try:read_model(path)
        except (OSError,ValueError,TypeError,KeyError,RecursionError) as error:
            QMessageBox.warning(self,'Model could not be opened',str(error));return False
        if not self.confirm_discard():return False
        try:return self.replace_model(read_model(path),path)
        except (OSError,ValueError,TypeError,KeyError,RecursionError) as error:
            QMessageBox.warning(self,'Model could not be opened',str(error));return False

    def replace_model(self,data,path=None):
        config,records,connections=decode_document(data)
        staged=Canvas();nodes={}
        titles={kind:title for title,kind in KINDS.items()}
        for record in records:
            node=Node(titles[record['kind']],'',*record['position'])
            node.identifier=record['id'];node.title=record['title'];node.settings=deepcopy(record['settings'])
            if node.kind in PARAMETERS:node.settings.setdefault('parameters',parameters_for(node.kind,config))
            staged.addItem(node);nodes[node.identifier]=node
        for wire in connections:staged.connect_nodes(nodes[wire['source']],nodes[wire['target']],wire['input'])
        if self.quick_insert is not None:self.quick_insert.reject()
        self._resetting=True;self.scene.blockSignals(True)
        try:
            self.clear_run_data();self.output_views.sync([]);self.close_rectangle_previews()
            self.scene.cancel_wire();self.undo_stack.clear()
            self.scene.wires.clear();self.scene.clear();self.config=config
            for wire in staged.wires:staged.removeItem(wire)
            for node in nodes.values():staged.removeItem(node);self.scene.addItem(node)
            self.scene.wires=list(staged.wires);staged.wires.clear()
            for wire in self.scene.wires:self.scene.addItem(wire);wire.update_path()
        finally:
            self.scene.blockSignals(False);self._resetting=False
        self.model_path=str(Path(path).absolute()) if path else None;self._history_epoch+=1
        self.settings_changed();self.selection_changed();self.fit_model()
        self.undo_stack.setClean();self.update_document_title();self.update_history_actions()
        self.statusBar().showMessage('Opened model · validate before training')
        return True

    def update_run_actions(self):
        if not hasattr(self,'save_run_action'):return
        locked=self.is_running() or self.scene.locked
        self.open_run_action.setEnabled(not locked)
        self.save_run_action.setEnabled(not locked and self.result is not None)
        self.resume_action.setEnabled(not locked and (self.loaded_checkpoint is not None or self.last_run is not None and self.last_run.model is not None))

    def save_current_run(self,path=None):
        if self.is_running() or self.result is None:return False
        if path is None:
            path,_=QFileDialog.getSaveFileName(self,'Save run and checkpoint','run.margam-run.pt','Margam run (*.margam-run.pt)')
            if path and not path.lower().endswith('.pt'):
                path+='.margam-run.pt'
                if Path(path).exists() and QMessageBox.question(self,'Replace saved run?','Replace '+Path(path).name+'?',QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)!=QMessageBox.StandardButton.Yes:return False
        if not path:return False
        try:save_run(path,capture_run(self))
        except Exception as error:QMessageBox.warning(self,'Run could not be saved',str(error));return False
        self.statusBar().showMessage('Saved run and method checkpoint · '+Path(path).name)
        return True

    def open_saved_run(self,path=None):
        if self.is_running() or self.scene.locked:return False
        if path is None:path,_=QFileDialog.getOpenFileName(self,'Open saved run','','Margam run (*.margam-run.pt)')
        if not path:return False
        try:data=load_run(path)
        except Exception as error:QMessageBox.warning(self,'Run could not be opened',str(error));return False
        if not self.confirm_discard():return False
        try:self.replace_model(data['model'])
        except Exception as error:QMessageBox.warning(self,'Run could not be opened',str(error));return False
        self.result=deepcopy(data['result']);self.result['config']=self.config
        for output in self.result['outputs'].values():output['config']=self.config
        self.monitor_records=deepcopy(data['monitors']);self.loaded_checkpoint=data['checkpoint']
        self.progress_bar.setValue(self.result['steps']);self.update_run_actions()
        self.statusBar().showMessage('Opened saved run · views stay closed until requested')
        return True

    def resume_training(self,additional=None):
        if self.is_running() or self.scene.locked:return False
        checkpoint=self.last_run.model.training_state() if self.last_run is not None and self.last_run.model is not None else self.loaded_checkpoint
        if checkpoint is None or checkpoint['iter']>=100000:return False
        if additional is None:
            additional,accepted=QInputDialog.getInt(self,'Resume checkpoint','Additional updates',1000,1,100000-checkpoint['iter'])
            if not accepted:return False
        if type(additional) is not int or not 1<=additional<=100000-checkpoint['iter']:return False
        self.change('iterations',checkpoint['iter']+additional)
        self.change('device','cpu')
        return self.start_training(checkpoint=checkpoint)

    def push_edit(self,text,redo,undo,merge_key=None):
        if self.is_running() or self.scene.locked or self._history_replaying:return
        if self._resetting:redo();return
        key=(self._history_epoch,merge_key) if merge_key is not None else None
        self.undo_stack.push(EditCommand(text,redo,undo,self.selection_changed,key))

    def update_history_actions(self,*args):
        locked=self.is_running() or self.scene.locked
        self.undo_action.setEnabled(not locked and self.undo_stack.canUndo())
        self.redo_action.setEnabled(not locked and self.undo_stack.canRedo())
        self.undo_action.setText('Undo '+self.undo_stack.undoText() if self.undo_stack.canUndo() else 'Undo')
        self.redo_action.setText('Redo '+self.undo_stack.redoText() if self.undo_stack.canRedo() else 'Redo')

    def undo_edit(self):
        if self.is_running() or self.scene.locked:return
        self.scene.cancel_wire()
        if self.quick_insert is not None:self.quick_insert.reject()
        self._history_replaying=True
        try:self.undo_stack.undo()
        finally:self._history_replaying=False

    def redo_edit(self):
        if self.is_running() or self.scene.locked:return
        self.scene.cancel_wire()
        if self.quick_insert is not None:self.quick_insert.reject()
        self._history_replaying=True
        try:self.undo_stack.redo()
        finally:self._history_replaying=False

    def select_all_blocks(self):
        self.view.setFocus()
        for node in self.scene.nodes():node.setSelected(True)

    def reset_panel_layout(self):
        messages_hidden=self.messages_dock.isHidden()
        for dock,area in [(self.library_dock,Qt.DockWidgetArea.LeftDockWidgetArea),
                          (self.properties_dock,Qt.DockWidgetArea.RightDockWidgetArea),
                          (self.messages_dock,Qt.DockWidgetArea.BottomDockWidgetArea)]:
            dock.setFloating(False);self.addDockWidget(area,dock);dock.show()
        if messages_hidden:self.messages_dock.hide()
        first_output=None
        for dock,_ in self.output_views.entries.values():
            hidden=dock.isHidden()
            dock.setFloating(False)
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea,dock)
            if first_output is not None:self.tabifyDockWidget(first_output,dock)
            else:first_output=dock
            if hidden:dock.hide()
        self.resizeDocks([self.library_dock,self.properties_dock],[240,310],Qt.Orientation.Horizontal)

    def export_block_data(self,node,path=None):
        if self.is_running():return False
        data=(self.result or {}).get('outputs',{}).get(node.identifier)
        if data is None:
            self.statusBar().showMessage('Connect this export block and train before exporting.');return False
        paraview=node.kind=='paraview'
        extension='.vti' if paraview else '.csv'
        label='ParaView predictions' if paraview else 'CSV predictions'
        if path is None:
            path,_=QFileDialog.getSaveFileName(self,'Export '+label,'prediction'+extension,'VTK ImageData (*.vti)' if paraview else 'CSV (*.csv)')
            if path and not path.lower().endswith(extension):
                path+=extension
                if Path(path).exists() and QMessageBox.question(self,'Replace export?','Replace '+Path(path).name+'?',QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)!=QMessageBox.StandardButton.Yes:return False
        if not path:return False
        try:
            if paraview:
                from ..components.paraview_export import export_vti
                export_vti(path,data)
            else:export_csv(path,data)
        except (OSError,ValueError,KeyError) as error:QMessageBox.warning(self,'Predictions could not be exported',str(error));return False
        self.statusBar().showMessage('Exported '+label+' · '+Path(path).name);return True

    def open_selected_output(self):
        nodes=[node for node in self.scene.nodes() if node.isSelected()]
        if len(nodes)==1:self.open_block(nodes[0])

    def open_block(self,node):
        if not isValid(node) or node.scene() is not self.scene:return
        if node.kind in COMPACT_RECIPES:return self.expand_recipe(node)
        if node.kind in ('export','paraview'):return self.export_block_data(node)
        if node.kind in OUTPUT_KINDS:
            return self.output_views.open(node)
        self.properties_dock.show()
        self.properties_dock.raise_()

    def create_block_menu(self, node):
        """Only existing block-specific actions; no universal plot capability."""
        menu=QMenu(self)
        menu.addAction('Properties').triggered.connect(lambda:self.show_block_properties(node))
        if node.kind in COMPACT_RECIPES:
            action=menu.addAction('Expand into components');action.setEnabled(not self.is_running())
            action.triggered.connect(lambda:self.expand_recipe(node))
        elif node.kind in TRAINER_RECIPES:
            recipe=TRAINER_RECIPES[node.kind]
            action=menu.addAction('Collapse as '+recipe.NAME);action.setEnabled(not self.is_running() and self.config_for_node(node).method==recipe.DEFAULTS['method'])
            action.triggered.connect(lambda:self.collapse_recipe(node))
        if node.kind=='domain' and self.config_for_node(node).shape=='Rectangle':
            menu.addSeparator()
            menu.addAction('Preview geometry').triggered.connect(lambda:self.open_rectangle_preview(node,False))
            menu.addAction('Preview samples').triggered.connect(lambda:self.open_rectangle_preview(node,True))
        elif node.kind in OUTPUT_KINDS:
            label={'monitor':'Open monitor','solution':'Open plot','comparison':'Open comparison','export':'Export CSV…','paraview':'Export ParaView…'}[node.kind]
            menu.addAction(label).triggered.connect(lambda:self.open_block(node))
        if node.kind in ('sa_recipe','sa_method'):
            summary=menu.addAction('Adaptive weight summary')
            summary.setEnabled(not self.is_running() and self.result is not None)
            summary.triggered.connect(self.show_adaptive_summary)
        if node.kind in ('lra_recipe','lra_method'):
            summary=menu.addAction('LRA loss-weight summary');summary.setEnabled(not self.is_running() and self.result is not None)
            summary.triggered.connect(self.show_lra_summary)
        menu.addSeparator()
        remove=menu.addAction('Delete block')
        remove.setEnabled(not self.is_running())
        remove.triggered.connect(lambda:self.delete_block(node))
        menu.addSeparator()
        menu.addAction(self.copy_action);menu.addAction(self.cut_action);menu.addAction(self.paste_action)
        menu.addAction(self.undo_action);menu.addAction(self.redo_action)
        menu.addAction('Help').triggered.connect(lambda:self.block_help(node))
        return menu

    def adaptive_summary(self):
        if self.last_run is not None and self.last_run.model is not None and hasattr(self.last_run.model,'adaptive_summary'):return self.last_run.model.adaptive_summary()
        if self.loaded_checkpoint and 'adaptive' in self.loaded_checkpoint:
            return {name:dict(minimum=w.min().item(),mean=w.mean().item(),maximum=w.max().item()) for name,w in self.loaded_checkpoint['adaptive']['weights'].items()}
        return {}

    def show_adaptive_summary(self):
        summary=self.adaptive_summary()
        text='Per-point λ values; the loss mask is λ².\n\n'+'\n'.join(f'{name}: min {v["minimum"]:.5g}, mean {v["mean"]:.5g}, max {v["maximum"]:.5g}' for name,v in summary.items())
        QMessageBox.information(self,'Adaptive weight summary',text if summary else 'Train the self-adaptive recipe first.')

    def lra_summary(self):
        if self.last_run is not None and self.last_run.model is not None and hasattr(self.last_run.model,'balancing_summary'):return self.last_run.model.balancing_summary()
        if self.loaded_checkpoint and 'lra' in self.loaded_checkpoint:
            state=self.loaded_checkpoint['lra'];latest=state['history'][-1] if state['history'] else None
            return dict(step=latest['step'] if latest else None,pde_max=latest['pde_max'] if latest else None,
                        terms={name:dict(coefficient=value,effective_weight=value*state['settings']['global_weights'].get(name,1.),mean_gradient=latest['means'][name] if latest else None) for name,value in state['weights'].items()})
        return {}

    def show_lra_summary(self):
        summary=self.lra_summary()
        text='LRA scalar loss coefficients; Adam base learning rate is unchanged.\n\n'
        if summary:
            text+=f"Last balance before update {summary['step']}\n"+'\n'.join(f"{name}: coefficient {values['coefficient']:.6g}, effective weight {values['effective_weight']:.6g}" for name,values in summary['terms'].items())
        else:text+='Train the LRA recipe first.'
        QMessageBox.information(self,'LRA loss-weight summary',text)

    def show_block_menu(self, node, position):
        menu=self.create_block_menu(node)
        menu.exec(position)
        menu.deleteLater()

    def show_block_properties(self, node):
        if not isValid(node) or node.scene() is not self.scene:return
        self.scene.clearSelection();node.setSelected(True)
        self.properties_dock.show();self.properties_dock.raise_()

    def delete_block(self, node):
        if self.is_running() or not isValid(node) or node.scene() is not self.scene:return
        self.scene.clearSelection();node.setSelected(True)
        self.delete_selected()

    def block_help(self, node):
        text=describe_block(node.kind)
        QMessageBox.information(self,node.title,text)

    def open_rectangle_preview(self, node, show_samples):
        if not isValid(node) or node.scene() is not self.scene or node.kind!='domain' or self.config_for_node(node).shape!='Rectangle':
            return
        key=(node.identifier,show_samples)
        if key not in self.rectangle_previews:
            title='Rectangle · samples' if show_samples else 'Rectangle · geometry'
            view=RectanglePreview(self.config_for_node(node),show_samples)
            dock=self.dock(title,view,Qt.DockWidgetArea.BottomDockWidgetArea)
            dock.setObjectName('rectangle-'+node.identifier+('-samples' if show_samples else '-geometry'))
            dock.setFloating(True);dock.resize(660,420)
            self.rectangle_previews[key]=(dock,view)
        dock,_=self.rectangle_previews[key]
        dock.show();dock.raise_()
        return dock

    def close_rectangle_previews(self):
        for dock,_ in self.rectangle_previews.values():
            dock.close();self.removeDockWidget(dock);dock.deleteLater()
        self.rectangle_previews.clear()

    def sync_rectangle_previews(self):
        owners={n.identifier:n for n in self.scene.nodes() if n.kind=='domain'}
        for key in list(self.rectangle_previews):
            if key[0] not in owners or self.config_for_node(owners[key[0]]).shape!='Rectangle':
                dock,_=self.rectangle_previews.pop(key)
                dock.close();self.removeDockWidget(dock);dock.deleteLater()


    def rename_output(self,node,name):
        name=name.strip()
        if not name or name==node.title or node.scene() is not self.scene:return
        old=node.title
        def apply(value):
            node.title=value;node.update()
            if node.isSelected():self.name.setText(value)
            entry=self.output_views.entries.get(node.identifier)
            if entry:entry[0].setWindowTitle(value)
        self.push_edit('Rename output',lambda:apply(name),lambda:apply(old))

    def change_output(self,node,key,value,edit_token=None):
        existed=key in node.settings;old=deepcopy(node.settings.get(key))
        if existed and old==value:return
        value=deepcopy(value)
        def apply(number,present=True):
            if present:node.settings[key]=deepcopy(number)
            else:node.settings.pop(key,None)
            if key=='history_limit':
                records=self.monitor_records.get(node.identifier,[])
                del records[:-number]
                self.output_views.refresh(node)
            else:self.graph_changed()
        merge=(node.identifier,key,edit_token) if edit_token is not None else None
        self.push_edit('Change output '+key.replace('_',' '),lambda:apply(value),lambda:apply(old,existed),merge)


    def dock(self,title,widget,area):
        dock=QDockWidget(title,self);dock.setObjectName(title)
        dock.setWidget(widget);self.addDockWidget(area,dock)
        return dock

    def fit_model(self):
        self.view.fit_model()

    def make_node(self,title,detail,x,y):
        node=Node(title,detail,x,y)
        if node.kind in PARAMETERS:node.settings['parameters']=parameters_for(node.kind,self.config)
        self.scene.addItem(node);return node

    def populate_template(self):
        definitions=[('Rectangle','Unit square',0,160),('Interior sampler','PDE points',240,40),
                     ('Boundary points','Named sides',240,270),('PDE residual','Poisson',480,40),
                     ('Boundary condition','Side conditions',480,270),('Neural network','Fully connected',480,440),
                     ('PINN method','Baseline PINN',730,160)]
        nodes=[self.make_node(*d) for d in definitions]
        for a,b in [(0,1),(0,2),(1,3),(2,4),(3,6),(4,6),(5,6)]:self.scene.connect_nodes(nodes[a],nodes[b])
        self.refresh_descriptions()
        nodes[0].setSelected(True)

    def reset_template(self):
        if self.is_running():return
        if self.quick_insert is not None:self.quick_insert.reject()
        self._resetting=True
        self.undo_stack.clear()
        self.output_views.sync([])
        self.close_rectangle_previews()
        self.scene.wires.clear();self.scene.pending=None;self.scene.clear()
        self.config=PoissonConfig()
        self.populate_template();self.settings_changed();self.fit_model()
        self.log.appendPlainText('Loaded the rectangle Poisson template.')
        self._resetting=False
        self.model_path=None;self.undo_stack.setClean();self.update_document_title()
        self.update_history_actions()

    def load_heat_template(self):
        if self.is_running():return
        self.reset_template()
        self._resetting=True
        self.config=heat_demo_config()
        for node in self.scene.nodes():
            if node.kind in PARAMETERS:node.settings['parameters']=parameters_for(node.kind,self.config)
        self.settings_changed()
        method=next(n for n in self.scene.nodes() if n.kind=='method')
        outputs=[]
        for title,position in [('Training Monitor',(1000,0)),('Solution Plot',(1000,180)),('Comparison',(1000,360))]:
            node=self.add_node(title);node.setPos(*position)
            self.scene.connect_nodes(method,node)
            outputs.append(node)
        reference=self.add_node('Reference');reference.setPos(730,470)
        self.scene.connect_nodes(reference,outputs[-1])
        self.scene.clearSelection();method.setSelected(True)
        self.validate_model();self.fit_model()
        self._resetting=False
        self.undo_stack.clear();self.undo_stack.setClean();self.update_document_title();self.update_history_actions()
        self.statusBar().showMessage('Heat demo ready · Train PINN, then open monitor, prediction or comparison blocks')
        self.log.appendPlainText('Loaded steady heat demo: unit conductivity, constant source 2, zero edge temperature. Explicit series reference and output blocks are connected; views remain closed until requested.')

    def load_flow_template(self):
        if self.is_running():return
        self.reset_template();self._resetting=True;self.config=flow_demo_config()
        for node in self.scene.nodes():
            if node.kind in PARAMETERS:node.settings['parameters']=parameters_for(node.kind,self.config)
        self.settings_changed()
        method=next(node for node in self.scene.nodes() if node.kind=='method');comparison=None
        for index,title in enumerate(('Training Monitor','Solution Plot','Comparison','CSV Export')):
            node=self.add_node(title);node.setPos(1000,index*150);self.scene.connect_nodes(method,node)
            if node.kind=='comparison':comparison=node
        reference=self.add_node('Reference');reference.setPos(730,500);self.scene.connect_nodes(reference,comparison)
        self.scene.clearSelection();method.setSelected(True)
        self._resetting=False;self.undo_stack.clear();self.undo_stack.setClean()
        self.validate_model();self.fit_model();self.update_document_title()
        self.statusBar().showMessage('Steady incompressible Couette demo · u, v, p · explicit reference and pressure gauge · views stay closed')

    def request_example(self,identifier):
        if self.is_running() or self.scene.locked or not self.confirm_discard():return False
        self.load_example(identifier);return True

    def load_example(self,identifier):
        config=example_config(identifier);example=EXAMPLE_PROBLEMS[identifier]
        if self.is_running() or self.scene.locked:return False
        self.reset_template();self._resetting=True;self.config=config
        old=next(n for n in self.scene.nodes() if n.kind=='pde')
        interior=next(n for n in self.scene.nodes() if n.kind=='interior')
        method=next(n for n in self.scene.nodes() if n.kind=='method')
        for wire in list(self.scene.wires):
            if wire.source.parentItem() is old or wire.target.parentItem() is old:
                self.scene.wires.remove(wire);self.scene.removeItem(wire)
        self.scene.removeItem(old)
        pde=self.make_node('Predefined PDE','',480,40)
        if config.method in ('self_adaptive','lra'):
            recipe=self_adaptive if config.method=='self_adaptive' else learning_rate_annealing
            title=next(title for kind,title,_,_ in recipe.COMPONENTS if kind==recipe.ROOT)
            old_method=method;method=self.make_node(title,'',old_method.pos().x(),old_method.pos().y())
            for wire in list(self.scene.wires):
                if wire.target.parentItem() is old_method:
                    self.scene.wires.remove(wire);self.scene.removeItem(wire)
                    self.scene.connect_nodes(wire.source.parentItem(),method,wire.target.name)
            self.scene.removeItem(old_method)
        for node in self.scene.nodes():
            if node.kind in PARAMETERS:node.settings['parameters']=parameters_for(node.kind,config)
        self.scene.connect_nodes(interior,pde);self.scene.connect_nodes(pde,method)
        comparison=None
        for index,title in enumerate(('Training Monitor','Solution Plot','Comparison','CSV Export')):
            node=self.add_node(title);node.setPos(1000,index*150);self.scene.connect_nodes(method,node)
            if node.kind=='comparison':comparison=node
        reference=self.add_node('Reference');reference.setPos(730,500)
        reference.settings['reference_expression']=example.reference;self.scene.connect_nodes(reference,comparison)
        self._resetting=False;self.scene.clearSelection();pde.setSelected(True)
        self.settings_changed();self.validate_model();self.fit_model()
        self.undo_stack.clear();self.undo_stack.setClean();self.update_document_title()
        self.statusBar().showMessage(example.label+' · '+example.description+' · views remain closed')
        return True

    def show_quick_insert(self,position,screen_position):
        if self.is_running() or self.scene.locked:return
        if self.quick_insert is not None:self.quick_insert.reject()
        popup=QuickInsert(self)
        self.quick_insert=popup
        popup.blockChosen.connect(lambda title:self.add_node(title,position))
        def finished():
            if self.quick_insert is popup:self.quick_insert=None
            self.view.setFocus()
        popup.finished.connect(finished)
        popup.move(screen_position)
        popup.show()
        screen=popup.screen().availableGeometry()
        popup.move(max(screen.left(),min(screen_position.x(),screen.right()-popup.width()+1)),
                   max(screen.top(),min(screen_position.y(),screen.bottom()-popup.height()+1)))
        popup.library.search.setFocus()

    def replace_recipe_parts(self,text,old_nodes,old_wires,new_nodes,new_wires,selected):
        """One reversible structural edit; retain objects and external block IDs."""
        original_wires=list(self.scene.wires)
        changed_wires=[w for w in original_wires if w not in old_wires]+new_wires
        def swap(remove_nodes,remove_wires,add_nodes,add_wires,selection,order):
            self.scene.cancel_wire();self.scene.blockSignals(True)
            try:
                self.scene.clearSelection()
                for wire in remove_wires:
                    self.scene.wires.remove(wire);self.scene.removeItem(wire)
                for node in remove_nodes:self.scene.removeItem(node)
                for node in add_nodes:self.scene.addItem(node)
                for wire in add_wires:
                    self.scene.addItem(wire);self.scene.wires.append(wire);wire.update_path()
                self.scene.wires[:]=order
                for node in selection:node.setSelected(True)
            finally:self.scene.blockSignals(False)
            self.graph_changed();self.selection_changed()
        self.push_edit(text,
            lambda:swap(old_nodes,old_wires,new_nodes,new_wires,selected,changed_wires),
            lambda:swap(new_nodes,new_wires,old_nodes,old_wires,old_nodes,original_wires))

    def recipe_components(self,parameters,position,recipe=baseline):
        parts=recipe.split(parameters);nodes={}
        for kind,title,offset,_ in recipe.COMPONENTS:
            node=Node(title,'',position.x()+offset[0],position.y()+offset[1])
            node.settings['parameters']=parts[kind];nodes[kind]=node
        wires=[Wire(nodes[source].output,nodes[target].inputs[port])
               for source,target,port in recipe.CONNECTIONS]
        return nodes,wires

    def insert_expanded_recipe(self,recipe=baseline):
        if self.is_running() or self.scene.locked:return False
        position=self.view.mapToScene(self.view.viewport().rect().center())-QPointF(85,40)
        nodes,wires=self.recipe_components(deepcopy(recipe.DEFAULTS),position,recipe)
        self.replace_recipe_parts('Insert expanded '+recipe.NAME,[],[],list(nodes.values()),wires,list(nodes.values()))
        self.statusBar().showMessage(recipe.NAME+' inserted · connect PDE and BC inputs · replace the network through its typed input')
        return nodes[recipe.ROOT]

    def expand_recipe(self,node):
        if self.is_running() or self.scene.locked or node.scene() is not self.scene or node.kind not in COMPACT_RECIPES:return False
        self.commit_inspector()
        recipe=COMPACT_RECIPES[node.kind]
        nodes,wires=self.recipe_components(node.settings['parameters'],node.pos(),recipe)
        root=nodes[recipe.ROOT];root.identifier=node.identifier
        external=[w for w in self.scene.wires if w.source.parentItem() is node or w.target.parentItem() is node]
        for wire in external:
            source=root.output if wire.source.parentItem() is node else wire.source
            target=root.inputs[wire.target.name] if wire.target.parentItem() is node else wire.target
            wires.append(Wire(source,target))
        self.replace_recipe_parts('Expand '+recipe.NAME,[node],external,list(nodes.values()),wires,[root])
        self.statusBar().showMessage(recipe.NAME+' expanded · edit or replace the network · right-click the trainer to collapse')
        return root

    def collapse_recipe(self,node):
        if self.is_running() or self.scene.locked or node.scene() is not self.scene or node.kind not in TRAINER_RECIPES:return False
        self.commit_inspector()
        incoming=[w for w in self.scene.wires if w.target is node.inputs['network']]
        if len(incoming)!=1 or incoming[0].source.parentItem().kind!='network':
            self.statusBar().showMessage('Collapse needs one connected supported neural network.');return False
        internal=incoming[0];network=internal.source.parentItem()
        if any(w.source.parentItem() is network and w is not internal for w in self.scene.wires):
            self.statusBar().showMessage('This network is shared. Keep it expanded or connect a private network before collapsing.');return False
        recipe=TRAINER_RECIPES[node.kind]
        try:parameters=recipe.combine(network.settings['parameters'],node.settings['parameters'])
        except ValueError as error:self.statusBar().showMessage(str(error));return False
        compact=Node(recipe.NAME,'',node.pos().x(),node.pos().y())
        compact.identifier=node.identifier;compact.settings['parameters']=parameters
        old=[w for w in self.scene.wires if w.source.parentItem() in (node,network) or w.target.parentItem() in (node,network)]
        wires=[]
        for wire in old:
            if wire is internal:continue
            source=compact.output if wire.source.parentItem() is node else wire.source
            target=compact.inputs[wire.target.name] if wire.target.parentItem() is node else wire.target
            wires.append(Wire(source,target))
        self.replace_recipe_parts('Collapse '+recipe.NAME,[network,node],old,[compact],wires,[compact])
        self.statusBar().showMessage(recipe.NAME+' collapsed · component settings retained · double-click to expand')
        return compact

    def add_node(self,title,position=None):
        if self.is_running() or self.scene.locked or title not in KINDS:return
        center=position if position is not None else self.view.mapToScene(self.view.viewport().rect().center())
        node=Node(title,'Not configured',center.x()-85,center.y()-40)
        if node.kind in OUTPUT_KINDS:
            node.settings['resolution']=101 if self.config.shape=='Interval' else 41
            node.title=title+' '+str(1+sum(n.kind==node.kind for n in self.scene.nodes()))
        def add():
            self.scene.addItem(node)
            self.scene.clearSelection();node.setSelected(True);self.graph_changed()
        def remove():
            self.scene.removeItem(node);self.graph_changed()
        self.push_edit('Add '+title,add,remove)
        return node

    def delete_selected(self):
        if not self.is_running():self.scene.delete_selection()

    def is_running(self):
        return self.worker is not None

    def clear_run_data(self):
        self.loaded_checkpoint=None
        self.result=None
        self.monitor_records={}
        self.active_monitors=set()
        self.output_views.clear_data()
        if self.last_run is not None:
            self.last_run.model=None
            self.last_run.deleteLater()
            self.last_run=None
        self.update_run_actions()

    def graph_changed(self):
        self.clear_run_data()
        self.output_views.sync(self.scene.nodes())
        self.sync_rectangle_previews()
        self.refresh_descriptions()
        self.progress_bar.setValue(0)
        self.validation.setText('Graph changed. Validate the model before training.')

    def refresh_descriptions(self):
        for node in self.scene.nodes():
            c=self.config_for_node(node)
            actual=int(c.points**.5)**2 if c.shape=='Rectangle' and c.sampling=='grid' else c.points
            details={'domain':f'x: {c.x_min:g}…{c.x_max:g}' if c.shape=='Interval' else f'{c.x_max-c.x_min:g} × {c.y_max-c.y_min:g}',
                     'interior':'Imported PDE points' if c.interior_data else f'{actual} points · {c.sampling}',
                     'boundary':f'{len(c.sides)} sets · {sum(c.boundary_recipe(s).points for s in c.sides) if c.shape=="Rectangle" else 2} points', 'pde':'Steady flow · u, v, p' if c.physics=='incompressible_2d' else c.pde_expression or f'Δu + {c.source:g} = 0',
                     'bc':'Component BCs + p gauge' if c.physics=='incompressible_2d' else ', '.join(c.bc_types[s][0] for s in c.sides),
                     'recipe':f'Baseline · Adam · {c.iterations} steps',
                     'lra_recipe':f'LRA · {c.iterations} steps', 'lra_method':f'Gradient balance · every {c.lra_interval}',
                     'sa_recipe':f'SA-PINN · {c.iterations} steps', 'sa_method':f'Adam min/max · {c.iterations} steps',
                     'network':NETWORK_NAMES[c.network_type]+' · '+' → '.join(map(str,c.layers)), 'method':f'{"Hard BC · " if c.method=="hard_dirichlet" else ""}Adam · {c.iterations} steps',
                     'predefined_pde':PREDEFINED_PDES.get(c.pde_preset,PREDEFINED_PDES['poisson']).label,
                     'solution':'Requested field results' if c.physics=='incompressible_2d' else 'Requested prediction plot', 'monitor':'Requested loss history', 'export':'Explicit CSV data request', 'paraview':'Explicit ParaView data request',
                     'comparison':'Model + explicit reference source', 'reference':'Explicit Couette reference' if c.physics=='incompressible_2d' else 'Explicit Poisson reference'}
            if node.kind=='domain':node.title=c.shape
            if node.kind=='network':node.title='Neural network' if c.network_type=='fcnn' else NETWORK_NAMES[c.network_type]
            node.detail=details[node.kind]
            if node.kind=='reference' and (node.settings.get('reference_expression') or node.settings.get('reference_data')):node.detail='Explicit user reference'
            node.update()
            if node.isSelected():
                self.name.setText(node.title);self.detail.setText(node.detail)

    def parameter_owner(self,attribute):
        kind=OWNERS[attribute]
        selected=[n for n in self.scene.nodes() if n.isSelected() and (role(n.kind)==kind or n.kind in COMPACT_RECIPES) and attribute in PARAMETERS.get(n.kind,())]
        candidates=selected or [n for n in self.scene.nodes() if (role(n.kind)==kind or n.kind in COMPACT_RECIPES) and attribute in PARAMETERS.get(n.kind,())]
        return candidates[0] if candidates else None

    def config_for_node(self,node):
        try:base=assemble(self.scene.nodes(),self.scene.wires,require_all=False)
        except ValueError:base=self.config
        return overlay(base,node.kind,node.settings['parameters']) if node.kind in PARAMETERS else deepcopy(base)

    def change(self,attribute,value,edit_token=None):
        node=self.parameter_owner(attribute)
        old=deepcopy(getattr(self.config_for_node(node),attribute)) if node else deepcopy(getattr(self.config,attribute))
        if old==value:return
        value=deepcopy(value)
        def apply(current):
            if node is not None:node.settings['parameters'][attribute]=encode_parameter(attribute,current)
            setattr(self.config,attribute,deepcopy(current))
            self.settings_changed()
            if edit_token is None or attribute in ('shape','physics'):self.selection_changed()
        merge=(node.identifier if node else None,attribute,edit_token) if edit_token is not None else None
        self.push_edit('Change '+attribute.replace('_',' '),lambda:apply(value),lambda:apply(old),merge)

    def settings_changed(self):
        self.clear_run_data()
        self.progress_bar.setRange(0,self.config.iterations);self.progress_bar.setValue(0)
        self.refresh_descriptions()
        self.sync_rectangle_previews()
        owners={node.identifier:node for node in self.scene.nodes()}
        for (identifier,_),(_,preview) in self.rectangle_previews.items():
            preview.config=self.config_for_node(owners[identifier]);preview.update()
        try:self.config.validate();self.validation.setText('Settings ready. Connections are checked before training.')
        except ValueError as error:self.validation.setText(str(error))
        self.statusBar().showMessage('Settings changed · previous results cleared')

    def edit_point_data(self,side=None,reference=None):
        if self.is_running() or self.scene.locked:return
        source=reference.settings.get('reference_data',{}) if reference is not None else self.config.boundary_data.get(side,{}) if side else self.config.interior_data
        dialog=DataMappingDialog(len(self.config.bounds[0]),source,self,coordinate_only=reference is None and side is None,components=len(self.config.field_names) if reference is not None else 1,allow_values=reference is not None or self.config.physics=='scalar',require_values=reference is not None)
        if dialog.exec()!=DataMappingDialog.DialogCode.Accepted:return
        if reference is not None:self.change_output(reference,'reference_data',dialog.source())
        elif side:
            values=deepcopy(self.config.boundary_data);values[side]=dialog.source();self.change('boundary_data',values)
        else:self.change('interior_data',dialog.source())
        self.selection_changed()

    def clear_point_data(self,side=None,reference=None):
        if reference is not None:self.change_output(reference,'reference_data',{})
        elif side:
            values=deepcopy(self.config.boundary_data);values.pop(side,None);self.change('boundary_data',values)
        else:self.change('interior_data',{})
        self.selection_changed()

    def property_section(self,title):
        heading=QLabel(title);heading.setObjectName('section_'+title.lower().replace(' ','_'))
        font=heading.font();font.setBold(True);heading.setFont(font)
        heading.setContentsMargins(0,10,0,2);self.form.addRow(heading)

    def field(self,label,attribute,kind='float',choices=None,minimum=-1000000,maximum=1000000):
        value=getattr(self.config,attribute)
        token=object()
        if kind=='choice':
            widget=QComboBox();widget.addItems(choices);widget.setCurrentText(str(value))
            if attribute=='sampling':
                for i,method in enumerate(choices):widget.setItemData(i,SAMPLERS[method],Qt.ItemDataRole.ToolTipRole)
            widget.currentTextChanged.connect(lambda text:self.change(attribute,text,token))
        elif kind=='int':
            widget=QSpinBox();widget.setKeyboardTracking(False);widget.setRange(int(minimum),int(maximum));widget.setValue(value)
            widget.valueChanged.connect(lambda number:self.change(attribute,number,token))
        else:
            widget=QDoubleSpinBox();widget.setKeyboardTracking(False);widget.setDecimals(6);widget.setRange(minimum,maximum);widget.setValue(value)
            widget.valueChanged.connect(lambda number:self.change(attribute,number,token))
        self.form.addRow(label,widget)
        return widget

    @Slot()
    def selection_changed(self):
        if not isValid(self.scene):
            return
        nodes=[i for i in self.scene.selectedItems() if isinstance(i,Node)]
        node=nodes[0] if len(nodes)==1 else None
        if node is not None:self.config=self.config_for_node(node)
        self.name.setText(node.title if node else ('Multiple blocks selected' if len(nodes)>1 else ''))
        self.update_clipboard_actions()
        self.open_output_action.setEnabled(node is not None and node.kind in OUTPUT_KINDS)
        self.detail.setText(node.detail if node else '')
        body=QWidget();self.form=QFormLayout(body)
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        if node is not None:
            capability=QLabel(describe_block(node.kind));capability.setWordWrap(True);self.form.addRow(capability)
        if node is None:
            self.form.addRow(QLabel('Select a block to edit its settings.'))
        elif node.kind=='domain':
            self.field('Domain','shape','choice',choices=['Interval','Rectangle'])
            self.field('x minimum','x_min');self.field('x maximum','x_max')
            if self.config.shape=='Rectangle':
                self.field('y minimum','y_min');self.field('y maximum','y_max')
            note=QLabel('Built-in meshless geometry. No external file or mesh is required.');note.setWordWrap(True);self.form.addRow(note)
        elif node.kind=='interior':
            imported=QPushButton('Map / preview prepared CSV or NPZ…');imported.clicked.connect(lambda:self.edit_point_data());self.form.addRow(imported)
            if self.config.interior_data:
                label=QLabel(self.config.interior_data['path']);label.setWordWrap(True);self.form.addRow('Imported source',label)
                clear=QPushButton('Use built-in sampler');clear.clicked.connect(lambda:self.clear_point_data());self.form.addRow(clear)
            self.field('Sampler','sampling','choice',choices=list(SAMPLERS))
            self.field('Point budget','points','int',minimum=1,maximum=100000)
            self.field('Seed','seed','int',minimum=0,maximum=2147483647)
            note=QLabel('Rectangle grids use complete layers and may return fewer points than the budget. Actual counts appear on the block and preview.');note.setWordWrap(True);self.form.addRow(note)
        elif node.kind=='boundary':
            if self.config.shape=='Rectangle':
                side_choice=QComboBox();side_choice.addItems(list(self.config.sides))
                side=getattr(self,'boundary_side','left')
                side_choice.setCurrentText(side);side_choice.setObjectName('boundary_side')
                side_choice.currentTextChanged.connect(self.select_boundary_side)
                self.form.addRow('Named point set',side_choice)
                imported=QPushButton('Map / preview this set from CSV or NPZ…');imported.clicked.connect(lambda checked=False,s=side:self.edit_point_data(s));self.form.addRow(imported)
                if self.config.boundary_data.get(side):
                    label=QLabel(self.config.boundary_data[side]['path']);label.setWordWrap(True);self.form.addRow('Imported source',label)
                    clear=QPushButton('Use built-in side sampler');clear.clicked.connect(lambda checked=False,s=side:self.clear_point_data(s));self.form.addRow(clear)
                recipe=self.config.boundary_recipe(side)
                method=QComboBox();method.addItems(list(SAMPLERS));method.setCurrentText(recipe.method)
                method.setObjectName('boundary_method')
                for i,key in enumerate(SAMPLERS):method.setItemData(i,SAMPLERS[key],Qt.ItemDataRole.ToolTipRole)
                method.currentTextChanged.connect(lambda value,s=side,t=object():self.change_boundary_sampling(s,'method',value,t))
                self.form.addRow('Sampler',method)
                count=QSpinBox();count.setKeyboardTracking(False);count.setRange(1,10000);count.setValue(recipe.points)
                count.setObjectName('boundary_count')
                count.valueChanged.connect(lambda value,s=side,t=object():self.change_boundary_sampling(s,'points',value,t))
                self.form.addRow('Points on this side',count)
                seed=QSpinBox();seed.setKeyboardTracking(False);seed.setRange(0,2147483647);seed.setValue(recipe.seed)
                seed.setObjectName('boundary_seed')
                seed.valueChanged.connect(lambda value,s=side,t=object():self.change_boundary_sampling(s,'seed',value,t))
                self.form.addRow('Seed',seed)
                note=QLabel('Each side has an independent recipe. All implemented side methods return exactly the selected count on that side. Grid points exclude corners. The seed is retained but unused for a grid. Interior settings do not change boundary recipes.')
            else:
                note=QLabel('Interval point sets are the left and right endpoints: one point each. Rectangle-side recipes are retained but do not apply to an interval.')
                for side in self.config.sides:
                    imported=QPushButton('Map '+side+' prepared point set…');imported.clicked.connect(lambda checked=False,s=side:self.edit_point_data(s));self.form.addRow(imported)
            note.setWordWrap(True);self.form.addRow(note)
            note=QLabel('Named point sets: '+', '.join(self.config.sides)+'. Types and targets are assigned later in the Boundary condition block. Open Preview samples from the Rectangle block to inspect the combined samples.')
            note.setWordWrap(True);self.form.addRow(note)
        elif node.kind=='predefined_pde':
            choices=QComboBox()
            for identifier,preset in PREDEFINED_PDES.items():choices.addItem(preset.label,identifier)
            choices.setCurrentIndex(choices.findData(self.config.pde_preset));choices.setObjectName('predefined_pde_choice')
            choices.currentIndexChanged.connect(lambda index,w=choices:self.change('pde_preset',w.itemData(index)))
            self.form.addRow('Predefined PDE',choices)
            preset=PREDEFINED_PDES[self.config.pde_preset]
            equation=QLabel(preset.equation);equation.setWordWrap(True);self.form.addRow(equation)
            note=QLabel('Domains: '+', '.join(preset.shapes)+'. Boundary conditions and reference values are configured separately. Use PDE residual for editable equations.');note.setWordWrap(True);self.form.addRow(note)
            if self.config.pde_preset!='incompressible_2d':
                self.field('Squared wavenumber k²' if self.config.pde_preset=='helmholtz' else 'Coefficient c','source')
                forcing=QLineEdit(self.config.pde_forcing);forcing.setMaxLength(384);forcing.setObjectName('predefined_forcing')
                forcing.editingFinished.connect(lambda e=forcing:self.change('pde_forcing',e.text().strip()))
                self.form.addRow('Forcing f',forcing)
            if self.config.pde_preset in ('advection_diffusion_1d','incompressible_2d'):
                self.field('Diffusivity / viscosity ν','viscosity',minimum=.000001,maximum=100000)
        elif node.kind=='pde':
            physics=QComboBox();physics.addItem('Scalar residual','scalar');physics.addItem('Steady incompressible 2D flow','incompressible_2d')
            physics.setCurrentIndex(max(0,physics.findData(self.config.physics)))
            physics.currentIndexChanged.connect(lambda index,w=physics:self.change('physics',w.itemData(index)))
            self.form.addRow('Physics',physics)
            if self.config.physics=='incompressible_2d':
                self.form.addRow(QLabel('Fields: u, v, p · steady, density-normalised pressure\nMomentum x / y and incompressibility; no physical time axis.'))
                self.field('Kinematic viscosity ν','viscosity',minimum=.000001,maximum=100000)
                self.field('Optional coefficient c','source')
                for index,label in enumerate(('Momentum x = 0','Momentum y = 0','Continuity = 0')):
                    edit=QLineEdit(self.config.flow_residuals[index]);edit.setObjectName('flow_residual_'+str(index))
                    edit.editingFinished.connect(lambda i=index,e=edit:self.change_flow_residual(i,e.text().strip()))
                    self.form.addRow(label,edit)
            else:
                note=QLabel('Poisson preset: Δu + c = 0\nUnknown: scalar field u\n1D: uₓₓ + c = 0\n2D: uₓₓ + uᵧᵧ + c = 0');note.setWordWrap(True);self.form.addRow(note)
                self.field('Constant c','source')
                edit=QLineEdit(self.config.pde_expression);edit.setObjectName('pde_expression')
                edit.setPlaceholderText('Blank: Poisson preset; e.g. laplace(u) + c')
                edit.editingFinished.connect(lambda e=edit:self.change('pde_expression',e.text().strip()))
                self.form.addRow('Residual = 0',edit)
                note=QLabel('Scalar expressions: x, y (rectangle), u, c, pi; D(u,x), laplace(u), sin/cos/exp/tanh/sqrt/abs. Arithmetic only; no Python code. Additional pointwise PDE inputs remain future work.');note.setWordWrap(True);self.form.addRow(note)
        elif node.kind=='bc':
            if self.config.physics=='incompressible_2d':
                side=getattr(self,'flow_bc_side','left');choice=QComboBox();choice.addItems(list(self.config.sides));choice.setCurrentText(side)
                choice.currentTextChanged.connect(self.select_flow_side);self.form.addRow('Boundary set',choice)
                for field in ('u','v','p'):
                    spec=self.config.flow_boundary[side].get(field,{'type':'None','target':'0','operator':''})
                    kind=QComboBox();kind.addItems(['None','Dirichlet','Neumann','Custom']);kind.setCurrentText(spec['type'])
                    kind.currentTextChanged.connect(lambda value,s=side,f=field:self.change_flow_boundary(s,f,'type',value));self.form.addRow(field+' type',kind)
                    for key,label in [('target',' target expression'),('operator',' Custom operator')]:
                        edit=QLineEdit(spec[key]);edit.setObjectName('flow_'+side+'_'+field+'_'+key)
                        edit.editingFinished.connect(lambda s=side,f=field,k=key,e=edit:self.change_flow_boundary(s,f,k,e.text().strip()))
                        self.form.addRow(field+label,edit)
                for index,label in enumerate(('Gauge x','Gauge y')):
                    spin=QDoubleSpinBox();spin.setDecimals(6);spin.setRange(-1000000,1000000);spin.setValue(self.config.pressure_point[index]);spin.setKeyboardTracking(False)
                    spin.valueChanged.connect(lambda value,i=index:self.change_pressure_point(i,value));self.form.addRow(label,spin)
                self.field('Pressure gauge p','pressure_value');self.field('Gauge loss weight','pressure_weight',minimum=.000001,maximum=100000)
                note=QLabel('Each velocity component needs a Dirichlet anchor. Pressure has one explicit point gauge. None leaves a component unconstrained on this set. Target expressions use coordinates; Custom operators can use u, v, p and dn(...).');note.setWordWrap(True);self.form.addRow(note)
            else:
                for side in self.config.sides:
                    row=QWidget();layout=QHBoxLayout(row);layout.setContentsMargins(0,0,0,0)
                    combo=QComboBox();combo.addItems(['Dirichlet','Neumann','Custom']);combo.setCurrentText(self.config.bc_types[side])
                    value=QDoubleSpinBox();value.setKeyboardTracking(False);value.setDecimals(5);value.setRange(-100000,100000);value.setValue(self.config.bc_values[side])
                    combo.currentTextChanged.connect(lambda text,s=side,t=object():self.change_bc(s,'type',text,t))
                    value.valueChanged.connect(lambda number,s=side,t=object():self.change_bc(s,'value',number,t))
                    layout.addWidget(combo);layout.addWidget(value);self.form.addRow(side.capitalize(),row)
                    target=QLineEdit(self.config.bc_targets.get(side,''));target.setObjectName(side+'_target_expression')
                    target.setPlaceholderText('Blank: constant target above; e.g. x*(1-x)')
                    target.editingFinished.connect(lambda s=side,e=target:self.change_bc_expression(s,'bc_targets',e.text().strip()))
                    self.form.addRow(side+' target expression',target)
                    operator=QLineEdit(self.config.bc_operators.get(side,''));operator.setObjectName(side+'_operator_expression')
                    operator.setPlaceholderText('Custom only: e.g. u + dn(u)')
                    operator.editingFinished.connect(lambda s=side,e=operator:self.change_bc_expression(s,'bc_operators',e.text().strip()))
                    self.form.addRow(side+' custom operator',operator)
                note=QLabel('Dirichlet: u = target.\nNeumann: ∂u/∂n = target (outward normal), not conductivity-scaled flux.\nAdjacent Dirichlet values must agree at shared rectangle corners. At least one Dirichlet side is required.');note.setWordWrap(True);self.form.addRow(note)
        elif node.kind=='network':
            architecture=QComboBox();architecture.setObjectName('network_architecture')
            for identifier,label in NETWORK_NAMES.items():architecture.addItem(label,identifier)
            architecture.setCurrentIndex(architecture.findData(self.config.network_type))
            architecture.currentIndexChanged.connect(lambda index,w=architecture:self.change_network_type(w.itemData(index)))
            self.form.addRow('Architecture',architecture)
            note=QLabel('Pointwise coordinates → fields · fixed domain scaling. No neural operators. Changing architecture resets its options; trainer settings stay separate.');note.setWordWrap(True);self.form.addRow(note)
            self.network_option_fields()
            edit=QLineEdit(', '.join(map(str,self.config.layers)))
            edit.setPlaceholderText('16, 16')
            token=object()
            edit.textChanged.connect(lambda text:self.change_layers(text,token))
            self.form.addRow('Hidden layers',edit)
            if len(self.config.field_names)>1:self.form.addRow(QLabel('Coordinates: x, y · outputs: u, v, p'))
            self.field('Network seed','network_seed','int',minimum=0,maximum=2147483647)
        elif node.kind in COMPACT_RECIPES:
            recipe=COMPACT_RECIPES[node.kind]
            note=QLabel(recipe.NAME+' · full-batch Adam\nNetwork and loss settings are retained inside. Double-click or expand to edit the components.');note.setWordWrap(True);self.form.addRow(note)
            self.property_section('PINN method')
            self.form.addRow(QLabel(recipe.NAME))
            for attribute in recipe.EXPOSED:
                if attribute=='lra_alpha':self.field('EMA new-estimate fraction','lra_alpha',minimum=.001,maximum=1)
                elif attribute=='lra_interval':self.field('Balance every (updates)','lra_interval','int',minimum=1,maximum=100000)
            self.property_section('Optimisation')
            self.form.addRow('Network optimiser',QLabel('Adam · descent'))
            self.field('Adam learning rate','lr',minimum=.000001,maximum=1)
            if 'sa_lr' in recipe.EXPOSED:
                self.form.addRow('Mask optimiser',QLabel('Adam · ascent'))
                self.field('Mask Adam learning rate','sa_lr',minimum=.000001,maximum=1)
            self.property_section('Trainer')
            self.field('Training updates','iterations','int',minimum=1,maximum=100000)
            self.field('Device','device','choice',choices=['cpu','auto','cuda'])
            button=QPushButton('Expand into components');button.clicked.connect(lambda checked=False,n=node:self.expand_recipe(n));self.form.addRow(button)
        elif node.kind in TRAINER_RECIPES:
            self.property_section('PINN method')
            variant=QComboBox()
            for key,label in [('baseline','Baseline PINN'),('hard_dirichlet','Hard Dirichlet PINN')]:variant.addItem(label,key)
            if node.kind=='sa_method':variant.addItem('Self-adaptive PINN','self_adaptive');variant.setEnabled(False)
            if node.kind=='lra_method':variant.addItem('Learning-rate annealing PINN','lra');variant.setEnabled(False)
            variant.setCurrentIndex(max(0,variant.findData(self.config.method)))
            if self.config.physics=='incompressible_2d':variant.model().item(1).setEnabled(False)
            variant.currentIndexChanged.connect(lambda index,v=variant:self.change('method',v.itemData(index)))
            self.form.addRow('Method variant',variant)
            if node.kind=='lra_method':
                self.form.addRow(QLabel('LRA · max PDE / mean constraint gradients\nScalar loss balancing; no base-rate decay'))
                self.field('EMA new-estimate fraction','lra_alpha',minimum=.001,maximum=1)
                self.field('Balance every (updates)','lra_interval','int',minimum=1,maximum=100000)
                self.field('Coefficient cap','lra_max',minimum=1,maximum=1e12)
                epsilon=QLineEdit(format(self.config.lra_epsilon,'.6g'));epsilon.setObjectName('lra_epsilon')
                epsilon.editingFinished.connect(lambda w=epsilon:self.change_lra_epsilon(w.text()))
                self.form.addRow('Gradient floor',epsilon)
                note=QLabel('One scalar per constraint term. Alpha weights the new estimate. Raw gradient denominator, before LRA weighting; tiny constraint gradients hold the prior coefficient. Existing architecture is unchanged.');note.setWordWrap(True);self.form.addRow(note)
            if node.kind=='sa_method':
                self.form.addRow(QLabel('Self-adaptive PINN · squared per-point masks\nNetwork Adam descent / mask Adam ascent'))
                self.field('Initial mask weight','sa_initial',minimum=.000001,maximum=1000000)
                self.field('Maximum mask weight','sa_max',minimum=.000001,maximum=1000000)
                adaptive=QCheckBox('Adapt boundary/constraint weights');adaptive.setChecked(self.config.sa_boundary)
                adaptive.toggled.connect(lambda value:self.change('sa_boundary',value));self.form.addRow(adaptive)
                note=QLabel('PDE weights always adapt. Fixed points, squared mask; no L-BFGS/resampling. Individual terms show raw MSE; total uses adaptive masks and may increase.');note.setWordWrap(True);self.form.addRow(note)
            self.field('PDE weight','pde_weight',minimum=.000001,maximum=100000)
            self.field('Mean BC weight','boundary_weight',minimum=.000001,maximum=100000)
            self.property_section('Optimisation')
            self.form.addRow('Network optimiser',QLabel('Adam · descent'))
            self.field('Adam learning rate','lr',minimum=.000001,maximum=1)
            if node.kind=='sa_method':
                self.form.addRow('Mask optimiser',QLabel('Adam · ascent'))
                self.field('Mask Adam learning rate','sa_lr',minimum=.000001,maximum=1)
            self.property_section('Trainer')
            self.form.addRow(QLabel('Fixed samples · full-batch updates'))
            self.field('Training updates','iterations','int',minimum=1,maximum=100000)
            self.field('Device','device','choice',choices=['cpu','auto','cuda'])
            self.form.addRow(QLabel('Stop requests are checked every 25 updates.'))
        elif node.kind in OUTPUT_KINDS:
            name=QLineEdit(node.title)
            name.editingFinished.connect(lambda n=node,e=name:self.rename_output(n,e.text()))
            self.form.addRow('Name',name)
            if node.kind=='monitor':
                count=QSpinBox();count.setKeyboardTracking(False);count.setRange(25,100000);count.setValue(node.settings['history_limit'])
                count.valueChanged.connect(lambda value,n=node,t=object():self.change_output(n,'history_limit',value,t))
                self.form.addRow('Retained updates',count)
                text='Connect PINN method → monitor to record losses. Closing the view does not stop recording. Removing the block removes the request.'
            else:
                count=QSpinBox();count.setKeyboardTracking(False);count.setRange(2,201);count.setValue(node.settings['resolution'])
                count.valueChanged.connect(lambda value,n=node,t=object():self.change_output(n,'resolution',value,t))
                self.form.addRow('Evaluation points / axis',count)
                text='Connect PINN method to this plot. No reference or error metric is assumed.' if node.kind=='solution' else 'Connect PINN method to the upper input and Reference to the lower input. The rectangle reference requires equal constant Dirichlet edges and uses a truncated sine series.'
            if node.kind=='paraview':text='Connect a trainer or recipe and train. Export predictions as VTI on a regular evaluation grid; training samples can be random. Flow includes a velocity vector. Current steady models export a single snapshot. Choose a filename when exporting.'
            if node.kind=='export':text='Connect PINN method, train, then explicitly export scalar x/y/u rows. No plot or file is created automatically.'
            note=QLabel(text);note.setWordWrap(True);self.form.addRow(note)
            button=QPushButton('Export CSV…' if node.kind=='export' else 'Export ParaView…' if node.kind=='paraview' else 'Open view');button.clicked.connect(lambda checked=False,n=node:self.open_block(n))
            self.form.addRow(button)
        elif node.kind=='reference':
            expression=QLineEdit(node.settings.get('reference_expression',''))
            expression.setPlaceholderText('Blank: preset; e.g. y; 0; 0' if self.config.physics=='incompressible_2d' else 'Blank: preset; e.g. x*(1-x)')
            expression.editingFinished.connect(lambda n=node,e=expression:self.change_output(n,'reference_expression',e.text().strip()))
            self.form.addRow('Reference expression',expression)
            imported=QPushButton('Map / preview imported reference values…');imported.clicked.connect(lambda checked=False,n=node:self.edit_point_data(reference=n));self.form.addRow(imported)
            if node.settings.get('reference_data'):
                clear=QPushButton('Clear imported reference');clear.clicked.connect(lambda checked=False,n=node:self.clear_point_data(reference=n));self.form.addRow(clear)
            note=QLabel('Explicit Poisson reference: analytical for intervals; 40 odd sine modes per axis for rectangles with equal constant Dirichlet edges. The rectangle reference is truncated, not an exact closed-form value. Connect it to Comparison; ordinary plots never evaluate it.');note.setWordWrap(True);self.form.addRow(note)
        old=self.property_scroll.takeWidget()
        if old is not None:old.deleteLater()
        self.property_scroll.setWidget(body)
        self.property_scroll.setEnabled(not self.is_running())

    def select_boundary_side(self,side):
        self.boundary_side=side
        self.selection_changed()

    def change_boundary_sampling(self,side,attribute,value,edit_token=None):
        recipe=self.config.boundary_recipe(side)
        updated=replace(recipe,**{attribute:value})
        if recipe==updated:return
        recipes=deepcopy(self.config.boundary_sampling);recipes[side]=updated
        self.change('boundary_sampling',recipes,edit_token)

    def change_flow_residual(self,index,text):
        values=list(self.config.flow_residuals);values[index]=text;self.change('flow_residuals',tuple(values))

    def select_flow_side(self,side):
        self.flow_bc_side=side;self.selection_changed()

    def change_flow_boundary(self,side,field,key,value):
        values=deepcopy(self.config.flow_boundary)
        values[side].setdefault(field,{'type':'None','target':'0','operator':''})[key]=value
        self.change('flow_boundary',values)

    def change_pressure_point(self,index,value):
        point=list(self.config.pressure_point);point[index]=value;self.change('pressure_point',tuple(point))

    def change_bc_expression(self,side,attribute,text):
        values=deepcopy(getattr(self.config,attribute))
        if text:values[side]=text
        else:values.pop(side,None)
        self.change(attribute,values)

    def change_bc(self,side,kind,value,edit_token=None):
        attribute='bc_types' if kind=='type' else 'bc_values'
        values=deepcopy(getattr(self.config,attribute));values[side]=value
        self.change(attribute,values,edit_token)

    def change_network_type(self,kind):
        if self.is_running() or self.scene.locked or kind not in NETWORK_NAMES:return
        node=self.parameter_owner('network_type')
        if node is None or node.settings['parameters']['network_type']==kind:return
        old=deepcopy(node.settings['parameters']);new=dict(old,network_type=kind,network_options={})
        def apply(parameters):
            node.settings['parameters']=deepcopy(parameters);self.config=self.config_for_node(node)
            self.settings_changed();self.selection_changed()
        self.push_edit('Change network architecture',lambda:apply(new),lambda:apply(old))

    def change_network_option(self,key,value,token=None):
        options=deepcopy(self.config.network_options);options[key]=value
        try:network_options(self.config.network_type,options)
        except ValueError as error:self.validation.setText(str(error));return
        self.change('network_options',options,token)

    def change_network_list(self,key,text):
        try:values=[float(value.strip()) for value in text.split(',')]
        except ValueError:self.validation.setText('Enter a comma-separated list of finite numbers.');return
        self.change_network_option(key,values)

    def network_option_fields(self):
        resolved=network_options(self.config.network_type,self.config.network_options)
        for key,spec in OPTION_SCHEMAS[self.config.network_type].items():
            if key=='features' and 'encoding' in resolved and resolved['encoding']!='gaussian':continue
            if key=='frequencies' and resolved.get('encoding')=='gaussian':continue
            value=resolved[key];kind=spec['kind'];token=object()
            if kind=='choice':
                widget=QComboBox();widget.addItems(spec['choices']);widget.setCurrentText(value)
                widget.currentTextChanged.connect(lambda v,k=key:self.change_network_option(k,v))
            elif kind=='bool':
                widget=QCheckBox();widget.setChecked(value)
                widget.toggled.connect(lambda v,k=key,t=token:self.change_network_option(k,v,t))
            elif kind=='list':
                widget=QLineEdit(', '.join(map(str,value)))
                widget.editingFinished.connect(lambda k=key,w=widget:self.change_network_list(k,w.text()))
            else:
                widget=QSpinBox() if kind=='int' else QDoubleSpinBox()
                if kind!='int':widget.setDecimals(6)
                widget.setKeyboardTracking(False);widget.setRange(spec['minimum'],spec['maximum']);widget.setValue(value)
                widget.valueChanged.connect(lambda v,k=key,t=token:self.change_network_option(k,v,t))
            widget.setObjectName('network_option_'+key);self.form.addRow(spec['label'],widget)

    def change_lra_epsilon(self,text):
        try:value=float(text)
        except ValueError:self.validation.setText('Enter a positive gradient floor, for example 1e-12.');return
        if not 0<value<=1:self.validation.setText('Gradient floor must be positive and at most one.');return
        self.change('lra_epsilon',value)

    def change_layers(self,text,edit_token=None):
        try:layers=tuple(int(n.strip()) for n in text.split(','))
        except ValueError:layers=()
        self.change('layers',layers,edit_token)

    def connection_rejected(self,message):
        self.validation.setText(message)
        self.statusBar().showMessage(message)
        self.log.appendPlainText('Connection: '+message)

    def validate_model(self):
        try:
            self.config=self.scene.compile_config()
            self.config.validate();requests=self.scene.output_requests(self.config)
        except ValueError as error:
            message=str(error);self.validation.setText(message)
            self.log.appendPlainText('Validation: '+message)
            self.statusBar().showMessage('Model needs attention · see Properties or Messages')
            return False
        self.validation.setText(f'Valid PINN graph ({", ".join(self.config.field_names)}). {len(requests)} connected output request(s).')
        self.statusBar().showMessage('Valid · PINN graph fields: '+', '.join(self.config.field_names))
        return True

    def set_busy(self,busy):
        if busy:
            self.scene.cancel_wire()
            if self.quick_insert is not None:self.quick_insert.reject()
        for control in [self.train_button,self.validate_button,self.library,self.property_scroll]:control.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.load_action.setEnabled(not busy);self.heat_action.setEnabled(not busy);self.delete_action.setEnabled(not busy)
        self.flow_action.setEnabled(not busy)
        self.examples_menu.setEnabled(not busy)
        self.recipe_menu.setEnabled(not busy)
        self.open_model_action.setEnabled(not busy);self.save_model_action.setEnabled(not busy);self.save_as_action.setEnabled(not busy)
        self.scene.locked=busy
        for node in self.scene.nodes():
            node.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable,not busy)
        self.run_action.setEnabled(not busy);self.check_action.setEnabled(not busy)
        self.stop_action.setEnabled(busy)
        self.update_history_actions();self.update_clipboard_actions();self.update_run_actions()

    def start_training(self,*,checkpoint=None):
        if self.is_running() or not self.validate_model():return False
        self._history_epoch+=1
        self.clear_run_data()
        requests=self.scene.output_requests(self.config)
        self.active_monitors={r.identifier for r in requests if r.kind=='monitor'}
        self.monitor_records={node.identifier:list(enumerate(checkpoint['loss_history'],1))[-node.settings['history_limit']:] if checkpoint is not None else [] for node in self.scene.nodes() if node.identifier in self.active_monitors}
        self.progress_bar.setRange(0,self.config.iterations);self.progress_bar.setValue(0)
        self.set_busy(True)
        self.worker=TrainingWorker(deepcopy(self.config),requests,parent=self,checkpoint=checkpoint)
        self.worker.progress.connect(self.training_progress)
        self.worker.completed.connect(self.training_completed)
        self.worker.failed.connect(self.training_failed)
        self.worker.finished.connect(self.training_finished)
        self.log.appendPlainText(f'Training {self.config.shape} · {self.config.iterations} updates · {self.config.device}')
        self.statusBar().showMessage('Preparing samples and network…')
        self.worker.start()
        return True

    def training_progress(self,step,records):
        self.progress_bar.setValue(step)
        indexed=list(zip(range(step-len(records)+1,step+1),records))
        for node in self.scene.nodes():
            if node.identifier in self.active_monitors:
                history=self.monitor_records[node.identifier]
                history.extend(indexed)
                del history[:-node.settings['history_limit']]
                self.output_views.refresh(node)
        self.statusBar().showMessage(f"Training {step}/{self.config.iterations} · loss {records[-1]['total']:.5g}")

    def training_completed(self,result):
        self.result=result
        for node in self.scene.nodes():self.output_views.refresh(node)
        status='Stopped' if result['cancelled'] else 'Completed'
        message=f"{status} · {result['steps']} updates · {result['device']}"
        self.statusBar().showMessage(message);self.log.appendPlainText(message)

    def training_failed(self,message):
        self.result=None
        self.statusBar().showMessage('Training failed · see Messages')
        self.log.appendPlainText('Training failed: '+message)
        self.messages_dock.show();self.messages_dock.raise_()

    def training_finished(self):
        worker=self.worker;self.worker=None
        self.set_busy(False)
        if worker is not None:
            if self.result is not None:self.last_run=worker
            else:worker.deleteLater()
        self.update_run_actions()
        if self._closing:QTimer.singleShot(0,self.close)

    def cancel_training(self):
        if self.worker is not None:
            self.worker.cancel();self.cancel_button.setEnabled(False)
            self.statusBar().showMessage('Stopping after the current update batch…')

    def apply_theme(self):
        if not hasattr(self,'scene'):return
        for name,action in getattr(self,'theme_actions',{}).items():
            action.setChecked(name==theme_name())
        c=colors();self.scene.setBackgroundBrush(QColor(c['canvas']))
        for item in self.scene.items():
            if isinstance(item,Node):item.setBrush(QColor(c['base']));item.setPen(QPen(QColor(c['border']),1.5))
            elif isinstance(item,Port):item.setBrush(QColor(c['highlight']));item.setPen(QPen(QColor(c['base']),2))
            elif isinstance(item,Wire):item.setPen(QPen(QColor(c['highlight']),2))
        self.scene.update()
        for widget in self.findChildren(QWidget):widget.update()

    def closeEvent(self,event):
        if self.is_running():
            self._closing=True;self.cancel_training();event.ignore();return
        if not self.confirm_discard():
            self._closing=False;event.ignore();return
        if self.desktop_state is not None:self.desktop_state.close()
        if not getattr(self,'_selection_disconnected',False):
            self.scene.selectionChanged.disconnect(self.selection_changed)
            self._selection_disconnected=True
        super().closeEvent(event)
