"""Block-based baseline Poisson workflow with background training."""
from uuid import uuid4
from copy import deepcopy
from dataclasses import replace
from ..templates.poisson import PoissonConfig, make_poisson_reference
from ..templates.heat import heat_demo_config
from ..components.outputs import OutputRequest
from ..components.sampling import SAMPLERS
from .plots import RectanglePreview
from .output_views import OutputViews, OUTPUT_KINDS
from .training import TrainingWorker
from .canvas_view import CanvasView
from .block_library import BlockLibrary, QuickInsert
from .block_catalogue import KINDS
from .theme import colors, configure_theme, theme_name
from PySide6.QtCore import Qt, QPointF, QTimer, Signal, Slot
from shiboken6 import isValid
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QActionGroup
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QGraphicsView,
    QGraphicsScene, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPathItem,
    QGraphicsItem, QWidget, QVBoxLayout, QLabel, QFormLayout, QLineEdit,
    QMenu, QPlainTextEdit, QScrollArea, QComboBox, QDoubleSpinBox,
    QSpinBox, QPushButton, QProgressBar, QHBoxLayout, QMessageBox,
)


class Port(QGraphicsEllipseItem):
    def __init__(self, node, output, name="input", y=40):
        super().__init__(-6, -6, 12, 12, node)
        self.output = output
        self.name = name
        self.setPos(170 if output else 0, y)
        self.setBrush(QColor(colors()['highlight']))
        self.setPen(QPen(QColor(colors()['base']), 2))
        self.setToolTip('Output: drag to an input' if output else name)
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
        self.setBrush(QColor(colors()['base']))
        self.setPen(QPen(QColor(colors()['border']), 1.5))
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
                      QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
                      QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.inputs = {"input": Port(self, False)}
        if self.kind == 'comparison':
            self.inputs['input'].name = 'model'
            self.inputs['input'].setToolTip('Trained model')
            self.inputs['input'].setY(28)
            self.inputs = {'model': self.inputs['input'], 'reference': Port(self, False, 'reference', 60)}
        self.input = next(iter(self.inputs.values()))
        self.output = Port(self, True, 'output')
        if self.kind == 'reference':self.input.hide()
        if self.kind in OUTPUT_KINDS:self.output.hide()
        self.setPos(x, y)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        painter.setPen(QColor(colors()['text']))
        painter.drawText(14, 28, self.title)
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

    def __init__(self):
        super().__init__()
        self.wires, self.pending = [], None
        self.locked = False
        self.setSceneRect(-5000, -5000, 10000, 10000)
        self.setBackgroundBrush(QColor(colors()['canvas']))

    def begin_wire(self, source):
        self.pending = Wire(source)
        self.addItem(self.pending)

    def finish_wire(self, position):
        if not self.pending:
            return
        targets = [i for i in self.items(position) if isinstance(i, Port) and not i.output]
        wire = self.pending
        if targets and targets[0].parentItem() is not wire.source.parentItem():
            wire.target = targets[0]
            if not any(w.source is wire.source and w.target is wire.target for w in self.wires):
                self.wires.append(wire)
                wire.update_path()
            else:
                self.removeItem(wire)
        else:
            self.removeItem(wire)
        self.pending = None
        self.graphChanged.emit()

    def connect_nodes(self, source, target, input_name=None):
        if input_name is None:
            input_name = ('reference' if source.kind == 'reference' else 'model') if target.kind == 'comparison' else 'input'
        wire = Wire(source.output, target.inputs[input_name])
        self.addItem(wire)
        self.wires.append(wire)
        self.graphChanged.emit()

    def nodes(self):
        return [item for item in self.items() if isinstance(item, Node)]

    def validate_graph(self):
        nodes = self.nodes()
        for kind in CORE_KINDS:
            if sum(n.kind == kind for n in nodes) != 1:
                raise ValueError(f'The Poisson template needs exactly one {kind} block.')
        core_edges = set()
        destinations = set()
        for wire in self.wires:
            source,target = wire.source.parentItem(),wire.target.parentItem()
            edge = (source.kind,target.kind)
            if source.kind in CORE_KINDS and target.kind in CORE_KINDS:
                if edge not in REQUIRED_EDGES or edge in core_edges:
                    raise ValueError('Unsupported or duplicate core connection.')
                core_edges.add(edge)
            else:
                allowed = source.kind == 'method' and target.kind in OUTPUT_KINDS
                allowed = allowed and wire.target.name == ('model' if target.kind=='comparison' else 'input')
                allowed = allowed or (source.kind=='reference' and target.kind=='comparison' and wire.target.name=='reference')
                if not allowed:raise ValueError('Unsupported output connection: '+source.title+' → '+target.title)
                key=(target.identifier,wire.target.name)
                if key in destinations:raise ValueError('Each output input accepts only one source.')
                destinations.add(key)
        missing=REQUIRED_EDGES-core_edges
        if missing:
            raise ValueError('Missing connections: '+', '.join(f'{a} → {b}' for a,b in sorted(missing)))

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
                reference=make_poisson_reference(config)
            if node.kind=='monitor':
                if not 25 <= node.settings['history_limit'] <= 100000:
                    raise ValueError('Monitor history limit must be between 25 and 100000.')
            elif not 2 <= node.settings['resolution'] <= 201:
                raise ValueError('Evaluation resolution must be between 2 and 201.')
            requests.append(OutputRequest(node.identifier,node.kind,node.settings['resolution'],reference))
        return requests

    def delete_selection(self):
        selected = set(self.selectedItems())
        for wire in list(self.wires):
            if wire in selected or wire.source.parentItem() in selected or wire.target.parentItem() in selected:
                self.wires.remove(wire)
                self.removeItem(wire)
        for item in selected:
            if isinstance(item, Node):self.removeItem(item)
        self.graphChanged.emit()


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = PoissonConfig()
        self.worker = None
        self.result = None
        self.monitor_records = {}
        self.active_monitors = set()
        self.last_run = None
        self.rectangle_previews = {}
        self.quick_insert = None
        self._closing = False
        self.setWindowTitle('Margam · baseline PINN builder')
        self.resize(1480,950)
        configure_theme(QApplication.instance(),'light')
        toolbar=self.addToolBar('Model')
        self.load_action=toolbar.addAction('Reset Poisson template')
        self.load_action.triggered.connect(self.reset_template)
        self.heat_action=toolbar.addAction('Heat demo')
        self.heat_action.triggered.connect(self.load_heat_template)
        self.zoom_in_action=toolbar.addAction('Zoom in')
        self.zoom_in_action.triggered.connect(lambda:self.view.zoom_by(1.2))
        self.zoom_out_action=toolbar.addAction('Zoom out')
        self.zoom_out_action.triggered.connect(lambda:self.view.zoom_by(1/1.2))
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
        self.train_button.clicked.connect(self.start_training)
        self.cancel_button=QPushButton('Stop')
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_training)
        for button in [self.validate_button,self.train_button,self.cancel_button]:toolbar.addWidget(button)
        self.progress_bar=QProgressBar();self.progress_bar.setMaximumWidth(210)
        self.progress_bar.setRange(0,self.config.iterations);self.progress_bar.setValue(0)
        toolbar.addWidget(self.progress_bar)
        self.scene=Canvas()
        self.view=CanvasView(self.scene)
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
        self.scene.blockOpened.connect(self.open_block)
        self.scene.blockContextRequested.connect(self.show_block_menu)
        self.scene.selectionChanged.connect(self.selection_changed)
        self.populate_template()
        self.scene.graphChanged.connect(self.graph_changed)
        self.scene.selectionChanged.emit()
        self.log.appendPlainText('Poisson template: Δu + c = 0. Select a block to configure it, validate, then train. Plot views are optional; add and connect an output block, then double-click it to open its view.\nBoundary sets are sampled first; assign a constant condition to each side in the BC block.')
        self.statusBar().showMessage('Ready · rectangle Poisson template · select blocks to edit settings')
        QTimer.singleShot(0,self.fit_model)

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, '_initial_layout_fitted', False):
            self._initial_layout_fitted = True
            QTimer.singleShot(0, self.fit_model)

    def setup_menus(self):
        file_menu=self.menuBar().addMenu('File')
        self.load_action.setShortcut('Ctrl+N')
        file_menu.addAction(self.load_action)
        file_menu.addAction(self.heat_action)
        file_menu.addSeparator()
        file_menu.addAction('Quit').triggered.connect(self.close)
        file_menu.actions()[-1].setShortcut('Ctrl+Q')
        edit=self.menuBar().addMenu('Edit')
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
        help_menu=self.menuBar().addMenu('Help')
        help_menu.addAction('Quick start').triggered.connect(lambda:QMessageBox.information(
            self,'Build and train a PINN',
            'Load Heat demo for a ready-wired monitor and reference comparison, or configure the basic Poisson blocks and train.\n\n'
            'Canvas: wheel to zoom, middle-drag or Space+drag to pan, tap Space to fit.\n\n'
            'Rectangle previews: right-click the Rectangle block and choose Preview geometry or Preview samples.\n\n'
            'Plots are optional. Add a Training Monitor or Solution Plot, connect the PINN method to it, '
            'then double-click the output block to open its view. Views can dock or float.\n\n'
            'A Comparison needs a model and an explicit Reference. References support intervals and rectangles with equal constant Dirichlet edges.'))
        help_menu.addAction('About Margam').triggered.connect(lambda:QMessageBox.about(
            self,'Margam','A visual builder for physics-informed neural networks.\nCurrent executable template: scalar Poisson on intervals and rectangles.'))

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

    def open_selected_output(self):
        nodes=[node for node in self.scene.nodes() if node.isSelected()]
        if len(nodes)==1:self.open_block(nodes[0])

    def open_block(self,node):
        if not isValid(node) or node.scene() is not self.scene:return
        if node.kind in OUTPUT_KINDS:
            return self.output_views.open(node)
        self.properties_dock.show()
        self.properties_dock.raise_()

    def create_block_menu(self, node):
        """Only existing block-specific actions; no universal plot capability."""
        menu=QMenu(self)
        menu.addAction('Properties').triggered.connect(lambda:self.show_block_properties(node))
        if node.kind=='domain' and self.config.shape=='Rectangle':
            menu.addSeparator()
            menu.addAction('Preview geometry').triggered.connect(lambda:self.open_rectangle_preview(node,False))
            menu.addAction('Preview samples').triggered.connect(lambda:self.open_rectangle_preview(node,True))
        elif node.kind in OUTPUT_KINDS:
            label={'monitor':'Open monitor','solution':'Open plot','comparison':'Open comparison'}[node.kind]
            menu.addAction(label).triggered.connect(lambda:self.open_block(node))
        menu.addSeparator()
        remove=menu.addAction('Delete block')
        remove.setEnabled(not self.is_running())
        remove.triggered.connect(lambda:self.delete_block(node))
        menu.addAction('Help').triggered.connect(lambda:self.block_help(node))
        return menu

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
        text={
            'domain':'Configure interval or rectangle bounds. Rectangle blocks provide optional local 2D geometry and sample previews through this menu.',
            'interior':'Choose the supported sampling method, point budget and seed for PDE points.',
            'boundary':'Choose a named rectangle side and configure its independent grid/random sampler, count and seed. Interval boundaries are fixed endpoints. Assign physical conditions separately in the BC block.',
            'pde':'The current template solves scalar Poisson with a constant source. Arbitrary residual editing is future work.',
            'bc':'Assign Dirichlet values or outward Neumann derivatives to the named sides.',
            'network':'Set hidden-layer sizes for the fully connected tanh network.',
            'method':'Configure baseline full-batch Adam training. Output blocks are optional.',
            'monitor':'Connect the PINN method to request bounded loss history. Open or hide its view independently of recording.',
            'solution':'Connect the PINN method to evaluate scalar predictions at the requested resolution. No reference is assumed.',
            'comparison':'Connect a model and an explicit Reference. The demo reference supports intervals and equal-Dirichlet rectangles (truncated series).',
            'reference':'An explicit Poisson reference for Comparison: analytical on intervals, truncated sine series on equal-Dirichlet rectangles.',
        }[node.kind]
        QMessageBox.information(self,node.title,text)

    def open_rectangle_preview(self, node, show_samples):
        if not isValid(node) or node.scene() is not self.scene or node.kind!='domain' or self.config.shape!='Rectangle':
            return
        key=(node.identifier,show_samples)
        if key not in self.rectangle_previews:
            title='Rectangle · samples' if show_samples else 'Rectangle · geometry'
            view=RectanglePreview(self.config,show_samples)
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
        owners={n.identifier for n in self.scene.nodes() if n.kind=='domain'}
        for key in list(self.rectangle_previews):
            if self.config.shape!='Rectangle' or key[0] not in owners:
                dock,_=self.rectangle_previews.pop(key)
                dock.close();self.removeDockWidget(dock);dock.deleteLater()


    def rename_output(self,node,name):
        name=name.strip()
        if not name:return
        node.title=name;node.update();self.name.setText(name)
        entry=self.output_views.entries.get(node.identifier)
        if entry:entry[0].setWindowTitle(name)

    def change_output(self,node,key,value):
        if self.is_running():return
        node.settings[key]=value
        if key=='history_limit':
            records=self.monitor_records.get(node.identifier,[])
            del records[:-value]
            self.output_views.refresh(node)
        else:self.graph_changed()


    def dock(self,title,widget,area):
        dock=QDockWidget(title,self);dock.setObjectName(title)
        dock.setWidget(widget);self.addDockWidget(area,dock)
        return dock

    def fit_model(self):
        self.view.fit_model()

    def make_node(self,title,detail,x,y):
        node=Node(title,detail,x,y);self.scene.addItem(node);return node

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
        self.output_views.sync([])
        self.close_rectangle_previews()
        self.scene.wires.clear();self.scene.pending=None;self.scene.clear()
        self.config=PoissonConfig()
        self.populate_template();self.settings_changed();self.fit_model()
        self.log.appendPlainText('Loaded the rectangle Poisson template.')

    def load_heat_template(self):
        if self.is_running():return
        self.reset_template()
        self.config=heat_demo_config()
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
        self.statusBar().showMessage('Heat demo ready · Train PINN, then open monitor, prediction or comparison blocks')
        self.log.appendPlainText('Loaded steady heat demo: unit conductivity, constant source 2, zero edge temperature. Explicit series reference and output blocks are connected; views remain closed until requested.')

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

    def add_node(self,title,position=None):
        if self.is_running() or self.scene.locked or title not in KINDS:return
        center=position if position is not None else self.view.mapToScene(self.view.viewport().rect().center())
        node=self.make_node(title,'Not configured',center.x()-85,center.y()-40)
        if node.kind in OUTPUT_KINDS:
            node.settings['resolution']=101 if self.config.shape=='Interval' else 41
            node.title=title+' '+str(sum(n.kind==node.kind for n in self.scene.nodes()))
        self.scene.clearSelection();node.setSelected(True);self.graph_changed()
        return node

    def delete_selected(self):
        if not self.is_running():self.scene.delete_selection()

    def is_running(self):
        return self.worker is not None

    def clear_run_data(self):
        self.result=None
        self.monitor_records={}
        self.active_monitors=set()
        self.output_views.clear_data()
        if self.last_run is not None:
            self.last_run.model=None
            self.last_run.deleteLater()
            self.last_run=None

    def graph_changed(self):
        self.clear_run_data()
        self.output_views.sync(self.scene.nodes())
        self.sync_rectangle_previews()
        self.refresh_descriptions()
        self.progress_bar.setValue(0)
        self.validation.setText('Graph changed. Validate the model before training.')

    def refresh_descriptions(self):
        c=self.config
        actual=int(c.points**.5)**2 if c.shape=='Rectangle' and c.sampling=='grid' else c.points
        details={'domain':f'x: {c.x_min:g}…{c.x_max:g}' if c.shape=='Interval' else f'{c.x_max-c.x_min:g} × {c.y_max-c.y_min:g}',
                 'interior':f'{actual} points · {c.sampling}',
                 'boundary':f'{len(c.sides)} sets · {sum(c.boundary_recipe(s).points for s in c.sides) if c.shape=="Rectangle" else 2} points', 'pde':f'Δu + {c.source:g} = 0',
                 'bc':', '.join(c.bc_types[s][0] for s in c.sides),
                 'network':' → '.join(map(str,c.layers)), 'method':f'Adam · {c.iterations} steps',
                 'solution':'Requested prediction plot', 'monitor':'Requested loss history',
                 'comparison':'Model + explicit reference source', 'reference':'Explicit Poisson reference'}
        for node in self.scene.items():
            if isinstance(node,Node):
                if node.kind=='domain':node.title=c.shape
                node.detail=details[node.kind];node.update()
                if node.isSelected():
                    self.name.setText(node.title);self.detail.setText(node.detail)

    def change(self,attribute,value):
        if self.is_running():return
        setattr(self.config,attribute,value)
        self.settings_changed()
        if attribute=='shape':self.selection_changed()

    def settings_changed(self):
        self.clear_run_data()
        self.progress_bar.setRange(0,self.config.iterations);self.progress_bar.setValue(0)
        self.refresh_descriptions()
        self.sync_rectangle_previews()
        for _,preview in self.rectangle_previews.values():
            preview.config=self.config;preview.update()
        try:self.config.validate();self.validation.setText('Settings ready. Connections are checked before training.')
        except ValueError as error:self.validation.setText(str(error))
        self.statusBar().showMessage('Settings changed · previous results cleared')

    def field(self,label,attribute,kind='float',choices=None,minimum=-1000000,maximum=1000000):
        value=getattr(self.config,attribute)
        if kind=='choice':
            widget=QComboBox();widget.addItems(choices);widget.setCurrentText(str(value))
            if attribute=='sampling':
                for i,method in enumerate(choices):widget.setItemData(i,SAMPLERS[method],Qt.ItemDataRole.ToolTipRole)
            widget.currentTextChanged.connect(lambda text:self.change(attribute,text))
        elif kind=='int':
            widget=QSpinBox();widget.setKeyboardTracking(False);widget.setRange(int(minimum),int(maximum));widget.setValue(value)
            widget.valueChanged.connect(lambda number:self.change(attribute,number))
        else:
            widget=QDoubleSpinBox();widget.setKeyboardTracking(False);widget.setDecimals(6);widget.setRange(minimum,maximum);widget.setValue(value)
            widget.valueChanged.connect(lambda number:self.change(attribute,number))
        self.form.addRow(label,widget)
        return widget

    @Slot()
    def selection_changed(self):
        if not isValid(self.scene):
            return
        nodes=[i for i in self.scene.selectedItems() if isinstance(i,Node)]
        node=nodes[0] if len(nodes)==1 else None
        self.name.setText(node.title if node else ('Multiple blocks selected' if len(nodes)>1 else ''))
        self.open_output_action.setEnabled(node is not None and node.kind in OUTPUT_KINDS)
        self.detail.setText(node.detail if node else '')
        body=QWidget();self.form=QFormLayout(body)
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        if node is None:
            self.form.addRow(QLabel('Select a block to edit its settings.'))
        elif node.kind=='domain':
            self.field('Domain','shape','choice',choices=['Interval','Rectangle'])
            self.field('x minimum','x_min');self.field('x maximum','x_max')
            if self.config.shape=='Rectangle':
                self.field('y minimum','y_min');self.field('y maximum','y_max')
            note=QLabel('Built-in meshless geometry. No external file or mesh is required.');note.setWordWrap(True);self.form.addRow(note)
        elif node.kind=='interior':
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
                recipe=self.config.boundary_recipe(side)
                method=QComboBox();method.addItems(list(SAMPLERS));method.setCurrentText(recipe.method)
                method.setObjectName('boundary_method')
                for i,key in enumerate(SAMPLERS):method.setItemData(i,SAMPLERS[key],Qt.ItemDataRole.ToolTipRole)
                method.currentTextChanged.connect(lambda value,s=side:self.change_boundary_sampling(s,'method',value))
                self.form.addRow('Sampler',method)
                count=QSpinBox();count.setKeyboardTracking(False);count.setRange(1,10000);count.setValue(recipe.points)
                count.setObjectName('boundary_count')
                count.valueChanged.connect(lambda value,s=side:self.change_boundary_sampling(s,'points',value))
                self.form.addRow('Points on this side',count)
                seed=QSpinBox();seed.setKeyboardTracking(False);seed.setRange(0,2147483647);seed.setValue(recipe.seed)
                seed.setObjectName('boundary_seed')
                seed.valueChanged.connect(lambda value,s=side:self.change_boundary_sampling(s,'seed',value))
                self.form.addRow('Seed',seed)
                note=QLabel('Each side has an independent recipe. Grid and seeded uniform random return exactly the selected count on that side. Grid points exclude corners. The seed is retained but unused for a grid. Interior settings do not change boundary recipes.')
            else:
                note=QLabel('Interval point sets are the left and right endpoints: one point each. Rectangle-side recipes are retained but do not apply to an interval.')
            note.setWordWrap(True);self.form.addRow(note)
            note=QLabel('Named point sets: '+', '.join(self.config.sides)+'. Types and targets are assigned later in the Boundary condition block. Open Preview samples from the Rectangle block to inspect the combined samples.')
            note.setWordWrap(True);self.form.addRow(note)
        elif node.kind=='pde':
            note=QLabel('Poisson preset: Δu + c = 0\nUnknown: scalar field u\n1D: uₓₓ + c = 0\n2D: uₓₓ + uᵧᵧ + c = 0');note.setWordWrap(True);self.form.addRow(note)
            self.field('Constant c','source')
            note=QLabel('Custom residual editing and additional pointwise PDE inputs are future work.');note.setWordWrap(True);self.form.addRow(note)
        elif node.kind=='bc':
            for side in self.config.sides:
                row=QWidget();layout=QHBoxLayout(row);layout.setContentsMargins(0,0,0,0)
                combo=QComboBox();combo.addItems(['Dirichlet','Neumann']);combo.setCurrentText(self.config.bc_types[side])
                value=QDoubleSpinBox();value.setDecimals(5);value.setRange(-100000,100000);value.setValue(self.config.bc_values[side])
                combo.currentTextChanged.connect(lambda text,s=side:self.change_bc(s,'type',text))
                value.valueChanged.connect(lambda number,s=side:self.change_bc(s,'value',number))
                layout.addWidget(combo);layout.addWidget(value);self.form.addRow(side.capitalize(),row)
            note=QLabel('Dirichlet: u = target.\nNeumann: ∂u/∂n = target (outward normal), not conductivity-scaled flux.\nAdjacent Dirichlet values must agree at shared rectangle corners. At least one Dirichlet side is required.');note.setWordWrap(True);self.form.addRow(note)
        elif node.kind=='network':
            note=QLabel('Fully connected · tanh · Xavier normal\nFixed input bounds follow the domain.');note.setWordWrap(True);self.form.addRow(note)
            edit=QLineEdit(', '.join(map(str,self.config.layers)))
            edit.setPlaceholderText('16, 16')
            edit.textChanged.connect(self.change_layers)
            self.form.addRow('Hidden layers',edit)
        elif node.kind=='method':
            self.form.addRow(QLabel('Method: baseline PINN\nFixed samples · full-batch Adam'))
            self.field('Iterations','iterations','int',minimum=1,maximum=100000)
            self.field('Learning rate','lr',minimum=.000001,maximum=1)
            self.field('PDE weight','pde_weight',minimum=.000001,maximum=100000)
            self.field('Mean BC weight','boundary_weight',minimum=.000001,maximum=100000)
            self.field('Device','device','choice',choices=['cpu','auto','cuda'])
            self.form.addRow(QLabel('Stop requests are checked every 25 updates.'))
        elif node.kind in OUTPUT_KINDS:
            name=QLineEdit(node.title)
            name.editingFinished.connect(lambda n=node,e=name:self.rename_output(n,e.text()))
            self.form.addRow('Name',name)
            if node.kind=='monitor':
                count=QSpinBox();count.setRange(25,100000);count.setValue(node.settings['history_limit'])
                count.valueChanged.connect(lambda value,n=node:self.change_output(n,'history_limit',value))
                self.form.addRow('Retained updates',count)
                text='Connect PINN method → monitor to record losses. Closing the view does not stop recording. Removing the block removes the request.'
            else:
                count=QSpinBox();count.setRange(2,201);count.setValue(node.settings['resolution'])
                count.valueChanged.connect(lambda value,n=node:self.change_output(n,'resolution',value))
                self.form.addRow('Evaluation points / axis',count)
                text='Connect PINN method to this plot. No reference or error metric is assumed.' if node.kind=='solution' else 'Connect PINN method to the upper input and Reference to the lower input. The rectangle reference requires equal constant Dirichlet edges and uses a truncated sine series.'
            note=QLabel(text);note.setWordWrap(True);self.form.addRow(note)
            button=QPushButton('Open view');button.clicked.connect(lambda checked=False,n=node:self.open_block(n))
            self.form.addRow(button)
        elif node.kind=='reference':
            note=QLabel('Explicit Poisson reference: analytical for intervals; 40 odd sine modes per axis for rectangles with equal constant Dirichlet edges. The rectangle reference is truncated, not an exact closed-form value. Connect it to Comparison; ordinary plots never evaluate it.');note.setWordWrap(True);self.form.addRow(note)
        old=self.property_scroll.takeWidget()
        if old is not None:old.deleteLater()
        self.property_scroll.setWidget(body)
        self.property_scroll.setEnabled(not self.is_running())

    def select_boundary_side(self,side):
        self.boundary_side=side
        self.selection_changed()

    def change_boundary_sampling(self,side,attribute,value):
        if self.is_running():return
        recipe=self.config.boundary_recipe(side)
        self.config.boundary_sampling[side]=replace(recipe,**{attribute:value})
        self.settings_changed()

    def change_bc(self,side,kind,value):
        if self.is_running():return
        target=self.config.bc_types if kind=='type' else self.config.bc_values
        target[side]=value;self.settings_changed()

    def change_layers(self,text):
        try:layers=tuple(int(n.strip()) for n in text.split(','))
        except ValueError:layers=()
        self.change('layers',layers)

    def validate_model(self):
        try:self.config.validate();requests=self.scene.output_requests(self.config)
        except ValueError as error:
            message=str(error);self.validation.setText(message)
            self.log.appendPlainText('Validation: '+message)
            self.statusBar().showMessage('Model needs attention · see Properties or Messages')
            return False
        self.validation.setText(f'Valid Poisson model. {len(requests)} connected output request(s).')
        self.statusBar().showMessage('Valid · baseline Poisson model')
        return True

    def set_busy(self,busy):
        if busy and self.quick_insert is not None:self.quick_insert.reject()
        for control in [self.train_button,self.validate_button,self.library,self.property_scroll]:control.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.load_action.setEnabled(not busy);self.heat_action.setEnabled(not busy);self.delete_action.setEnabled(not busy)
        self.scene.locked=busy
        for node in self.scene.nodes():
            node.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable,not busy)
        self.run_action.setEnabled(not busy);self.check_action.setEnabled(not busy)
        self.stop_action.setEnabled(busy)

    def start_training(self):
        if self.is_running() or not self.validate_model():return False
        self.clear_run_data()
        requests=self.scene.output_requests(self.config)
        self.active_monitors={r.identifier for r in requests if r.kind=='monitor'}
        self.monitor_records={identifier:[] for identifier in self.active_monitors}
        self.progress_bar.setRange(0,self.config.iterations);self.progress_bar.setValue(0)
        self.set_busy(True)
        self.worker=TrainingWorker(deepcopy(self.config),requests,parent=self)
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
        if not getattr(self,'_selection_disconnected',False):
            self.scene.selectionChanged.disconnect(self.selection_changed)
            self._selection_disconnected=True
        super().closeEvent(event)
