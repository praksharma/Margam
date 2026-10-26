"""Block-based baseline Poisson workflow with background training."""
import sys
from copy import deepcopy
from ..templates.poisson import PoissonConfig
from ..components.sampling import SAMPLERS
from .plots import DomainPreview, CurvePlot, SolutionView
from .training import TrainingWorker
from .theme import colors, configure_theme
from PySide6.QtCore import Qt, QPointF, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QListWidget, QGraphicsView,
    QGraphicsScene, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPathItem,
    QGraphicsItem, QWidget, QVBoxLayout, QLabel, QFormLayout, QLineEdit,
    QTabWidget, QPlainTextEdit, QScrollArea, QComboBox, QDoubleSpinBox,
    QSpinBox, QPushButton, QProgressBar, QHBoxLayout,
)


class Port(QGraphicsEllipseItem):
    def __init__(self, node, output):
        super().__init__(-6, -6, 12, 12, node)
        self.output = output
        self.setPos(170 if output else 0, 40)
        self.setBrush(QColor(colors()['highlight']))
        self.setPen(QPen(QColor(colors()['base']), 2))
        self.setToolTip('Drag to an input' if output else 'Input')
        self.setZValue(2)

    def mousePressEvent(self, event):
        if self.output:
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


KINDS = {"Rectangle": "domain", "Interior sampler": "interior",
         "Boundary points": "boundary", "PDE residual": "pde",
         "Boundary condition": "bc", "Neural network": "network",
         "PINN method": "method", "Solution": "solution"}

REQUIRED_EDGES = {("domain","interior"),("domain","boundary"),
                  ("interior","pde"),("boundary","bc"),("pde","method"),
                  ("bc","method"),("network","method"),("method","solution")}


class Node(QGraphicsRectItem):
    def __init__(self, title, detail, x, y):
        super().__init__(0, 0, 170, 80)
        self.title, self.detail = title, detail
        self.kind = KINDS[title]
        self.setBrush(QColor(colors()['base']))
        self.setPen(QPen(QColor(colors()['border']), 1.5))
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
                      QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
                      QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.input = Port(self, False)
        self.output = Port(self, True)
        self.setPos(x, y)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        painter.setPen(QColor(colors()['text']))
        painter.drawText(14, 28, self.title)
        painter.setPen(QColor(colors()['muted']))
        painter.drawText(14, 55, painter.fontMetrics().elidedText(self.detail, Qt.TextElideMode.ElideRight, 145))

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and self.scene():
            for wire in self.scene().wires:
                wire.update_path()
        return super().itemChange(change, value)


class Canvas(QGraphicsScene):
    graphChanged = Signal()

    def __init__(self):
        super().__init__()
        self.wires, self.pending = [], None
        self.setSceneRect(-80, -80, 1300, 650)
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

    def connect_nodes(self, source, target):
        wire = Wire(source.output, target.input)
        self.addItem(wire)
        self.wires.append(wire)
        self.graphChanged.emit()

    def validate_graph(self):
        nodes = [i for i in self.items() if isinstance(i, Node)]
        for kind in KINDS.values():
            if sum(n.kind == kind for n in nodes) != 1:
                raise ValueError(f'The Poisson template needs exactly one {kind} block.')
        edges = {(w.source.parentItem().kind,w.target.parentItem().kind) for w in self.wires}
        if len(edges) != len(self.wires):
            raise ValueError('Duplicate connections are not supported.')
        if edges != REQUIRED_EDGES:
            missing = REQUIRED_EDGES-edges
            extra = edges-REQUIRED_EDGES
            detail = ', '.join(f'{a} → {b}' for a,b in sorted(missing or extra))
            raise ValueError(('Missing connections: ' if missing else 'Unsupported connections: ')+detail)

    def delete_selection(self):
        selected = set(self.selectedItems())
        for wire in list(self.wires):
            if wire in selected or wire.source.parentItem() in selected or wire.target.parentItem() in selected:
                self.wires.remove(wire)
                self.removeItem(wire)
        for item in selected:
            if isinstance(item, Node):self.removeItem(item)
        self.graphChanged.emit()


class ModelView(QGraphicsView):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene() is not None:
            self.fitInView(self.scene().itemsBoundingRect().adjusted(-35,-35,35,35),
                           Qt.AspectRatioMode.KeepAspectRatio)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = PoissonConfig()
        self.worker = None
        self.result = None
        self.history = []
        self._closing = False
        self.setWindowTitle('Margam · baseline PINN builder')
        self.resize(1480,950)
        configure_theme(QApplication.instance(),'light')
        appearance=self.menuBar().addMenu('Appearance')
        for label in ('Light','Dark'):
            appearance.addAction(label).triggered.connect(lambda checked=False,name=label.lower():configure_theme(QApplication.instance(),name))
        toolbar=self.addToolBar('Model')
        self.load_action=toolbar.addAction('Reset Poisson template')
        self.load_action.triggered.connect(self.reset_template)
        toolbar.addAction('Fit model').triggered.connect(self.fit_model)
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
        self.view=ModelView(self.scene)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setCentralWidget(self.view)
        self.library=QListWidget();self.library.addItems(list(KINDS))
        self.library.itemDoubleClicked.connect(lambda item:self.add_node(item.text()))
        self.dock('Block library · double-click to add',self.library,Qt.DockWidgetArea.LeftDockWidgetArea)
        properties=QWidget();layout=QVBoxLayout(properties)
        self.name=QLineEdit();self.name.setReadOnly(True)
        self.detail=QLineEdit();self.detail.setReadOnly(True)
        layout.addWidget(QLabel('Selected block'));layout.addWidget(self.name);layout.addWidget(self.detail)
        self.property_scroll=QScrollArea();self.property_scroll.setWidgetResizable(True)
        layout.addWidget(self.property_scroll)
        self.validation=QLabel('');self.validation.setWordWrap(True);layout.addWidget(self.validation)
        self.dock('Properties',properties,Qt.DockWidgetArea.RightDockWidgetArea)
        self.tabs=QTabWidget()
        self.preview=DomainPreview(self.config)
        self.loss_plot=CurvePlot(logarithmic=True)
        self.solution=SolutionView()
        self.log=QPlainTextEdit();self.log.setReadOnly(True)
        for widget,title in [(self.preview,'Domain & samples'),(self.loss_plot,'Losses'),(self.solution,'Solution'),(self.log,'Messages')]:
            self.tabs.addTab(widget,title)
        self.tabs.setMinimumHeight(285)
        self.dock('Preview and results',self.tabs,Qt.DockWidgetArea.BottomDockWidgetArea)
        self.scene.selectionChanged.connect(self.selection_changed)
        self.populate_template()
        self.scene.graphChanged.connect(self.graph_changed)
        self.scene.selectionChanged.emit()
        self.log.appendPlainText('Poisson template: Δu + c = 0. Select a block to configure it, validate, then train.\nBoundary sets are sampled first; assign a constant condition to each side in the BC block.')
        self.statusBar().showMessage('Ready · rectangle Poisson template · select blocks to edit settings')
        QTimer.singleShot(0,self.fit_model)

    def dock(self,title,widget,area):
        dock=QDockWidget(title,self);dock.setObjectName(title)
        dock.setWidget(widget);self.addDockWidget(area,dock)
        return dock

    def fit_model(self):
        self.view.fitInView(self.scene.itemsBoundingRect().adjusted(-35,-35,35,35),Qt.AspectRatioMode.KeepAspectRatio)

    def make_node(self,title,detail,x,y):
        node=Node(title,detail,x,y);self.scene.addItem(node);return node

    def populate_template(self):
        definitions=[('Rectangle','Unit square',0,160),('Interior sampler','PDE points',240,40),
                     ('Boundary points','Named sides',240,270),('PDE residual','Poisson',480,40),
                     ('Boundary condition','Side conditions',480,270),('Neural network','Fully connected',480,440),
                     ('PINN method','Baseline PINN',730,160),('Solution','Field and loss plots',970,160)]
        nodes=[self.make_node(*d) for d in definitions]
        for a,b in [(0,1),(0,2),(1,3),(2,4),(3,6),(4,6),(5,6),(6,7)]:self.scene.connect_nodes(nodes[a],nodes[b])
        self.refresh_descriptions()
        nodes[0].setSelected(True)

    def reset_template(self):
        if self.is_running():return
        self.scene.wires.clear();self.scene.pending=None;self.scene.clear()
        self.config=PoissonConfig();self.preview.config=self.config
        self.populate_template();self.settings_changed();self.fit_model()
        self.log.appendPlainText('Loaded the rectangle Poisson template.')

    def add_node(self,title):
        if self.is_running():return
        center=self.view.mapToScene(self.view.viewport().rect().center())
        node=self.make_node(title,'Not configured',center.x(),center.y())
        self.scene.clearSelection();node.setSelected(True);self.graph_changed()

    def delete_selected(self):
        if not self.is_running():self.scene.delete_selection()

    def is_running(self):
        return self.worker is not None and self.worker.isRunning()

    def graph_changed(self):
        self.result=None;self.history=[];self.solution.clear()
        self.loss_plot.set_series([],'Graph changed. Validate and train again.')
        self.progress_bar.setValue(0)
        self.validation.setText('Graph changed. Validate the model before training.')

    def refresh_descriptions(self):
        c=self.config
        actual=int(c.points**.5)**2 if c.shape=='Rectangle' and c.sampling=='grid' else c.points
        details={'domain':f'x: {c.x_min:g}…{c.x_max:g}' if c.shape=='Interval' else f'{c.x_max-c.x_min:g} × {c.y_max-c.y_min:g}',
                 'interior':f'{actual} points · {c.sampling}',
                 'boundary':f'{len(c.sides)} named sets', 'pde':f'Δu + {c.source:g} = 0',
                 'bc':', '.join(c.bc_types[s][0] for s in c.sides),
                 'network':' → '.join(map(str,c.layers)), 'method':f'Adam · {c.iterations} steps',
                 'solution':'Field and loss plots'}
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
        self.result=None;self.history=[];self.solution.clear()
        self.loss_plot.set_series([],'Settings changed. Train to calculate losses.')
        self.progress_bar.setRange(0,self.config.iterations);self.progress_bar.setValue(0)
        self.refresh_descriptions();self.preview.update()
        try:self.config.validate();self.validation.setText('Settings ready. Connections are checked before training.')
        except ValueError as error:self.validation.setText(str(error))
        self.statusBar().showMessage('Settings changed · previous results cleared')

    def field(self,label,attribute,kind='float',choices=None,minimum=-1000000,maximum=1000000):
        value=getattr(self.config,attribute)
        if kind=='choice':
            widget=QComboBox();widget.addItems(choices);widget.setCurrentText(str(value))
            widget.currentTextChanged.connect(lambda text:self.change(attribute,text))
        elif kind=='int':
            widget=QSpinBox();widget.setKeyboardTracking(False);widget.setRange(int(minimum),int(maximum));widget.setValue(value)
            widget.valueChanged.connect(lambda number:self.change(attribute,number))
        else:
            widget=QDoubleSpinBox();widget.setKeyboardTracking(False);widget.setDecimals(6);widget.setRange(minimum,maximum);widget.setValue(value)
            widget.valueChanged.connect(lambda number:self.change(attribute,number))
        self.form.addRow(label,widget)
        return widget

    def selection_changed(self):
        nodes=[i for i in self.scene.selectedItems() if isinstance(i,Node)]
        node=nodes[0] if nodes else None
        self.name.setText(node.title if node else '')
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
            self.field('Point budget','points','int',minimum=1,maximum=10000)
            self.field('Seed','seed','int',minimum=0,maximum=2147483647)
            note=QLabel('Rectangle grids use complete layers and may return fewer points than the budget. Actual counts appear on the block and preview.');note.setWordWrap(True);self.form.addRow(note)
        elif node.kind=='boundary':
            if self.config.shape=='Rectangle':self.field('Points per side','boundary_points','int',minimum=1,maximum=1000)
            note=QLabel('Named point sets: '+', '.join(self.config.sides)+'.\nTypes and targets are assigned in the Boundary condition block. Grid samples exclude corners; interval sets are the endpoints.');note.setWordWrap(True);self.form.addRow(note)
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
        else:
            note=QLabel('Results appear below.\nInterval: prediction, analytical reference and RMSE.\nRectangle: predicted field heatmap; no analytical error estimate.');note.setWordWrap(True);self.form.addRow(note)
        old=self.property_scroll.takeWidget()
        if old is not None:old.deleteLater()
        self.property_scroll.setWidget(body)
        self.property_scroll.setEnabled(not self.is_running())

    def change_bc(self,side,kind,value):
        if self.is_running():return
        target=self.config.bc_types if kind=='type' else self.config.bc_values
        target[side]=value;self.settings_changed()

    def change_layers(self,text):
        try:layers=tuple(int(n.strip()) for n in text.split(','))
        except ValueError:layers=()
        self.change('layers',layers)

    def validate_model(self):
        try:self.config.validate();self.scene.validate_graph()
        except ValueError as error:
            message=str(error);self.validation.setText(message)
            self.log.appendPlainText('Validation: '+message)
            self.statusBar().showMessage('Model needs attention · see Properties or Messages')
            return False
        self.validation.setText('Valid Poisson model. Ready to train.')
        self.statusBar().showMessage('Valid · baseline Poisson model')
        return True

    def set_busy(self,busy):
        for control in [self.train_button,self.validate_button,self.library,self.property_scroll]:control.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.load_action.setEnabled(not busy);self.delete_action.setEnabled(not busy)
        self.view.setInteractive(not busy)

    def start_training(self):
        if self.is_running() or not self.validate_model():return False
        self.result=None;self.history=[];self.solution.clear()
        self.progress_bar.setRange(0,self.config.iterations);self.progress_bar.setValue(0)
        self.loss_plot.set_series([],'Starting baseline PINN…')
        self.tabs.setCurrentWidget(self.loss_plot)
        self.set_busy(True)
        self.worker=TrainingWorker(deepcopy(self.config),self)
        self.worker.progress.connect(self.training_progress)
        self.worker.completed.connect(self.training_completed)
        self.worker.failed.connect(self.training_failed)
        self.worker.finished.connect(self.training_finished)
        self.log.appendPlainText(f'Training {self.config.shape} · {self.config.iterations} updates · {self.config.device}')
        self.statusBar().showMessage('Preparing samples and network…')
        self.worker.start()
        return True

    def training_progress(self,step,records):
        self.history.extend(records);self.progress_bar.setValue(step)
        sides=self.config.sides
        series=[('Total',[(i+1,r['total']) for i,r in enumerate(self.history)],'#2b9d91'),
                ('PDE (raw)',[(i+1,r['pde']) for i,r in enumerate(self.history)],'#d58b35'),
                ('Mean BC (raw)',[(i+1,sum(r[s] for s in sides)/len(sides)) for i,r in enumerate(self.history)],'#7089d5')]
        self.loss_plot.set_series(series,'Training losses · logarithmic scale · measured before each update')
        self.statusBar().showMessage(f"Training {step}/{self.config.iterations} · loss {records[-1]['total']:.5g}")

    def training_completed(self,result):
        self.result=result;self.solution.show_result(result)
        self.tabs.setCurrentWidget(self.solution)
        status='Stopped' if result['cancelled'] else 'Completed'
        message=f"{status} · {result['steps']} updates · {result['device']}"
        if result['rmse'] is not None:message+=f" · RMSE {result['rmse']:.6g}"
        self.statusBar().showMessage(message);self.log.appendPlainText(message)

    def training_failed(self,message):
        self.result=None
        self.statusBar().showMessage('Training failed · see Messages')
        self.log.appendPlainText('Training failed: '+message)
        self.tabs.setCurrentWidget(self.log)

    def training_finished(self):
        worker=self.worker;self.worker=None
        self.set_busy(False)
        if worker is not None:worker.deleteLater()
        if self._closing:QTimer.singleShot(0,self.close)

    def cancel_training(self):
        if self.worker is not None:
            self.worker.cancel();self.cancel_button.setEnabled(False)
            self.statusBar().showMessage('Stopping after the current update batch…')

    def apply_theme(self):
        if not hasattr(self,'scene'):return
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
