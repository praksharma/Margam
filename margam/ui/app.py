"""Margam layout study. Run with Python and PySide6; no solver is connected."""
import sys
from .theme import colors, configure_theme
from PySide6.QtCore import Qt, QPointF, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QListWidget, QGraphicsView,
    QGraphicsScene, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPathItem,
    QGraphicsItem, QWidget, QVBoxLayout, QLabel, QFormLayout, QLineEdit,
    QTabWidget, QPlainTextEdit,
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
        self.update_path()

    def update_path(self, end=None):
        a = self.source.scenePos()
        b = self.target.scenePos() if self.target else (end or a)
        bend = max(60, abs(b.x() - a.x()) / 2)
        path = QPainterPath(a)
        path.cubicTo(a + QPointF(bend, 0), b - QPointF(bend, 0), b)
        self.setPath(path)


class Node(QGraphicsRectItem):
    def __init__(self, title, detail, x, y):
        super().__init__(0, 0, 170, 80)
        self.title, self.detail = title, detail
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
        painter.drawText(14, 55, self.detail)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and self.scene():
            for wire in self.scene().wires:
                wire.update_path()
        return super().itemChange(change, value)


class Canvas(QGraphicsScene):
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

    def connect_nodes(self, source, target):
        wire = Wire(source.output, target.input)
        self.addItem(wire)
        self.wires.append(wire)


class DomainPreview(QWidget):
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        x, y, w, h = 70, 40, min(420, self.width() - 140), max(40, self.height() - 85)
        p.setPen(QPen(QColor(colors()['highlight']), 3))
        p.drawRect(x, y, w, h)
        p.setPen(QColor(colors()['muted']))
        for i in range(1, 15):
            for j in range(1, 6):
                p.drawEllipse(QPointF(x + i*w/15, y + j*h/6), 2, 2)
        p.setPen(QColor(colors()['text']))
        p.drawText(x, 23, 'Rectangle · illustrative samples')
        p.drawText(x + w + 12, y + h//2, 'Right')
        p.drawText(10, y + h//2, 'Left')
        p.drawText(x + w//2, y + h + 23, 'Bottom')


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Margam · PINN builder · layout prototype')
        self.resize(1420, 900)
        configure_theme(QApplication.instance(), 'light')
        appearance = self.menuBar().addMenu('Appearance')
        for label in ('Light', 'Dark'):
            action = appearance.addAction(label)
            action.triggered.connect(lambda checked=False, name=label.lower():
                                     configure_theme(QApplication.instance(), name))
        self.scene = Canvas()
        view = QGraphicsView(self.scene)
        view.setRenderHint(QPainter.RenderHint.Antialiasing)
        view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setCentralWidget(view)
        library = QListWidget()
        library.addItems(['Rectangle', 'Interior sampler', 'Boundary points', 'PDE residual',
                          'Boundary condition', 'Neural network', 'PINN method', 'Solution'])
        library.itemDoubleClicked.connect(lambda item: self.add_node(item.text()))
        self.dock('Block library · double-click to add', library, Qt.DockWidgetArea.LeftDockWidgetArea)
        properties = QWidget()
        layout = QVBoxLayout(properties)
        layout.addWidget(QLabel('Selected block'))
        form = QFormLayout()
        self.name = QLineEdit()
        self.name.setReadOnly(True)
        self.detail = QLineEdit()
        self.detail.setReadOnly(True)
        form.addRow('Name', self.name)
        form.addRow('Summary', self.detail)
        layout.addLayout(form)
        layout.addWidget(QLabel('Layout study: settings and solving\nwill be connected in a later stage.'))
        layout.addStretch()
        self.dock('Properties', properties, Qt.DockWidgetArea.RightDockWidgetArea)
        tabs = QTabWidget()
        tabs.addTab(DomainPreview(), '2D domain preview')
        log = QPlainTextEdit('No solver connected.\nDrag blocks to move them.\nDrag an output port to an input port to connect.\nConnections are illustrative; physics compatibility is not checked.')
        log.setReadOnly(True)
        tabs.addTab(log, 'Messages')
        tabs.setMinimumHeight(230)
        self.dock('Preview and messages', tabs, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.statusBar().showMessage('Prototype · drag blocks and connect ports · no training backend')
        self.scene.selectionChanged.connect(self.selection_changed)
        definitions = [
            ('Rectangle', 'Unit square', 0, 160),
            ('Interior sampler', 'PDE points', 240, 40),
            ('Boundary points', 'Four named sides', 240, 270),
            ('PDE residual', 'User-defined equation', 480, 40),
            ('Boundary condition', 'Assign type and target', 480, 270),
            ('Neural network', 'Fully connected', 480, 440),
            ('PINN method', 'Baseline PINN', 730, 160),
            ('Solution', 'Field and loss plots', 970, 160),
        ]
        nodes = [self.make_node(*d) for d in definitions]
        for a, b in [(0,1),(0,2),(1,3),(2,4),(3,6),(4,6),(5,6),(6,7)]:
            self.scene.connect_nodes(nodes[a], nodes[b])
        QTimer.singleShot(0, lambda: view.fitInView(
            self.scene.itemsBoundingRect().adjusted(-35, -35, 35, 35),
            Qt.AspectRatioMode.KeepAspectRatio))

    def apply_theme(self):
        if not hasattr(self, "scene"):
            return
        c = colors()
        self.scene.setBackgroundBrush(QColor(c["canvas"]))
        for item in self.scene.items():
            if isinstance(item, Node):
                item.setBrush(QColor(c["base"]))
                item.setPen(QPen(QColor(c["border"]), 1.5))
            elif isinstance(item, Port):
                item.setBrush(QColor(c["highlight"]))
                item.setPen(QPen(QColor(c["base"]), 2))
            elif isinstance(item, Wire):
                item.setPen(QPen(QColor(c["highlight"]), 2))
        self.scene.update()
        for preview in self.findChildren(DomainPreview):
            preview.update()

    def dock(self, title, widget, area):
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)

    def make_node(self, title, detail, x, y):
        node = Node(title, detail, x, y)
        self.scene.addItem(node)
        return node

    def add_node(self, title):
        center = self.centralWidget().mapToScene(self.centralWidget().viewport().rect().center())
        self.make_node(title, 'Not configured', center.x(), center.y())

    def closeEvent(self, event):
        # Stop scene teardown from invoking a slot on an already deleted scene.
        if not getattr(self, "_selection_disconnected", False):
            self.scene.selectionChanged.disconnect(self.selection_changed)
            self._selection_disconnected = True
        super().closeEvent(event)

    def selection_changed(self):
        nodes = [i for i in self.scene.selectedItems() if isinstance(i, Node)]
        self.name.setText(nodes[0].title if nodes else '')
        self.detail.setText(nodes[0].detail if nodes else '')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = Window()
    window.show()
    sys.exit(app.exec())
