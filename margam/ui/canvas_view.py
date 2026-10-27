"""2D model navigation: pointer-centred wheel zoom, pan, and explicit fitting."""
from PySide6.QtCore import Qt, Signal, QEvent
from PySide6.QtWidgets import QGraphicsView
from .block_catalogue import dropped_block


class CanvasView(QGraphicsView):
    zoomChanged = Signal(float)
    panModeChanged = Signal(bool)
    blockInsertRequested = Signal(str, object)
    quickInsertRequested = Signal(object, object)

    def __init__(self, scene):
        super().__init__(scene)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)
        self.pan_mode = False
        self._space = False
        self._panning = False
        self._pan_moved = False
        self._pan_position = None

    def update_pan_cursor(self):
        if self._panning:self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
        elif self.pan_mode or self._space:self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        else:self.viewport().unsetCursor()

    def set_pan_mode(self,enabled):
        enabled=bool(enabled)
        if self.pan_mode==enabled:return
        self.pan_mode=enabled;self._panning=False
        self.scene().cancel_wire()
        self.update_pan_cursor();self.panModeChanged.emit(enabled)

    def dragEnterEvent(self, event):
        self.dragMoveEvent(event)

    def dragMoveEvent(self, event):
        if not self.scene().locked and dropped_block(event.mimeData()):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        title = dropped_block(event.mimeData())
        if self.scene().locked or not title:
            event.ignore()
            return
        self.blockInsertRequested.emit(title, self.mapToScene(event.position().toPoint()))
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()

    def request_quick_insert(self, point):
        if not self.scene().locked and not self.scene().pending:
            self.quickInsertRequested.emit(self.mapToScene(point), self.viewport().mapToGlobal(point))

    def event(self, event):
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Tab and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            self.request_quick_insert(self.viewport().rect().center())
            event.accept()
            return True
        return super().event(event)

    def mouseDoubleClickEvent(self, event):
        if self.pan_mode:
            event.accept();return
        point = event.position().toPoint()
        if event.button() == Qt.MouseButton.LeftButton and self.itemAt(point) is None and not self._space and not self._panning:
            self.request_quick_insert(point)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def zoom_by(self, factor, position=None):
        if position is None:position = self.viewport().rect().center()
        before = self.mapToScene(position)
        current = self.transform().m11()
        target = max(.1, min(4., current*factor))
        self.scale(target/current, target/current)
        after = self.mapToScene(position)
        delta = after-before
        self.translate(delta.x(), delta.y())
        self.zoomChanged.emit(self.transform().m11())

    def actual_size(self):
        self.zoom_by(1/self.transform().m11())

    def fit_model(self):
        if not self.scene().items():return
        bounds = self.scene().itemsBoundingRect().adjusted(-40,-40,40,40)
        self.scene().setSceneRect(self.scene().sceneRect().united(bounds.adjusted(-1000,-1000,1000,1000)))
        self.fitInView(bounds, Qt.AspectRatioMode.KeepAspectRatio)
        self.zoomChanged.emit(self.transform().m11())

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if delta:self.zoom_by(1.2**(delta/120),event.position().toPoint())
        event.accept()

    def keyPressEvent(self, event):
        if event.key()==Qt.Key.Key_Escape and self.pan_mode:
            self.set_pan_mode(False);event.accept();return
        if event.key()==Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                self._space=True;self._pan_moved=False
                self.update_pan_cursor()
            event.accept();return
        super().keyPressEvent(event)

    def keyReleaseEvent(self,event):
        if event.key()==Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                self._space=False
                if not self._panning and not self._pan_moved:self.fit_model()
                self.update_pan_cursor()
            event.accept();return
        super().keyReleaseEvent(event)

    def mousePressEvent(self,event):
        if event.button()==Qt.MouseButton.MiddleButton or ((self._space or self.pan_mode) and event.button()==Qt.MouseButton.LeftButton):
            self._panning=True;self._pan_position=event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept();return
        super().mousePressEvent(event)

    def mouseMoveEvent(self,event):
        if self._panning:
            position=event.position().toPoint();delta=position-self._pan_position
            if delta.manhattanLength()>0:self._pan_moved=True
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value()-delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value()-delta.y())
            self._pan_position=position;event.accept();return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self,event):
        if self._panning and event.button() in (Qt.MouseButton.MiddleButton,Qt.MouseButton.LeftButton):
            self._panning=False
            self.update_pan_cursor()
            event.accept();return
        super().mouseReleaseEvent(event)

    def focusOutEvent(self,event):
        self._space=False;self._panning=False;self.update_pan_cursor()
        super().focusOutEvent(event)
