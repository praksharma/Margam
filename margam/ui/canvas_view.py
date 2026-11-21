"""2D model navigation: pointer-centred wheel zoom, pan, and explicit fitting."""
from PySide6.QtCore import Qt, Signal, QEvent, QPointF, QRectF
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QGraphicsView
from .block_catalogue import dropped_block
from .theme import colors


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
        self.show_minimap = False
        self._minimap_dragging = False
        self._space = False
        self._panning = False
        self._pan_moved = False
        self._pan_position = None

    def set_minimap_visible(self, visible):
        self.show_minimap = bool(visible)
        self._minimap_dragging = False
        self.viewport().update()

    def _minimap_geometry(self):
        vw = max(200, self.viewport().width())
        vh = max(140, self.viewport().height())
        panel = QRectF(vw - 184, vh - 124, 172, 112)
        inner = panel.adjusted(6, 6, -6, -6)
        view_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        items_rect = self.scene().itemsBoundingRect().adjusted(-60, -60, 60, 60) if self.scene() and self.scene().items() else view_rect
        world = items_rect.united(view_rect)
        if world.width() < 200:
            cx = world.center().x()
            world.setLeft(cx - 100); world.setRight(cx + 100)
        if world.height() < 200:
            cy = world.center().y()
            world.setTop(cy - 100); world.setBottom(cy + 100)
        scale = min(inner.width() / world.width(), inner.height() / world.height())
        draw_w = world.width() * scale
        draw_h = world.height() * scale
        draw_rect = QRectF(inner.center().x() - draw_w / 2, inner.center().y() - draw_h / 2, draw_w, draw_h)
        return panel, draw_rect, world, scale

    def _minimap_point_to_scene(self, pos):
        _, draw_rect, world, scale = self._minimap_geometry()
        if scale <= 0:
            return world.center()
        sx = world.left() + (pos.x() - draw_rect.left()) / scale
        sy = world.top() + (pos.y() - draw_rect.top()) / scale
        return QPointF(sx, sy)

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if not self.show_minimap or self.scene() is None or not self.scene().items():
            return
        panel, draw_rect, world, scale = self._minimap_geometry()
        c = colors()
        painter.save()
        painter.resetTransform()
        bg = QColor(c['base']); bg.setAlpha(225)
        painter.setBrush(bg)
        painter.setPen(QPen(QColor(c['border']), 1))
        painter.drawRoundedRect(panel, 4, 4)
        for item in self.scene().items():
            if hasattr(item, 'kind') and hasattr(item, 'sceneBoundingRect'):
                r = item.sceneBoundingRect()
                mr = QRectF(
                    draw_rect.left() + (r.left() - world.left()) * scale,
                    draw_rect.top() + (r.top() - world.top()) * scale,
                    max(2.0, r.width() * scale),
                    max(2.0, r.height() * scale),
                )
                fill = QColor(c['highlight']) if item.isSelected() else QColor(c['muted'])
                fill.setAlpha(180)
                painter.setBrush(fill)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(mr)
        vr = self.mapToScene(self.viewport().rect()).boundingRect()
        mvr = QRectF(
            draw_rect.left() + (vr.left() - world.left()) * scale,
            draw_rect.top() + (vr.top() - world.top()) * scale,
            max(4.0, vr.width() * scale),
            max(4.0, vr.height() * scale),
        ).intersected(panel.adjusted(2, 2, -2, -2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(c['highlight']), 1.5))
        painter.drawRect(mvr)
        painter.restore()

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
        if self.show_minimap:self.viewport().update()

    def set_zoom(self, target_zoom, position=None):
        current = self.transform().m11()
        if current <= 0:return
        self.zoom_by(float(target_zoom) / current, position)

    def actual_size(self):
        self.zoom_by(1/self.transform().m11())

    def fit_model(self):
        if not self.scene().items():return
        bounds = self.scene().itemsBoundingRect().adjusted(-40,-40,40,40)
        self.scene().setSceneRect(self.scene().sceneRect().united(bounds.adjusted(-1000,-1000,1000,1000)))
        self.fitInView(bounds, Qt.AspectRatioMode.KeepAspectRatio)
        self.zoomChanged.emit(self.transform().m11())

    def fit_items(self,items):
        items=[item for item in items if item.scene() is self.scene()]
        if not items:return False
        bounds=items[0].sceneBoundingRect()
        for item in items[1:]:bounds=bounds.united(item.sceneBoundingRect())
        bounds=bounds.adjusted(-40,-40,40,40)
        self.scene().setSceneRect(self.scene().sceneRect().united(bounds.adjusted(-1000,-1000,1000,1000)))
        self.fitInView(bounds,Qt.AspectRatioMode.KeepAspectRatio)
        self.zoomChanged.emit(self.transform().m11())
        return True

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
        if self.show_minimap and event.button()==Qt.MouseButton.LeftButton and self.scene() and self.scene().items():
            panel,_,_,_=self._minimap_geometry()
            if panel.contains(event.position()):
                self._minimap_dragging=True
                self.centerOn(self._minimap_point_to_scene(event.position()))
                self.viewport().update()
                event.accept();return
        if event.button()==Qt.MouseButton.MiddleButton or ((self._space or self.pan_mode) and event.button()==Qt.MouseButton.LeftButton):
            self._panning=True;self._pan_position=event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept();return
        super().mousePressEvent(event)

    def mouseMoveEvent(self,event):
        if self._minimap_dragging:
            self.centerOn(self._minimap_point_to_scene(event.position()))
            self.viewport().update()
            event.accept();return
        if self._panning:
            position=event.position().toPoint();delta=position-self._pan_position
            if delta.manhattanLength()>0:self._pan_moved=True
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value()-delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value()-delta.y())
            self._pan_position=position
            if self.show_minimap:self.viewport().update()
            event.accept();return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self,event):
        if self._minimap_dragging and event.button()==Qt.MouseButton.LeftButton:
            self._minimap_dragging=False
            self.viewport().update()
            event.accept();return
        if self._panning and event.button() in (Qt.MouseButton.MiddleButton,Qt.MouseButton.LeftButton):
            self._panning=False
            self.update_pan_cursor()
            event.accept();return
        super().mouseReleaseEvent(event)

    def focusOutEvent(self,event):
        self._space=False;self._panning=False;self.update_pan_cursor()
        super().focusOutEvent(event)
