"""Small original vector glyphs, rendered using the current Margam palette."""
from PySide6.QtCore import QByteArray, QRectF
from PySide6.QtGui import QIcon, QIconEngine, QPixmap
from PySide6.QtSvg import QSvgRenderer
from .theme import colors

# 24-unit artwork; no icon font, desktop theme or network dependency.
PATHS = {
    'evaluation_points':'M4 4h16v16H4z M8 8h1v1H8z M15 7h1v1h-1z M10 15h1v1h-1z M16 16h1v1h-1z',
    'reference_data':'M4 3h16v18H4z M4 8h16 M9 8v13 M13 12h4 M13 16h4',
    'scale_recipe':'M3 3h18v18H3z M6 16h4l3-8 3 6h3',
    'scale_method':'M3 18h4l3-12 4 9 3-5h4 M4 4h4',
    'open':'M3 19V6h7l2 3h9l-3 10H3l3-8h14',
    'save':'M5 3h12l4 4v14H3V3h2 M7 3v6h10V3 M7 21v-8h10v8',
    'examples':'M3 4h7v7H3z M14 4h7v7h-7z M3 15h7v6H3z M14 15h7v6h-7z',
    'undo':'M9 4L3 10l6 6 M3 10h11a6 6 0 0 1 0 12',
    'redo':'M15 4l6 6-6 6 M21 10H10a6 6 0 0 0 0 12',
    'copy':'M8 8h13v13H8z M4 16H3V3h13v1',
    'cut':'M9 9l11 12 M9 15L20 3 M8 7a3 3 0 1 1-6 0 3 3 0 0 1 6 0 M8 17a3 3 0 1 1-6 0 3 3 0 0 1 6 0',
    'paste':'M9 4H4v17h16V4h-5 M9 2h6v5H9z M8 12h8 M8 16h6',
    'delete':'M3 6h18 M9 6V3h6v3 M6 6l1 15h10l1-15 M10 10v7 M14 10v7',
    'zoom_in':'M16 16l5 5 M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0 M6 10h8 M10 6v8',
    'zoom_out':'M16 16l5 5 M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0 M6 10h8',
    'fit':'M3 9V3h6 M15 3h6v6 M21 15v6h-6 M9 21H3v-6 M8 8h8v8H8z',
    'actual':'M4 4h16v16H4z M9 9h6v6H9z',
    'pan':'M7 12V5a2 2 0 0 1 4 0v6 M11 10V3a2 2 0 0 1 4 0v8 M15 10V5a2 2 0 0 1 4 0v8 M19 11a2 2 0 0 1 3 2v3c0 4-3 6-7 6h-3c-2 0-3-1-4-3l-5-7c-1-2 1-4 3-2l3 3',
    'check':'M4 12l5 5L20 5',
    'play':'M7 3l14 9-14 9z',
    'stop':'M5 5h14v14H5z',
    'expert_recipe':'M3 3h18v18H3z M7 8h10 M7 12h7 M7 16h10 M17 10v4',
    'expert_method':'M3 18l5-5 4 2 8-12 M3 5h5 M3 9h3 M18 3h3v3',
    'lra_recipe':'M3 3h18v18H3z M6 7h4l3 5h5 M6 17h4l3-5',
    'lra_method':'M3 6h18 M3 12h18 M3 18h18 M7 3v6 M12 9v6 M17 15v6',
    'sa_recipe':'M3 3h18v18H3z M6 16l4-5 4 2 4-7 M15 6h3v3',
    'sa_method':'M3 5h18 M3 12h18 M3 19h18 M7 2v6 M16 9v6 M10 16v6 M20 15v7',
    'recipe':'M3 3h18v18H3z M7 7h10v10H7z M3 12h4 M17 12h4',
    'domain':'M3 4h18v16H3z M3 16h4 M7 20v-4',
    'interior':'M5 5h1 M11 5h1 M17 5h1 M5 11h1 M11 11h1 M17 11h1 M5 17h1 M11 17h1 M17 17h1',
    'rar':'M3 3h18v18H3z M7 7h1 M15 7h1 M7 15h1 M12 15h6 M15 12v6',
    'importance':'M3 3h18v18H3z M7 7h1 M15 7h1 M7 15h1 M12 12h6v6h-6z',
    'boundary':'M4 4h16v16H4z M2 4h4 M18 4h4 M2 20h4 M18 20h4 M4 10v4 M20 10v4',
    'network':'M5 5l7 7-7 7 M5 5l14 0-7 7 7 7H5 M12 12h7 M3 3h4v4H3z M3 17h4v4H3z M10 10h4v4h-4z M17 3h4v4h-4z M17 17h4v4h-4z',
    'pde':'M19 4H6l7 8-7 8h13',
    'predefined_pde':'M4 3h16v18H4z M16 7H8l4 5-4 5h8',
    'bc':'M5 3v18 M10 8h11 M10 16h11 M17 4l4 4-4 4',
    'method':'M3 5h18 M3 12h18 M3 19h18 M7 2v6 M16 9v6 M10 16v6',
    'solution':'M3 3v18h18 M6 16l4-6 5 3 5-8',
    'monitor':'M3 4h18v14H3z M8 22h8 M12 18v4 M5 12h3l2-5 4 8 2-4h3',
    'reference':'M5 3h14v18H5z M8 7h8 M8 11h8 M8 15h5',
    'comparison':'M4 4h6v16H4z M14 4h6v16h-6z M10 12h4',
    'paraview':'M3 3h12v12H3z M8 8h13v13H8z M3 3l5 5 M15 3l6 5 M3 15l5 6 M15 15l6 6',
    'export':'M13 3H4v18h16v-9 M10 14L21 3 M14 3h7v7',
    'help':'M9 8a3 3 0 1 1 5 2c-2 1-2 2-2 4 M12 18h.01 M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0',
    'quit':'M10 3H3v18h7 M9 12h13 M17 7l5 5-5 5',
}


class GlyphEngine(QIconEngine):
    def __init__(self,name):
        super().__init__();self.name=name

    def clone(self):return GlyphEngine(self.name)

    def paint(self,painter,rect,mode,state):
        color=colors()['muted' if mode==QIcon.Mode.Disabled else 'text']
        if mode==QIcon.Mode.Selected:color=colors()['highlight_text']
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="{PATHS[self.name]}" fill="none" stroke="{color}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        painter.save()
        side=min(rect.width(),rect.height())
        target=QRectF(rect.x()+(rect.width()-side)/2,rect.y()+(rect.height()-side)/2,side,side)
        QSvgRenderer(QByteArray(svg.encode())).render(painter,target)
        painter.restore()

    def pixmap(self,size,mode,state):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPainter
        result=QPixmap(size);result.fill(Qt.GlobalColor.transparent)
        painter=QPainter(result);self.paint(painter,result.rect(),mode,state);painter.end()
        return result


PATHS['interior_data']=PATHS['evaluation_points']
PATHS['boundary_data']=PATHS['reference_data']

PATHS['gpinn_recipe']='M3 3h18v18H3z M6 17l4-10 4 7 4-5 M6 12h12'
PATHS['gpinn_method']=PATHS['gpinn_recipe']

def icon(name):return QIcon(GlyphEngine(name))
