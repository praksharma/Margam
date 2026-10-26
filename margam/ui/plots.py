"""Native 2D sample, convergence and solution views; no plotting dependency."""
import math
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QImage
from PySide6.QtWidgets import QWidget
from .theme import colors
from ..templates.poisson import sample_problem


class RectanglePreview(QWidget):
    """A rectangle block's local 2D view; not a general geometry renderer."""
    def __init__(self, config, show_samples=True, parent=None):
        super().__init__(parent)
        if config.shape != 'Rectangle':
            raise ValueError('This preview supports rectangle blocks only.')
        self.config = config
        self.show_samples = show_samples
        self.setMinimumHeight(240)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QColor(colors()['text']))
        c = self.config
        try:
            if c.shape != 'Rectangle':
                raise ValueError('This preview supports rectangle blocks only.')
            if not all(math.isfinite(v) for v in (c.x_min,c.x_max,c.y_min,c.y_max)) or c.x_max<=c.x_min or c.y_max<=c.y_min:
                raise ValueError('Rectangle bounds must be finite and strictly increasing.')
            interior,boundaries = sample_problem(c) if self.show_samples else ([],{})
        except ValueError as error:
            p.drawText(self.rect().adjusted(20,20,-20,-20),Qt.TextFlag.TextWordWrap,str(error))
            return
        width,height = c.x_max-c.x_min,c.y_max-c.y_min
        h=max(40,self.height()-110)
        w=min(max(80,self.width()-160),h*width/height)
        h=min(h,w*height/width)
        area=QRectF(70,48,w,h)
        caption=f'Rectangle · {len(interior)} interior · {sum(map(len,boundaries.values()))} boundary points' if self.show_samples else f'Rectangle · width {width:g} · height {height:g}'
        p.drawText(18,24,caption)
        p.setPen(QPen(QColor(colors()['border']),2));p.drawRect(area)
        def mapped(row):
            return QPointF(area.left()+(row[0]-c.x_min)/width*w,area.bottom()-(row[1]-c.y_min)/height*h)
        if self.show_samples:
            p.setPen(QPen(QColor(colors()['muted']),3))
            stride=max(1,math.ceil(len(interior)/1500))
            for row in interior[::stride]:p.drawPoint(mapped(row))
            for side,rows in boundaries.items():
                color='#d58b35' if c.bc_types.get(side)=='Neumann' else colors()['highlight']
                p.setPen(QPen(QColor(color),4))
                stride=max(1,math.ceil(len(rows)/500))
                for row in rows[::stride]:p.drawPoint(mapped(row))
        p.setPen(QColor(colors()['text']))
        p.drawText(12,int(area.center().y()),'Left')
        p.drawText(int(area.right()+12),int(area.center().y()),'Right')
        p.drawText(int(area.center().x()-15),int(area.top()-8),'Top')
        p.drawText(int(area.center().x()-20),int(area.bottom()+22),'Bottom')
        footer='Teal: Dirichlet · orange: outward derivative · large clouds shown as a subset' if self.show_samples else f'x: [{c.x_min:g}, {c.x_max:g}] · y: [{c.y_min:g}, {c.y_max:g}]'
        p.drawText(18,self.height()-10,footer)


class CurvePlot(QWidget):
    def __init__(self, logarithmic=False, parent=None):
        super().__init__(parent)
        self.series = []
        self.logarithmic = logarithmic
        self.caption = 'Run training to see results.'
        self.setMinimumHeight(190)

    def set_series(self, series, caption):
        self.series,self.caption = series,caption
        self.update()

    def paintEvent(self,event):
        p=QPainter(self);p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QColor(colors()['text']));p.drawText(18,22,self.caption)
        series=[(name,[(float(x),math.log10(max(float(y),1e-16)) if self.logarithmic else float(y)) for x,y in rows if math.isfinite(x) and math.isfinite(y)],color) for name,rows,color in self.series]
        rows=[row for _,data,_ in series for row in data]
        if not rows:return
        xs,ys=zip(*rows);xmin,xmax=min(xs),max(xs);ymin,ymax=min(ys),max(ys)
        if xmax==xmin:xmax=xmin+1
        if ymax==ymin:ymin-=.5;ymax+=.5
        margin=(ymax-ymin)*.05;ymin-=margin;ymax+=margin
        area=QRectF(72,52,max(50,self.width()-110),max(30,self.height()-95))
        p.setPen(QPen(QColor(colors()['border']),1));p.drawRect(area)
        def mapped(x,y):return QPointF(area.left()+(x-xmin)/(xmax-xmin)*area.width(),area.bottom()-(y-ymin)/(ymax-ymin)*area.height())
        for name,data,color in series:
            p.setPen(QPen(QColor(color),2))
            path=QPainterPath()
            data=data[::max(1,len(data)//2000)]
            for i,(x,y) in enumerate(data):
                if i==0:path.moveTo(mapped(x,y))
                else:path.lineTo(mapped(x,y))
            p.drawPath(path)
        p.setPen(QColor(colors()['text']))
        p.drawText(5,int(area.top()+8),f'{10**ymax:.2g}' if self.logarithmic else f'{ymax:.3g}')
        p.drawText(5,int(area.bottom()),f'{10**ymin:.2g}' if self.logarithmic else f'{ymin:.3g}')
        p.drawText(int(area.left()),int(area.bottom()+22),f'{xmin:.3g}')
        p.drawText(int(area.right()-35),int(area.bottom()+22),f'{xmax:.3g}')
        offset=76
        for name,_,color in series:
            p.setPen(QColor(color));p.drawText(offset,42,name);offset+=130


class SolutionView(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QStackedLayout
        self.curve=CurvePlot()
        self.field=FieldPlot()
        self.stack=QStackedLayout(self)
        self.stack.addWidget(self.curve);self.stack.addWidget(self.field)

    def clear(self):
        self.curve.set_series([],'Settings changed. Run training for a new solution.')
        self.stack.setCurrentWidget(self.curve)

    def show_result(self,result):
        if result['config'].shape=='Interval':
            xs=[p[0] for p in result['points']]
            data=[('Prediction',list(zip(xs,result['values'])),'#2b9d91')]
            caption=f"Prediction u(x) · {result['steps']} steps"
            if result.get('reference') is not None:
                data.append(('Reference',list(zip(xs,result['reference'])),'#d58b35'))
                caption+=f" · RMSE {result['rmse']:.4g}"
            self.curve.set_series(data,caption)
            self.stack.setCurrentWidget(self.curve)
        else:
            self.field.set_result(result)
            self.stack.setCurrentWidget(self.field)


class FieldPlot(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent);self.result=None;self.image=None

    def set_result(self,result):
        self.result=result
        size=result['size'];values=result['values'];lo,hi=min(values),max(values)
        self.image=QImage(size,size,QImage.Format.Format_RGB32)
        for j in range(size):
            for i in range(size):
                t=(values[j*size+i]-lo)/(hi-lo) if hi>lo else .5
                self.image.setPixelColor(i,size-j-1,QColor.fromHsvF((1-t)*.65,.75,.95))
        self.update()

    def paintEvent(self,event):
        p=QPainter(self);p.setPen(QColor(colors()['text']))
        if self.result is None:
            p.drawText(20,25,'Run training to see the predicted field.');return
        c=self.result['config'];values=self.result['values']
        p.drawText(20,22,f"Predicted u(x,y) · {self.result['steps']} steps · range [{min(values):.4g}, {max(values):.4g}] · prediction only")
        ratio=(c.x_max-c.x_min)/(c.y_max-c.y_min)
        h=max(30,self.height()-80);w=min(max(30,self.width()-130),h*ratio);h=min(h,w/ratio)
        rect=QRectF(65,40,w,h)
        p.drawImage(rect,self.image)
        p.drawText(15,48,f'{c.y_max:g}')
        p.drawText(15,int(rect.bottom()),f'{c.y_min:g}')
        p.drawText(65,int(rect.bottom()+23),f'{c.x_min:g}')
        p.drawText(int(rect.right()-25),int(rect.bottom()+23),f'{c.x_max:g}')
        p.drawText(int(rect.right()+18),65,'Blue: low')
        p.drawText(int(rect.right()+18),87,'Red: high')
