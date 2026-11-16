"""Explicit two-run overlay of saved monitor histories; no current-model mutation."""
import math
from PySide6.QtCore import Qt,QRectF,QPointF
from PySide6.QtGui import QPainter,QPainterPath,QPen,QColor
from PySide6.QtWidgets import QComboBox,QCheckBox,QLabel,QTableWidgetItem,QWidget
from .run_comparison import RunComparison
from .study_results import Reader
from .theme import colors
from ..history_comparison import read_histories,available_curves,curve_label,compare_histories


class HistoryPlot(QWidget):
    def __init__(self,parent=None):super().__init__(parent);self.comparison=None;self.setMinimumHeight(260)
    def set_comparison(self,value):self.comparison=value;self.update()
    def paintEvent(self,event):
        p=QPainter(self);p.setRenderHint(QPainter.RenderHint.Antialiasing);p.setPen(QColor(colors()['text']))
        if self.comparison is None:p.drawText(18,24,'Choose saved monitors and compare a recorded term.');return
        data=self.comparison;log=data['logarithmic'];curves=[(side,data[side.lower()]['segments'],color) for side,color in [('A','#176b73'),('B','#d58b35')]]
        all_rows=[(x,math.log10(y) if log else y) for _,segments,_ in curves for segment in segments for x,y in segment]
        p.drawText(18,22,'Log10 value' if log else 'Recorded value');p.setPen(QColor('#176b73'));p.drawText(190,22,'A');p.setPen(QColor('#d58b35'));p.drawText(240,22,'B')
        if not all_rows:
            p.setPen(QColor(colors()['text']));p.drawText(18,62,'No plottable values on this scale. Zero/negative values remain in the table.');return
        xs,ys=zip(*all_rows);xmin,xmax=min(xs),max(xs);ymin,ymax=min(ys),max(ys);scale=max(abs(ymin),abs(ymax)) or 1.;lo,hi=ymin/scale,ymax/scale
        area=QRectF(90,40,max(50,self.width()-125),max(35,self.height()-90));p.setPen(QPen(QColor(colors()['border']),1));p.drawRect(area)
        def mapped(x,y):return QPointF(area.center().x() if xmin==xmax else area.left()+(x-xmin)/(xmax-xmin)*area.width(),area.center().y() if lo==hi else area.bottom()-(y/scale-lo)/(hi-lo)*area.height())
        for name,segments,color in curves:
            p.setPen(QPen(QColor(color),2));total=sum(map(len,segments));stride=max(1,math.ceil(total/2000));position=0
            for segment in segments:
                path=QPainterPath();selected=[]
                for x,y in segment:
                    if position%stride==0 or position==total-1:selected.append(mapped(x,math.log10(y) if log else y))
                    position+=1
                for index,point in enumerate(selected):
                    if index==0:path.moveTo(point)
                    else:path.lineTo(point)
                if selected:p.drawPath(path)
                if len(selected)==1:p.drawEllipse(selected[0],2.,2.)
        p.setPen(QColor(colors()['text']));p.drawText(5,int(area.top()+8),f'{ymax:.4g}');p.drawText(5,int(area.bottom()),f'{ymin:.4g}')
        p.drawText(int(area.left()),int(area.bottom()+20),str(xmin));p.drawText(int(area.right()-45),int(area.bottom()+20),str(xmax))
        p.drawText(18,self.height()-6,'Recorded update / attempt · gaps preserved · plot may show a bounded subset')


class HistoryComparison(RunComparison):
    def __init__(self,owner):
        super().__init__(owner);self.setWindowTitle('Compare saved training histories')
        # Reuse the concrete two-file reader lifecycle, not prediction alignment.
        note=self.findChildren(QLabel)[0];note.setText('Overlay explicitly retained Training Monitor data from two saved runs. Update indices and missing terms stay unchanged; no checkpoint history is substituted and no current model is loaded.')
        for combo in self.choices:combo.currentIndexChanged.connect(lambda *_:self.refresh_terms())
        self.term=QComboBox();self.term.setMinimumWidth(280);self.term.currentIndexChanged.connect(self.invalidate)
        self.log_scale=QCheckBox('Log10 scale');self.log_scale.setChecked(True);self.log_scale.toggled.connect(self.invalidate)
        # Add controls as their own row to keep the inherited two-file layout stable.
        from PySide6.QtWidgets import QHBoxLayout
        row=QHBoxLayout();row.addWidget(QLabel('Recorded term'));row.addWidget(self.term,1);row.addWidget(self.log_scale);self.layout().insertLayout(3,row)
        self.compare_button.setText('Overlay selected histories');self.metrics.setColumnCount(6);self.metrics.setHorizontalHeaderLabels(['Run','Retained records','Term present','Missing term','Hidden on log scale','Last recorded value'])
        self.plot=HistoryPlot(self);self.layout().insertWidget(self.layout().indexOf(self.preview),self.plot,3)
        self.preview.setMaximumHeight(160);self.details.setMaximumHeight(125)
        self.resize(1120,900)
    def invalidate(self,*_):
        super().invalidate()
        if hasattr(self,'plot'):self.plot.set_comparison(None)
        self.status.setText('Choose monitors and a recorded term, then overlay.')
    def buttons(self):
        super().buttons()
        if hasattr(self,'term'):
            self.term.setEnabled(self.worker is None);self.log_scale.setEnabled(self.worker is None)
            self.compare_button.setEnabled(self.compare_button.isEnabled() and self.term.currentData() is not None)
    def choose_file(self,index,path=None):
        from pathlib import Path
        from PySide6.QtWidgets import QFileDialog
        if self.worker is not None or index not in (0,1):return False
        if path is None:path,_=QFileDialog.getOpenFileName(self,'Choose saved monitor run','','Margam run (*.margam-run.pt)')
        if not path:return False
        self.runs[index]=None;self.choices[index].clear();self.term.clear();self.labels[index].setText(('A' if index==0 else 'B')+': '+Path(path).name);self.invalidate()
        self.pending=index;self.worker=Reader(lambda check:read_histories(path,check=check),self);self.worker.finished.connect(self.read_finished)
        self.status.setText('Reading retained monitor records from a restricted CPU snapshot…');self.buttons();self.worker.start();return True
    def read_finished(self):
        worker=self.sender()
        if worker is None or worker is not self.worker:return
        self.worker=None
        if worker.error:self.status.setText('History could not be read: '+worker.error)
        elif worker.abort_requested:self.status.setText('History reading cancelled.')
        else:
            run=worker.value;index=self.pending;self.runs[index]=run;combo=self.choices[index];combo.blockSignals(True)
            for identifier,records in run['monitors'].items():combo.addItem(run['titles'].get(identifier,identifier)+' · '+identifier[:8]+f' · {len(records)} records',identifier)
            combo.blockSignals(False);self.labels[index].setText(('A' if index==0 else 'B')+f": {run['name']} · {run['steps']} steps · {run['device']}"+(' · partial' if run['cancelled'] else ''))
            self.status.setText('Select monitors and a recorded term.' if run['monitors'] else 'No saved Training Monitor data. Checkpoint loss history is not substituted.')
        worker.value=None;worker.deleteLater();self.refresh_terms();self.buttons()
    def refresh_terms(self):
        if not hasattr(self,'term'):return
        previous=self.term.currentData();self.term.blockSignals(True);self.term.clear()
        if all(self.runs) and all(combo.currentData() is not None for combo in self.choices):
            for key in available_curves(self.runs[0],self.choices[0].currentData(),self.runs[1],self.choices[1].currentData()):self.term.addItem(curve_label(key),key)
        self.term.setCurrentIndex(max(0,self.term.findData(previous)));self.term.blockSignals(False)
    def compare(self):
        if self.worker is not None:return False
        self.invalidate();self.refresh_terms()
        if not all(self.runs) or any(combo.currentData() is None for combo in self.choices):
            self.status.setText('No saved Training Monitor data selected in one or both runs.');return False
        if self.term.currentData() is None:
            self.status.setText('Selected monitors have no recorded terms.');return False
        try:value=compare_histories(self.runs[0],self.choices[0].currentData(),self.runs[1],self.choices[1].currentData(),self.term.currentData(),logarithmic=self.log_scale.isChecked())
        except (ValueError,KeyError,TypeError) as error:self.status.setText('History overlay unavailable: '+str(error));return False
        self.comparison=value;self.plot.set_comparison(value)
        for index,side in enumerate(('a','b')):
            record=value[side];self.metrics.insertRow(index)
            cells=[side.upper(),record['recorded'],record['present'],record['missing'],record['hidden_nonpositive'],record['last_value'] if record['last_value'] is not None else 'Not recorded']
            for column,item in enumerate(cells):self.metrics.setItem(index,column,QTableWidgetItem(str(item)))
        self.preview.setColumnCount(3);self.preview.setHorizontalHeaderLabels(['Recorded update / attempt','A · exact value','B · exact value']);self.preview.setRowCount(len(value['preview']))
        for row,items in enumerate(value['preview']):
            for column,item in enumerate(items):self.preview.setItem(row,column,QTableWidgetItem('Not recorded' if item is None else str(item)))
        self.status.setText(f"{value['label']} · preview {len(value['preview'])} of {value['preview_total']} recorded indices. Gaps are not filled; nonpositive values are hidden only on log scale.")
        self.details.setPlainText(value['scope']+'\nChanged configuration keys: '+(', '.join(value['configuration_differences']) or 'none')+'\nA retained range: '+str((value['a']['first_step'],value['a']['last_step']))+' · B retained range: '+str((value['b']['first_step'],value['b']['last_step']))+'\nSnapshot SHA256 A: '+self.runs[0]['sha256']+'\nSnapshot SHA256 B: '+self.runs[1]['sha256'])
        return True
