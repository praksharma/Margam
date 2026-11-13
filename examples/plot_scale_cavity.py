"""Plot independently loaded cavity checkpoints against local CFD centerline cuts.

Uses existing QtCharts for standalone scientific figures, not an in-app heatmap.
CFD fields remain local; global error metrics use the original supplied rows.
"""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from margam.research.scale_cavity import ScaleCavityNetwork
from examples.scale_cavity_benchmark import load_reference,evaluate


def inspect_run(directory,reference_directory,*,best=False):
    directory=Path(directory);prefix='best-' if best else '';report=json.loads((directory/(prefix+'report.json')).read_text())
    state=torch.load(directory/(prefix+'checkpoint.pt'),map_location='cpu',weights_only=True)
    if state.get('format')!='margam-scale-cavity-research' or state.get('version') not in (1,2) or state['step']!=report['steps']:raise ValueError('Report/checkpoint identity mismatch.')
    settings=state['signature']['protocol'];network=ScaleCavityNetwork(settings['width']).float();network.load_state_dict(state['network']);network.eval()
    data,wall,corner,identity=load_reference(reference_directory,settings['reynolds'])
    if identity!=state['signature']['reference']:raise ValueError('Reference bytes differ from the trained case.')
    metrics,prediction=evaluate(network,data,wall,corner)
    old=report['metrics']['raw_without_corners'];new=metrics['raw_without_corners']
    if not np.isclose(old['velocity_relative_l2'],new['velocity_relative_l2'],rtol=1e-4,atol=1e-8):raise ValueError('Fresh CPU inference disagrees with the recorded GPU metric.')
    return network,data,report,metrics,prediction


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--reference-directory',type=Path,default=Path.cwd().parent/'SCALE-PINN'/'ns-ldc');p.add_argument('--output',type=Path,required=True);p.add_argument('--vtu',type=Path);p.add_argument('--best',action='store_true',help='Plot the explicitly reference-selected checkpoint instead of the final iterate.');a=p.parse_args()
    if a.output.exists() or a.vtu and a.vtu.exists():p.error('Choose fresh figure/export filenames.')
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen');torch.set_num_threads(1)
    network,data,report,metrics,prediction=inspect_run(a.run,a.reference_directory,best=a.best)
    if a.vtu:
        from margam.components.point_export import export_vtu
        export_vtu(a.vtu,dict(points=data[:,:2].tolist(),values=prediction.tolist(),fields=['u','v','p']))
    xs,ys=np.unique(data[:,0]),np.unique(data[:,1]);ordered=data[np.lexsort((data[:,0],data[:,1]))]
    u=ordered[:,2].reshape(len(ys),len(xs));v=ordered[:,3].reshape(len(ys),len(xs))
    u_ref=np.array([np.interp(.5,xs,row) for row in u]);v_ref=np.array([np.interp(.5,ys,v[:,column]) for column in range(len(xs))])
    axis=np.linspace(0,1,201)
    with torch.no_grad():
        u_pred=network(torch.tensor(np.column_stack((np.full_like(axis,.5),axis)),dtype=torch.float32))[:,0].numpy()
        v_pred=network(torch.tensor(np.column_stack((axis,np.full_like(axis,.5))),dtype=torch.float32))[:,1].numpy()
    from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout,QHBoxLayout,QLabel
    from PySide6.QtCharts import QChart,QChartView,QLineSeries,QValueAxis
    from PySide6.QtCore import Qt,QMargins
    from PySide6.QtGui import QPainter,QColor,QPen,QFont
    app=QApplication([]);app.setStyle('Fusion');window=QWidget();window.setStyleSheet('background:white;color:#172b36;');layout=QVBoxLayout(window)
    setting=report['settings'];error=report['metrics']['raw_without_corners']['velocity_relative_l2']
    updates=report.get('total_updates',report['steps']);phase=' (warm restart)' if report.get('initialization',{}).get('kind')=='warm_start_reset_adam' else ''
    if a.best:phase+=' [reference-selected]'
    title=QLabel(f"SCALE-PINN cavity: Re={setting['reynolds']}, seed={setting['seed']}, {updates:,} updates{phase} — raw velocity L2={100*error:.2f}%")
    title.setFont(QFont('Sans Serif',13));layout.addWidget(title);row=QHBoxLayout();layout.addLayout(row)
    for name,px,py,rx,ry,xlabel,ylabel in [('Vertical centerline',u_pred,axis,u_ref,ys,'u at x=0.5','y'),('Horizontal centerline',axis,v_pred,xs,v_ref,'x','v at y=0.5')]:
        chart=QChart();chart.setTitle(name);chart.setMargins(QMargins(12,12,12,12));chart.setBackgroundBrush(QColor('white'));chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
        for label,x,y,color,dashed in [('CFD reference (linear cut)',rx,ry,'#303b42',True),('Margam raw prediction',px,py,'#007f91',False)]:
            series=QLineSeries();series.setName(label)
            for a_x,a_y in zip(x,y):series.append(float(a_x),float(a_y))
            pen=QPen(QColor(color));pen.setWidthF(2.);pen.setStyle(Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine);series.setPen(pen);chart.addSeries(series)
        ax=QValueAxis();ay=QValueAxis();ax.setTitleText(xlabel);ay.setTitleText(ylabel);ax.setTickCount(6);ay.setTickCount(6);ax.setLabelFormat('%.2f');ay.setLabelFormat('%.2f')
        chart.addAxis(ax,Qt.AlignmentFlag.AlignBottom);chart.addAxis(ay,Qt.AlignmentFlag.AlignLeft)
        for series in chart.series():series.attachAxis(ax);series.attachAxis(ay)
        ax.setRange(-.5,1.05) if xlabel.startswith('u') else ax.setRange(0,1)
        ay.setRange(0,1) if ylabel=='y' else ay.setRange(-.7,.7)
        view=QChartView(chart);view.setRenderHint(QPainter.RenderHint.Antialiasing);row.addWidget(view)
    note=QLabel('No boundary replacement. CFD is interpolated only for these centerline cuts; reported global L2 uses original CFD rows. Pressure gauge is not compared.')
    note.setWordWrap(True);layout.addWidget(note);window.resize(1280,650);window.show()
    for _ in range(5):app.processEvents()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if not window.grab().save(str(a.output)):raise RuntimeError('Could not save centerline figure.')
    window.close();app.processEvents();print(json.dumps(dict(figure=a.output.name,fresh_cpu_metrics=metrics['raw_without_corners']),indent=2))


if __name__=='__main__':main()
