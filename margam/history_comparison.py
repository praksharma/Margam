"""Compare explicitly retained monitor histories, without filling absent records."""
from dataclasses import asdict
from pathlib import Path
from .run_comparison import _read_snapshot


def read_histories(path,*,check=None):
    data,config,digest=_read_snapshot(path,check=check)
    titles={node['id']:node['title'] for node in data['model']['nodes']}
    return dict(name=Path(path).name,sha256=digest,config=asdict(config),monitors=data['monitors'],titles=titles,
        steps=data['result']['steps'],cancelled=data['result']['cancelled'],device=data['result']['device'])


def curve_label(key):
    if key=='total':return 'Total objective (recorded)'
    if key=='pde':return 'PDE (raw)'
    if key=='pde_weighted':return 'PDE attention-weighted (before global coefficient)'
    if key.startswith('weighted:'):return key.removeprefix('weighted:')+' · objective contribution (recorded)'
    return key+' (raw)'


def available_curves(left,left_monitor,right,right_monitor):
    return sorted({key for run,identifier in ((left,left_monitor),(right,right_monitor)) for _,row in run['monitors'][identifier] for key in row},key=lambda key:(key!='total',key!='pde',key))


def retained_curve(records,key,logarithmic):
    """Split at absent updates/terms and nonpositive values on log scale."""
    import math
    if type(logarithmic) is not bool:raise ValueError('Choose linear or logarithmic scale.')
    segments=[];current=[];previous=None;missing=0;hidden=0;values=[];table=[]
    for step,row in records:
        if type(step) is not int or step<1 or previous is not None and step<=previous:raise ValueError('History update indices must be positive and strictly increasing.')
        if previous is not None and step!=previous+1:
            if current:segments.append(current);current=[]
        previous=step;value=row.get(key)
        if value is None:
            missing+=1
            if current:segments.append(current);current=[]
            continue
        if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('History values must be finite real numbers.')
        values.append((step,value));table.append((step,value))
        if logarithmic and value<=0:
            hidden+=1
            if current:segments.append(current);current=[]
        else:current.append((step,value))
    if current:segments.append(current)
    return dict(segments=segments,recorded=len(records),present=len(values),missing=missing,hidden_nonpositive=hidden,
        first_step=records[0][0] if records else None,last_step=records[-1][0] if records else None,
        last_value=values[-1][1] if values else None,values=table)


def compare_histories(left,left_monitor,right,right_monitor,key,*,logarithmic=True):
    if key not in available_curves(left,left_monitor,right,right_monitor):raise ValueError('Choose a term present in at least one selected monitor.')
    a=retained_curve(left['monitors'][left_monitor],key,logarithmic);b=retained_curve(right['monitors'][right_monitor],key,logarithmic)
    changed=sorted(name for name in left['config'].keys()|right['config'].keys() if left['config'].get(name)!=right['config'].get(name))
    # Preview exact recorded values, not interpolated/aligned or zero-padded curves.
    av=dict(a['values']);bv=dict(b['values']);indices=sorted(av.keys()|bv.keys())
    return dict(key=key,label=curve_label(key),a=a,b=b,logarithmic=logarithmic,configuration_differences=changed,
        preview=[(step,av.get(step),bv.get(step)) for step in indices[:200]],preview_total=len(indices),
        scope='Recorded updates/attempts, not physical time. Losses/objectives across methods or settings are not automatically comparable. No rescaling, interpolation, ranking or missing-history reconstruction.')
