"""Explicit, exact-row comparison of saved native predictions; no interpolation."""
from dataclasses import asdict
import hashlib
import math
import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
from .components.error_norms import array,error_norms


def check_cancel(check):
    if check is not None and check():raise InterruptedError('Run comparison cancelled.')


def read_predictions(path,*,check=None):
    """Copy a bounded stable archive, validate CPU payload, retain no checkpoint."""
    from .ui.run_document import LIMIT,load_run_payload
    from .execution.study_state import safe_path
    path=Path(path).absolute();safe_path(path.parent,path.name);check_cancel(check)
    with TemporaryDirectory(prefix='margam-run-comparison-') as directory:
        frozen=Path(directory)/'run.pt';hasher=hashlib.sha256();size=0
        with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as source,frozen.open('xb') as target:
            before=os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size>LIMIT:raise ValueError('Choose a regular native run archive within 256 MiB.')
            while chunk:=source.read(1024*1024):
                check_cancel(check);size+=len(chunk)
                if size>LIMIT:raise ValueError('Run exceeds 256 MiB.')
                hasher.update(chunk);target.write(chunk)
            after=os.fstat(source.fileno())
            if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('Archive changed while reading.')
        check_cancel(check);data,config=load_run_payload(frozen);check_cancel(check)
        titles={node['id']:node['title'] for node in data['model']['nodes']}
        return dict(name=path.name,sha256=hasher.hexdigest(),coordinates=list(config.coordinate_names),config=asdict(config),
            outputs=data['result']['outputs'],titles=titles,steps=data['result']['steps'],cancelled=data['result']['cancelled'],device=data['result']['device'])


def compare_predictions(left,left_id,right,right_id):
    """B−A on identically ordered coordinates/fields/IDs; metrics cover every row."""
    import numpy as np
    if left['coordinates']!=right['coordinates']:raise ValueError('Coordinate names/order differ. Re-evaluate on the same coordinate contract.')
    a=left['outputs'][left_id];b=right['outputs'][right_id];fields=a.get('fields',['u'])
    if fields!=b.get('fields',['u']):raise ValueError('Field names/order differ; no implicit field mapping is performed.')
    points=array(a['points']);other=array(b['points'])
    if points.shape[1]!=len(left['coordinates']) or points.shape!=other.shape or a['points']!=b['points'] or not np.array_equal(points,other):raise ValueError('Evaluation coordinates/order differ. Re-evaluate both runs on the same points; no interpolation or row sorting is performed.')
    ids=a.get('point_ids');other_ids=b.get('point_ids')
    if (ids is None)!=(other_ids is None) or ids is not None and (len(ids)!=len(points) or len(ids)!=len(other_ids) or any(type(x) is not type(y) or x!=y for x,y in zip(ids,other_ids))):raise ValueError('Point IDs/types/order differ; rows are not automatically matched.')
    values_a=array(a['values']);values_b=array(b['values'])
    if values_a.shape!=values_b.shape or values_a.shape!=(len(points),len(fields)):raise ValueError('Prediction shape differs from the point/field contract.')
    references=[]
    for data in (a,b):
        ref=None if data.get('reference') is None else array(data['reference'])
        if ref is not None and ref.shape!=values_a.shape:raise ValueError('Reference shape differs from predictions.')
        references.append(ref)
    diagnostics=[]
    for field in fields:
        record=dict(field=field,rms_difference=error_norms(values_b,values_a,fields=fields,selected=[field],norm='rms'),
            max_difference=error_norms(values_b,values_a,fields=fields,selected=[field],norm='max'),
            relative_difference=error_norms(values_b,values_a,fields=fields,selected=[field],norm='relative_l2'))
        for side,values,reference in zip(('a','b'),(values_a,values_b),references):
            record['reference_'+side]=None if reference is None else error_norms(values,reference,fields=fields,selected=[field])
        diagnostics.append(record)
    headers=list(left['coordinates'])+(['Point ID'] if ids is not None else [])
    for field in fields:headers.extend([field+' · A',field+' · B',field+' · B−A'])
    preview=[]
    for index in range(min(200,len(points))):
        row=points[index].tolist()+([ids[index]] if ids is not None else [])
        for column in range(len(fields)):
            x=float(values_a[index,column]);y=float(values_b[index,column]);difference=y-x
            row.extend([x,y,difference if math.isfinite(difference) else 'overflow'])
        preview.append(row)
    changed=sorted(key for key in left['config'].keys()|right['config'].keys() if left['config'].get(key)!=right['config'].get(key))
    return dict(rows=len(points),fields=fields,diagnostics=diagnostics,headers=headers,preview=preview,configuration_differences=changed,
        identical_reference_values=references[0] is not None and references[1] is not None and a['reference']==b['reference'],
        scope='Prediction differences are not accuracy claims. Relative difference uses A as denominator. Own reference errors are separate; no unit or physics equivalence is inferred.')
