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


def _read_snapshot(path,*,check=None):
    """Copy a bounded stable archive and validate its restricted CPU payload."""
    import errno
    from .ui.run_document import LIMIT,load_run_payload
    raw_path=Path(os.path.abspath(path))
    if not raw_path.name or raw_path.is_symlink():
        raise ValueError('Choose a regular native run archive, without a symlink.')
    try:
        parent=raw_path.parent.resolve(strict=True)
    except OSError as error:
        raise ValueError('Choose a regular results directory, without a symlink.') from error
    if not parent.is_dir():
        raise ValueError('Choose a regular results directory, without a symlink.')
    target_path=parent/raw_path.name
    if target_path.is_symlink():
        raise ValueError('Choose a regular native run archive, without a symlink.')
    check_cancel(check)
    with TemporaryDirectory(prefix='margam-run-comparison-') as directory:
        frozen=Path(directory)/'run.pt';hasher=hashlib.sha256();size=0
        try:
            fd=os.open(target_path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))
        except OSError as error:
            if error.errno==errno.ELOOP:
                raise ValueError('Choose a regular native run archive, without a symlink.') from error
            raise
        with os.fdopen(fd,'rb') as source,frozen.open('xb') as target:
            before=os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size>LIMIT:raise ValueError('Choose a regular native run archive within 256 MiB.')
            while chunk:=source.read(1024*1024):
                check_cancel(check);size+=len(chunk)
                if size>LIMIT:raise ValueError('Run exceeds 256 MiB.')
                hasher.update(chunk);target.write(chunk)
            after=os.fstat(source.fileno())
            if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('Archive changed while reading.')
        check_cancel(check);data,config=load_run_payload(frozen);check_cancel(check)
        return data,config,hasher.hexdigest()


def read_predictions(path,*,check=None):
    """Retain explicit prediction data only; never restore a network."""
    data,config,digest=_read_snapshot(path,check=check)
    titles={node['id']:node['title'] for node in data['model']['nodes']}
    return dict(name=Path(path).name,sha256=digest,coordinates=list(config.coordinate_names),config=asdict(config),
        outputs=data['result']['outputs'],titles=titles,steps=data['result']['steps'],cancelled=data['result']['cancelled'],device=data['result']['device'])


def validate_exact_alignment(
    coords_a,
    coords_b,
    points_a,
    points_b,
    *,
    fields_a=None,
    fields_b=None,
    ids_a=None,
    ids_b=None,
):
    """Enforce Margam's exact coordinate-name, coordinate-value, ID, and field contract."""
    import numpy as np
    if list(coords_a) != list(coords_b):
        raise ValueError('Coordinate names/order differ. Re-evaluate on the same coordinate contract.')
    if fields_a is not None and fields_b is not None and list(fields_a) != list(fields_b):
        raise ValueError('Field names/order differ; no implicit field mapping is performed.')
    pa = array(points_a)
    pb = array(points_b)
    raw_a = points_a.tolist() if hasattr(points_a, 'tolist') else points_a
    raw_b = points_b.tolist() if hasattr(points_b, 'tolist') else points_b
    if pa.ndim != 2 or pa.shape[1] != len(coords_a) or pa.shape != pb.shape or raw_a != raw_b or not np.array_equal(pa, pb):
        raise ValueError('Evaluation coordinates/order differ. Re-evaluate both runs on the same points; no interpolation or row sorting is performed.')
    if (ids_a is None) != (ids_b is None) or ids_a is not None and (
        len(ids_a) != len(pa)
        or len(ids_a) != len(ids_b)
        or any(type(x) is not type(y) or x != y for x, y in zip(ids_a, ids_b))
    ):
        raise ValueError('Point IDs/types/order differ; rows are not automatically matched.')
    return pa


def compare_predictions(left,left_id,right,right_id):
    """B−A on identically ordered coordinates/fields/IDs; metrics cover every row."""
    import numpy as np
    a=left['outputs'][left_id];b=right['outputs'][right_id];fields=a.get('fields',['u'])
    ids=a.get('point_ids');other_ids=b.get('point_ids')
    points=validate_exact_alignment(
        left['coordinates'],right['coordinates'],a['points'],b['points'],
        fields_a=fields,fields_b=b.get('fields',['u']),ids_a=ids,ids_b=other_ids,
    )
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
    raw_ref_a=a.get('reference');raw_ref_b=b.get('reference')
    if hasattr(raw_ref_a,'tolist'):raw_ref_a=raw_ref_a.tolist()
    if hasattr(raw_ref_b,'tolist'):raw_ref_b=raw_ref_b.tolist()
    return dict(rows=len(points),fields=fields,diagnostics=diagnostics,headers=headers,preview=preview,configuration_differences=changed,
        identical_reference_values=references[0] is not None and references[1] is not None and raw_ref_a==raw_ref_b and np.array_equal(references[0],references[1]),
        scope='Prediction differences are not accuracy claims. Relative difference uses A as denominator. Own reference errors are separate; no unit or physics equivalence is inferred.')


def compare_prediction_group(selections,*,check=None):
    """Compare two–eight explicit outputs against A using the exact-row contract."""
    if not 2<=len(selections)<=8:raise ValueError('Choose between two and eight saved prediction runs.')
    left,left_id=selections[0];comparisons=[]
    for index,(right,right_id) in enumerate(selections[1:],1):
        check_cancel(check)
        try:value=compare_predictions(left,left_id,right,right_id)
        except (ValueError,KeyError,TypeError) as error:raise ValueError(f'Run {chr(65+index)} versus A: {error}') from error
        comparisons.append(dict(run=chr(65+index),**value))
    check_cancel(check)
    first=comparisons[0];prefix=len(left['coordinates'])+(left['outputs'][left_id].get('point_ids') is not None);headers=first['headers'][:prefix]
    for field in first['fields']:
        headers.append(field+' · A')
        for item in comparisons:headers.extend([field+' · '+item['run'],field+' · '+item['run']+'−A'])
    preview=[]
    for i in range(len(first['preview'])):
        row=first['preview'][i][:prefix]
        for column in range(len(first['fields'])):
            start=prefix+3*column;row.append(first['preview'][i][start])
            for item in comparisons:row.extend(item['preview'][i][start+1:start+3])
        preview.append(row)
    return dict(rows=first['rows'],fields=first['fields'],headers=headers,preview=preview,comparisons=comparisons,
        diagnostics=[dict(run=item['run'],**record) for item in comparisons for record in item['diagnostics']],
        configuration_differences=sorted({key for item in comparisons for key in item['configuration_differences']}),
        identical_reference_values=all(item['identical_reference_values'] for item in comparisons),
        scope='Every run is compared separately with A on identical ordered coordinates, fields and typed IDs. Relative differences use A as denominator; own-reference errors remain separate. No interpolation, rescaling, automatic ranking or unit/physics equivalence is inferred.')
