"""Explicit scalar CSV export; separate from views and ParaView integration."""
import csv
import numpy as np
import os
from pathlib import Path
import tempfile


def export_csv(path,data):
    path=Path(path);temporary=None
    points,values=data['points'],data['values']
    if not points:raise ValueError('CSV requires at least one evaluation point.')
    dimension=len(points[0])
    fields=list(data.get('fields',['u']));array=np.asarray(values,dtype=float);array=array[:,None] if array.ndim==1 else array
    reference=data.get('reference');refs=None if reference is None else np.asarray(reference,dtype=float);refs=refs[:,None] if refs is not None and refs.ndim==1 else refs
    coordinates=getattr(data.get('config'),'coordinate_names',('x','y')[:dimension])
    ids=data.get('point_ids')
    if dimension not in (1,2,3) or len(coordinates)!=dimension or array.shape!=(len(points),len(fields)) or not np.isfinite(np.asarray(points,dtype=float)).all() or not np.isfinite(array).all() or refs is not None and (refs.shape!=array.shape or not np.isfinite(refs).all()) or ids is not None and len(ids)!=len(points):raise ValueError('CSV requires finite matching coordinates, fields and optional references/IDs.')
    try:
        with tempfile.NamedTemporaryFile(mode='w',newline='',encoding='utf-8',dir=path.parent,prefix='.margam-',delete=False) as stream:
            temporary=stream.name;writer=csv.writer(stream)
            supplied='evaluation' in data
            writer.writerow((['source_row'] if supplied else [])+(['point_id'] if ids is not None else [])+list(coordinates)+fields+(['ref_'+f for f in fields]+['error_'+f for f in fields] if refs is not None else []))
            for i,(point,value) in enumerate(zip(points,array)):
                writer.writerow(([i] if supplied else [])+([ids[i]] if ids is not None else [])+list(point)+value.tolist()+(refs[i].tolist()+(value-refs[i]).tolist() if refs is not None else []))
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None and os.path.exists(temporary):os.unlink(temporary)
