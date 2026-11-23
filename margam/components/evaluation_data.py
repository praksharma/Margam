"""Immutable snapshots of supplied evaluation coordinates and reference fields."""
from dataclasses import dataclass
import csv,hashlib,io,json
from pathlib import Path
import zipfile
import numpy as np
from .prepared_data import LIMIT, cluster_1d, match_coordinates, real_array


def _time_tolerance(times):
    times = np.asarray(times, dtype=float)
    if len(times) == 0:
        return 1e-12
    return max(
        1e-12,
        float(np.ptp(times)) * 1e-9,
        float(np.finfo(float).eps * max(1.0, float(np.abs(times).max())) * 8),
    )


@dataclass(frozen=True)
class PointData:
    points: list
    point_ids: list | None
    values: list | None
    provenance: dict


def validate_source(source,*,reference=False):
    if not isinstance(source,dict) or set(source)!={'path','coordinates','values','ids'} or not all(isinstance(v,str) and len(v)<=4096 for v in source.values()):raise ValueError('Point source needs path, coordinates, values and optional IDs mappings.')
    if not source['path'] or not source['coordinates'] or Path(source['path']).suffix.lower() not in ('.csv','.npz'):raise ValueError('Choose a CSV/NPZ file and explicit coordinate mapping.')
    if reference and not source['values']:raise ValueError('Map reference values in the model field order.')
    if not reference and source['values']:raise ValueError('Evaluation points supply coordinates/IDs, not reference fields.')


def validate_ids(ids,points,transient=False):
    if ids is None:return
    if not isinstance(ids,list) or len(ids)!=len(points) or any(type(v) not in (str,int) or isinstance(v,str) and (not v.strip() or len(v)>256 or '\x00' in v) for v in ids):raise ValueError('IDs must be nonempty strings (up to 256 characters) or integers, one per row.')
    if transient and len(points):
        times=np.asarray([point[-1] for point in points],dtype=float)
        frames=cluster_1d(times,_time_tolerance(times)).tolist()
        keys=[(str(v),frame) for v,frame in zip(ids,frames)]
    else:
        keys=[(str(v),None) for v in ids]
    if len(set(keys))!=len(keys):raise ValueError('Point IDs must be unique within each physical time frame.')


def validate_points(points,config,ids=None):
    a=real_array(points);lower,upper=config.bounds
    if a.ndim!=2 or a.shape[1]!=len(lower) or not 1<=len(a)<=100000 or not np.isfinite(a).all():raise ValueError('Evaluation coordinates must be finite (N,dimension), with 1–100000 rows.')
    tolerance=np.maximum((np.asarray(upper)-lower)*1e-7,np.finfo(float).eps*np.maximum(1,np.abs([lower,upper]).max(axis=0))*8)
    if ((a<np.asarray(lower)-tolerance)|(a>np.asarray(upper)+tolerance)).any():raise ValueError('Evaluation points lie outside the configured domain; extrapolation is not implicit.')
    from .geometry import fluid_mask, obstacle_spec
    obs=obstacle_spec(config)
    if obs is not None:
        obs_dim=3 if obs['kind']=='sphere' else 2
        if not fluid_mask(a,obs,strict_interior=False,tolerance=float(max(tolerance[:obs_dim]))).all():
            raise ValueError('Evaluation points lie inside the solid '+obs['kind']+'.')
    validate_ids(ids,a.tolist(),config.is_time_dependent)
    return a


def read_point_data(source,config,*,reference=False):
    validate_source(source,reference=reference);path=Path(source['path'])
    if path.stat().st_size>LIMIT:raise ValueError('Point data exceeds 64 MiB.')
    with path.open('rb') as stream:raw=stream.read(LIMIT+1)
    if len(raw)>LIMIT:raise ValueError('Point data exceeds 64 MiB.')
    dimension=len(config.bounds[0]);components=len(config.field_names);ids=None;values=None
    id_col=source['ids'].strip()
    if path.suffix.lower()=='.csv':
        reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig'),newline=''))
        coords=[s.strip() for s in source['coordinates'].split(',')];fields=[s.strip() for s in source['values'].split(',')] if reference else []
        if len(coords)!=dimension or fields and len(fields)!=components:raise ValueError('Map one column per coordinate and one reference column per model field.')
        requested=coords+fields+([id_col] if id_col else [])
        if not reader.fieldnames or len(reader.fieldnames)!=len(set(reader.fieldnames)) or not set(requested)<=set(reader.fieldnames):raise ValueError('A mapped CSV column is missing or duplicated.')
        points=[];rows=[];ids=[] if id_col else None
        try:
            for row in reader:
                if len(points)>=100000:raise ValueError('Point data supports at most 100000 rows.')
                points.append([float(row[k]) for k in coords])
                if fields:rows.append([float(row[k]) for k in fields])
                if ids is not None:ids.append(row[id_col])
        except (TypeError,KeyError) as error:raise ValueError('Invalid mapped CSV row.') from error
        if reference:values=rows
    else:
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                if sum(v.file_size for v in archive.infolist())>LIMIT:raise ValueError('Expanded NPZ data exceeds 64 MiB.')
        except zipfile.BadZipFile as error:raise ValueError('Invalid NPZ archive.') from error
        with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
            try:
                points=real_array(archive[source['coordinates'].strip()]).tolist()
                if reference:
                    values=real_array(archive[source['values'].strip()]);values=values[:,None] if values.ndim==1 else values;values=values.tolist()
                if id_col:
                    array=archive[id_col]
                    if array.ndim==2 and array.shape[1]==1:array=array[:,0]
                    if array.ndim!=1 or array.dtype.kind not in 'iuUS':raise ValueError('NPZ IDs need a one-dimensional integer or Unicode/string array; object arrays are not supported.')
                    ids=[v.decode('utf-8') if isinstance(v,bytes) else v for v in array.tolist()]
            except KeyError as error:raise ValueError('A mapped NPZ array is missing.') from error
    points=validate_points(points,config,ids).tolist()
    if reference:
        a=real_array(values)
        if a.shape!=(len(points),components) or not np.isfinite(a).all():raise ValueError('Reference fields must be finite and match coordinate rows and model fields.')
        values=a.tolist()
    return PointData(points,ids,values,dict(kind='supplied',source_name=path.name,sha256=hashlib.sha256(raw).hexdigest()))


def align_reference(reference,points,point_ids,config):
    a=validate_points(reference.points,config,reference.point_ids);b=validate_points(points,config,point_ids)
    values=real_array(reference.values)
    if len(a)!=len(b) or values.shape!=(len(a),len(config.field_names)) or not np.isfinite(values).all():raise ValueError('Reference data needs one finite value per requested point and field.')
    tolerance=np.maximum(np.ptp(b,axis=0)*1e-9,np.finfo(float).eps*np.maximum(1,np.abs(b).max(axis=0))*8)
    if point_ids is not None and reference.point_ids is not None:
        if config.is_time_dependent:
            combined_t=np.concatenate((a[:,-1],b[:,-1]))
            t_ranks=cluster_1d(combined_t,float(tolerance[-1]))
            t_a,t_b=t_ranks[:len(a)].tolist(),t_ranks[len(a):].tolist()
        else:
            t_a=t_b=[0]*len(a)
        source=sorted(range(len(a)),key=lambda i:(str(reference.point_ids[i]),t_a[i]))
        target=sorted(range(len(b)),key=lambda i:(str(point_ids[i]),t_b[i]))
        if any(str(reference.point_ids[i])!=str(point_ids[j]) for i,j in zip(source,target)):raise ValueError('Reference IDs do not match the evaluation IDs.')
        if not np.all(np.abs(a[source]-b[target])<=tolerance):raise ValueError('Reference coordinates do not match evaluation points; interpolation is not implicit.')
    else:
        dup_msg='Repeated coordinates need IDs in both evaluation and reference sources for unambiguous matching.'
        mismatch_msg='Reference coordinates do not match evaluation points; interpolation is not implicit.'
        if len(np.unique(a,axis=0))!=len(a) or len(np.unique(b,axis=0))!=len(b):raise ValueError(dup_msg)
        source,target=match_coordinates(a,b,tolerance,mismatch_message=mismatch_msg,duplicate_message=dup_msg)
    result=np.empty_like(values);result[target]=values[source]
    return result
