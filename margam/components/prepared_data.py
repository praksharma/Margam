"""Explicit CSV/NPZ coordinate, normal and scalar-value mappings; no inference."""
import csv
from pathlib import Path
import zipfile
import numpy as np

LIMIT=64*1024*1024


def validate_mapping(source):
    if not isinstance(source,dict) or set(source)!={'path','coordinates','normals','values'}:
        raise ValueError('Data mapping needs path, coordinates, normals and values.')
    if not all(isinstance(value,str) and len(value)<=4096 for value in source.values()) or not source['path'] or not source['coordinates']:
        raise ValueError('Choose a data path and an explicit coordinate mapping.')
    if Path(source['path']).suffix.lower() not in ('.csv','.npz'):raise ValueError('Prepared data supports CSV or NPZ.')


def read_points(source,dimension):
    validate_mapping(source)
    path=Path(source['path'])
    if path.stat().st_size>LIMIT:raise ValueError('Prepared data exceeds 64 MiB.')
    if path.suffix.lower()=='.csv':
        coordinates=[value.strip() for value in source['coordinates'].split(',')]
        normals=[value.strip() for value in source['normals'].split(',')] if source['normals'] else []
        if len(coordinates)!=dimension or normals and len(normals)!=dimension:raise ValueError('CSV mappings need one column per coordinate dimension.')
        points=[];normal_rows=[];values=[]
        with path.open(newline='',encoding='utf-8-sig') as stream:
            reader=csv.DictReader(stream)
            requested=coordinates+normals+([source['values']] if source['values'] else [])
            if not reader.fieldnames or len(reader.fieldnames)!=len(set(reader.fieldnames)) or not set(requested)<=set(reader.fieldnames):raise ValueError('A mapped CSV column is missing.')
            try:
                for row in reader:
                    if len(points)>=100000:raise ValueError('Prepared data supports at most 100000 points.')
                    points.append([float(row[key]) for key in coordinates])
                    if normals:normal_rows.append([float(row[key]) for key in normals])
                    if source['values']:values.append(float(row[source['values']]))
            except (TypeError,KeyError) as error:raise ValueError('Invalid numeric CSV row.') from error
        points=np.asarray(points,dtype=float)
        normals=np.asarray(normal_rows,dtype=float) if normal_rows else None
        values=np.asarray(values,dtype=float)[:,None] if values else None
    else:
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist())>LIMIT:raise ValueError('Expanded NPZ data exceeds 64 MiB.')
        with np.load(path,allow_pickle=False) as archive:
            try:
                points=np.asarray(archive[source['coordinates']],dtype=float)
                normals=np.asarray(archive[source['normals']],dtype=float) if source['normals'] else None
                values=np.asarray(archive[source['values']],dtype=float) if source['values'] else None
            except KeyError as error:raise ValueError('A mapped NPZ array is missing.') from error
    if points.ndim!=2 or points.shape[1]!=dimension or not 1<=len(points)<=100000 or not np.isfinite(points).all():raise ValueError('Coordinates must be a finite non-empty (N,dimension) array.')
    if normals is not None:
        if normals.shape!=points.shape or not np.isfinite(normals).all() or not np.allclose(np.linalg.norm(normals,axis=1),1,atol=1e-5):raise ValueError('Normals must be finite, unit length, and match coordinate rows. Their orientation must be supplied by the user.')
    if values is not None:
        if values.ndim==1:values=values[:,None]
        if values.shape!=(len(points),1) or not np.isfinite(values).all():raise ValueError('Values must be finite scalars matching coordinate rows.')
    return points,normals,values


def reference_values(source,points):
    requested=np.asarray(points,dtype=float)
    coordinates,_,values=read_points(source,requested.shape[1])
    if values is None or len(coordinates)!=len(requested):raise ValueError('Reference data must provide one scalar per requested evaluation coordinate; interpolation is not implicit.')
    source_order=np.lexsort(coordinates.T[::-1]);target_order=np.lexsort(requested.T[::-1])
    tolerance=np.maximum(np.ptp(requested,axis=0)*1e-9,np.finfo(float).eps*np.maximum(1,np.abs(requested).max(axis=0))*8)
    if len(np.unique(coordinates,axis=0))!=len(coordinates) or not np.all(np.abs(coordinates[source_order]-requested[target_order])<=tolerance):raise ValueError('Reference coordinates do not match the evaluation points. Supply matching points or resample explicitly.')
    output=np.empty(len(requested));output[target_order]=values[source_order,0]
    return output.tolist()
