"""Explicit CSV/NPZ coordinate, normal and scalar-value mappings; no inference."""
import csv
import hashlib
from pathlib import Path
import zipfile
import numpy as np

LIMIT=64*1024*1024


def real_array(value):
    if np.iscomplexobj(value):raise ValueError('Prepared coordinates, normals and fields must be real; complex-valued physics is not supported.')
    return np.asarray(value,dtype=float)


def normal_fingerprint(normals):
    array=np.asarray(normals,dtype=float)
    digest=hashlib.sha256(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def validate_mapping(source):
    if not isinstance(source,dict) or set(source)!={'path','coordinates','normals','values'}:
        raise ValueError('Data mapping needs path, coordinates, normals and values.')
    if not all(isinstance(value,str) and len(value)<=4096 for value in source.values()) or not source['path'] or not source['coordinates']:
        raise ValueError('Choose a data path and an explicit coordinate mapping.')
    if Path(source['path']).suffix.lower() not in ('.csv','.npz','.nc'):raise ValueError('Prepared data supports CSV, NPZ or a mapped Jorvik NetCDF snapshot.')


def read_points(source,dimension,components=1):
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
            value_columns=[key.strip() for key in source['values'].split(',')] if source['values'] else []
            if value_columns and len(value_columns)!=components:raise ValueError('Map one value column per explicit field.')
            requested=coordinates+normals+value_columns
            if not reader.fieldnames or len(reader.fieldnames)!=len(set(reader.fieldnames)) or not set(requested)<=set(reader.fieldnames):raise ValueError('A mapped CSV column is missing.')
            try:
                for row in reader:
                    if len(points)>=100000:raise ValueError('Prepared data supports at most 100000 points.')
                    points.append([float(row[key]) for key in coordinates])
                    if normals:normal_rows.append([float(row[key]) for key in normals])
                    if value_columns:values.append([float(row[key]) for key in value_columns])
            except (TypeError,KeyError) as error:raise ValueError('Invalid numeric CSV row.') from error
        points=np.asarray(points,dtype=float)
        normals=np.asarray(normal_rows,dtype=float) if normal_rows else None
        values=np.asarray(values,dtype=float) if values else None
    elif path.suffix.lower()=='.nc':
        from .jorvik_data import read_array
        raw,axes=read_array(path,source['coordinates']);points=raw[:,axes]
        normals=None
        if source['normals']:
            normal_rows,normal_axes=read_array(path,source['normals'])
            if normal_axes!=axes:raise ValueError('Coordinates and normals must use the same source axes.')
            normals=normal_rows[:,axes]
        if source['values']:raise ValueError('Jorvik supplies geometry samples, not boundary target values.')
        values=None
    else:
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist())>LIMIT:raise ValueError('Expanded NPZ data exceeds 64 MiB.')
        with np.load(path,allow_pickle=False) as archive:
            try:
                points=real_array(archive[source['coordinates']])
                normals=real_array(archive[source['normals']]) if source['normals'] else None
                values=real_array(archive[source['values']]) if source['values'] else None
            except KeyError as error:raise ValueError('A mapped NPZ array is missing.') from error
    if points.ndim!=2 or points.shape[1]!=dimension or not 1<=len(points)<=100000 or not np.isfinite(points).all():raise ValueError('Coordinates must be a finite non-empty (N,dimension) array.')
    if normals is not None:
        if normals.shape!=points.shape or not np.isfinite(normals).all() or not np.allclose(np.linalg.norm(normals,axis=1),1,atol=1e-5):raise ValueError('Normals must be finite, unit length, and match coordinate rows. Their orientation must be supplied by the user.')
    if values is not None:
        if values.ndim==1:values=values[:,None]
        if values.shape!=(len(points),components) or not np.isfinite(values).all():raise ValueError('Values must be finite and match coordinate rows and explicit fields.')
    return points,normals,values


def reference_values(source,points,components=1):
    requested=np.asarray(points,dtype=float)
    coordinates,_,values=read_points(source,requested.shape[1],components)
    if values is None or len(coordinates)!=len(requested):raise ValueError('Reference data must provide the explicit field values for every requested evaluation coordinate; interpolation is not implicit.')
    source_order=np.lexsort(coordinates.T[::-1]);target_order=np.lexsort(requested.T[::-1])
    tolerance=np.maximum(np.ptp(requested,axis=0)*1e-9,np.finfo(float).eps*np.maximum(1,np.abs(requested).max(axis=0))*8)
    if len(np.unique(coordinates,axis=0))!=len(coordinates) or not np.all(np.abs(coordinates[source_order]-requested[target_order])<=tolerance):raise ValueError('Reference coordinates do not match the evaluation points. Supply matching points or resample explicitly.')
    output=np.empty((len(requested),components));output[target_order]=values[source_order]
    return output[:,0].tolist() if components==1 else output.tolist()
