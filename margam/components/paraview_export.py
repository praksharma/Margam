"""Dependency-light VTK ImageData and timestamped ParaView collections.

No solver time is inferred from training updates. See docs/paraview-export.md.
"""
import math
import os
from numbers import Real
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4
import xml.etree.ElementTree as ET
import numpy as np


def _grid(data):
    try:
        points=np.asarray(data['points'],dtype=np.float64)
        values=np.asarray(data['values'],dtype=np.float64)
    except (KeyError,TypeError,ValueError) as error:raise ValueError('Export needs numeric prediction points and values.') from error
    if points.ndim!=2 or points.shape[1] not in (1,2) or not len(points) or not np.isfinite(points).all():
        raise ValueError('VTI export requires finite 1D or 2D coordinates.')
    fields=data.get('fields',['u'])
    if not isinstance(fields,(list,tuple)) or not fields or any(not isinstance(f,str) or not f.strip() or f=='velocity' for f in fields) or len(set(fields))!=len(fields):
        raise ValueError('Export fields must be distinct names; velocity is reserved for the derived flow vector.')
    if values.ndim==1:values=values[:,None]
    if values.shape!=(len(points),len(fields)) or not np.isfinite(values).all():raise ValueError('Prediction values must be finite and match points and fields.')
    axes=[np.unique(points[:,i]) for i in range(points.shape[1])]
    if any(len(a)<2 for a in axes) or math.prod(len(a) for a in axes)!=len(points):raise ValueError('VTI needs a complete non-degenerate regular line or rectangle grid.')
    spacing=[float((a[-1]-a[0])/(len(a)-1)) for a in axes]
    ideal=[np.linspace(a[0],a[-1],len(a)) for a in axes]
    if not all(math.isfinite(v) and v>0 for v in spacing):raise ValueError('Grid spacing must be positive and finite.')
    tolerance=max(1e-12,1e-7*max(float(a[-1]-a[0]) for a in axes),8*np.finfo(float).eps*float(np.abs(points).max()))
    if any(not np.allclose(a,b,rtol=0,atol=tolerance) for a,b in zip(axes,ideal)):
        raise ValueError('VTI requires uniform spacing; use a different format for irregular samples.')
    expected=np.array([[x] for x in ideal[0]] if len(axes)==1 else [[x,y] for y in ideal[1] for x in ideal[0]])
    if not np.allclose(points,expected,rtol=0,atol=tolerance):raise ValueError('VTI coordinates must be complete and ordered with x varying fastest.')
    shape=[len(a) for a in axes]+[1]*(3-len(axes))
    origin=[float(a[0]) for a in axes]+[0.]*(3-len(axes))
    return shape,origin,spacing+[1.]*(3-len(axes)),list(fields),values


def _time(value):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not math.isfinite(value):raise ValueError('Physical times must be finite numbers.')
    return float(value)


def _array(parent,name,values,components=1):
    array=ET.SubElement(parent,'DataArray',type='Float64',Name=name,NumberOfComponents=str(components),format='ascii')
    array.text=' '.join(format(float(v),'.17g') for v in np.asarray(values).reshape(-1))
    return array


def _document(grid,time=None):
    shape,origin,spacing,fields,values=grid
    extent=' '.join(str(v) for n in shape for v in (0,n-1))
    root=ET.Element('VTKFile',type='ImageData',version='1.0',byte_order='LittleEndian')
    image=ET.SubElement(root,'ImageData',WholeExtent=extent,Origin=' '.join(map(str,origin)),Spacing=' '.join(map(str,spacing)))
    if time is not None:
        field=ET.SubElement(image,'FieldData')
        array=_array(field,'TimeValue',[_time(time)]);array.set('NumberOfTuples','1')
    piece=ET.SubElement(image,'Piece',Extent=extent)
    attributes={'Scalars':fields[0]}
    flow=fields==['u','v','p']
    if flow:attributes['Vectors']='velocity'
    point_data=ET.SubElement(piece,'PointData',**attributes)
    for index,name in enumerate(fields):_array(point_data,name,values[:,index])
    if flow:_array(point_data,'velocity',np.column_stack((values[:,0],values[:,1],np.zeros(len(values)))),3)
    ET.SubElement(piece,'CellData')
    return ET.ElementTree(root)


def _write(path,tree):
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.margam-',delete=False) as stream:
            temporary=stream.name;tree.write(stream,encoding='utf-8',xml_declaration=True)
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None and os.path.exists(temporary):os.unlink(temporary)


def export_vti(path,data,*,time=None):
    """Export explicit regular-grid predictions; optional time is physical time."""
    path=Path(path)
    if path.suffix.lower()!='.vti':raise ValueError('A VTI snapshot needs a .vti filename.')
    tree=_document(_grid(data),time)
    _write(path,tree)
    return path


def export_pvd(path,frames):
    """Export (physical_time, prediction_data) frames as one playable collection.

    Strictly increasing times, same grid/field schema. Stage new companions and
    publish the index last: a failed replacement leaves the prior series intact.
    Keep the PVD and its companion directory together when moving the result.
    """
    path=Path(path)
    if path.suffix.lower()!='.pvd':raise ValueError('A time series needs a .pvd filename.')
    stage=Path(tempfile.mkdtemp(dir=path.parent,prefix='.margam-series-'))
    destination=path.parent/(path.stem+'_frames_'+uuid4().hex)
    published=False
    try:
        root=ET.Element('VTKFile',type='Collection',version='0.1',byte_order='LittleEndian')
        collection=ET.SubElement(root,'Collection');previous=None;schema=None;count=0
        for index,(time,data) in enumerate(frames):
            time=_time(time)
            if previous is not None and time<=previous:raise ValueError('Physical times must be strictly increasing; duplicate times are not frames.')
            grid=_grid(data)
            if schema is None:schema=grid[:4]
            elif grid[:4]!=schema:raise ValueError('All time frames must use the same regular grid and fields.')
            name=f'frame_{index:06d}.vti';_write(stage/name,_document(grid,time))
            ET.SubElement(collection,'DataSet',timestep=format(time,'.17g'),group='',part='0',file=destination.name+'/'+name)
            previous=time;count+=1
        if not count:raise ValueError('A time series needs at least one physical-time frame.')
        stage.rename(destination)
        _write(path,ET.ElementTree(root));published=True
        return path
    finally:
        if stage.exists():shutil.rmtree(stage)
        if not published and destination.exists():shutil.rmtree(destination)
