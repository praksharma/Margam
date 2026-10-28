"""VTU vertex point clouds for the concrete channel/cylinder examples."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def point_grid(data):
    points=np.asarray(data['points'],dtype=float);values=np.asarray(data['values'],dtype=float)
    fields=list(data.get('fields',['u']))
    if values.ndim==1:values=values[:,None]
    if points.ndim!=2 or points.shape[1]!=2 or not len(points) or not np.isfinite(points).all() or values.shape!=(len(points),len(fields)) or not np.isfinite(values).all():raise ValueError('VTU point export requires finite spatial 2D points and matching fields.')
    if len(set(fields))!=len(fields) or any(not isinstance(f,str) or not f or f=='velocity' for f in fields):raise ValueError('Invalid VTU field names.')
    return points,fields,values


def point_document(grid,time=None):
    from .paraview_export import _array
    points,fields,values=grid;n=len(points)
    root=ET.Element('VTKFile',type='UnstructuredGrid',version='0.1',byte_order='LittleEndian');mesh=ET.SubElement(root,'UnstructuredGrid')
    if time is not None:
        field=ET.SubElement(mesh,'FieldData');array=_array(field,'TimeValue',[time]);array.set('NumberOfTuples','1')
    piece=ET.SubElement(mesh,'Piece',NumberOfPoints=str(n),NumberOfCells=str(n))
    point_data=ET.SubElement(piece,'PointData',Scalars=fields[0])
    for i,name in enumerate(fields):_array(point_data,name,values[:,i])
    if fields in (['u','v'],['u','v','p'],['u','v','w']):
        point_data.set('Vectors','velocity');_array(point_data,'velocity',np.column_stack((values[:,:2],np.zeros(n))),3)
    spatial=ET.SubElement(piece,'Points');_array(spatial,'Points',np.column_stack((points,np.zeros(n))),3)
    cells=ET.SubElement(piece,'Cells')
    for name,rows in (('connectivity',range(n)),('offsets',range(1,n+1)),('types',[1]*n)):
        array=ET.SubElement(cells,'DataArray',type='Int64' if name!='types' else 'UInt8',Name=name,format='ascii');array.text=' '.join(map(str,rows))
    ET.SubElement(piece,'CellData')
    return ET.ElementTree(root)


def export_vtu(path,data):
    from .paraview_export import _write
    path=Path(path)
    if path.suffix.lower()!='.vtu':raise ValueError('A point cloud needs a .vtu filename.')
    if getattr(data.get('config'),'is_time_dependent',False):raise ValueError('Use a PVD collection for space–time predictions.')
    _write(path,point_document(point_grid(data)));return path
