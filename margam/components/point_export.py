"""VTU vertex point clouds for the concrete channel/cylinder examples."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def point_grid(data):
    points=np.asarray(data['points'],dtype=float);values=np.asarray(data['values'],dtype=float)
    fields=list(data.get('fields',['u']))
    if values.ndim==1:values=values[:,None]
    if points.ndim!=2 or points.shape[1] not in (1,2) or not len(points) or not np.isfinite(points).all() or values.shape!=(len(points),len(fields)) or not np.isfinite(values).all():raise ValueError('VTU point export requires finite spatial 1D/2D points and matching fields.')
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
    spatial=ET.SubElement(piece,'Points');_array(spatial,'Points',np.column_stack((points,np.zeros((n,3-points.shape[1])))),3)
    cells=ET.SubElement(piece,'Cells')
    for name,rows in (('connectivity',range(n)),('offsets',range(1,n+1)),('types',[1]*n)):
        array=ET.SubElement(cells,'DataArray',type='Int64' if name!='types' else 'UInt8',Name=name,format='ascii');array.text=' '.join(map(str,rows))
    ET.SubElement(piece,'CellData')
    return ET.ElementTree(root)


def export_vtu(path,data):
    from .paraview_export import _write,_reference_arrays
    path=Path(path)
    if path.suffix.lower()!='.vtu':raise ValueError('A point cloud needs a .vtu filename.')
    if getattr(data.get('config'),'is_time_dependent',False):raise ValueError('Use a PVD collection for space–time predictions.')
    _write(path,_reference_arrays(point_document(point_grid(data)),data));return path


def supplied_document(data,time=None):
    """Point-cloud fields, source row identity and lossless integer/text IDs."""
    import json
    from .paraview_export import _array
    grid=point_grid(data);tree=point_document(grid,time);points,fields,values=grid
    mesh=tree.getroot().find('UnstructuredGrid');pd=mesh.find('Piece/PointData')
    def integer_array(parent,name,rows,kind='Int64'):
        a=ET.SubElement(parent,'DataArray',type=kind,Name=name,NumberOfComponents='1',format='ascii');a.text=' '.join(str(v) for v in rows);return a
    rows=data.get('source_rows',list(range(len(points))))
    if len(rows)!=len(points):raise ValueError('Source row metadata does not match points.')
    integer_array(pd,'source_row',rows)
    ids=data.get('point_ids')
    if ids is not None:
        from .evaluation_data import validate_ids
        validate_ids(ids,points.tolist())
        fd=mesh.find('FieldData')
        if fd is None:fd=ET.SubElement(mesh,'FieldData')
        encoded=json.dumps(ids,ensure_ascii=False,separators=(',',':')).encode('utf-8')
        a=integer_array(fd,'point_ids_json_utf8',encoded,'UInt8');a.set('NumberOfTuples',str(len(encoded)))
        try:
            numbers=[int(v) for v in ids]
            numeric=all(str(a)==str(b) and -2**63<=a<2**63 for a,b in zip(numbers,ids))
        except (ValueError,TypeError):numeric=False
        if numeric:integer_array(pd,'point_id',numbers)
    from .paraview_export import _reference_arrays
    _reference_arrays(tree,data)
    return tree


def export_supplied(path,data):
    """Supplied snapshots use VTU vertices; transient snapshots form physical-time PVD."""
    from .paraview_export import _write,export_pvd
    config=data['config'];path=Path(path)
    if not config.is_time_dependent:
        if path.suffix.lower()!='.vtu':raise ValueError('Supplied spatial points need a .vtu filename.')
        _write(path,supplied_document(data));return path
    points=np.asarray(data['points'],dtype=float)
    frames=[]
    for time in np.unique(points[:,-1]):
        indices=np.flatnonzero(points[:,-1]==time).tolist()
        frame=dict(points=points[indices,:-1].tolist(),values=[data['values'][i] for i in indices],fields=data.get('fields',['u']),source_rows=indices)
        for key in ('point_ids','reference'):
            if data.get(key) is not None:frame[key]=[data[key][i] for i in indices]
        frames.append((float(time),frame))
    return export_pvd(path,frames,unstructured=True,supplied=True)
