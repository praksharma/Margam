"""Explicit axis-aligned planar adapter for GeoSampler schema-v2 NPZ exports.

No Gmsh/CAD runtime, automatic projection, normal orientation inference or viewport.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np
from .prepared_data import LIMIT,real_array


def prepare_planar_export(source,destination,axes=(0,1),labels=None):
    source=Path(source);destination=Path(destination)
    if destination.exists():raise FileExistsError('Choose a new output directory.')
    if len(axes)!=2 or len(set(axes))!=2 or not set(axes)<={0,1,2}:raise ValueError('Choose two distinct source axes explicitly.')
    if source.stat().st_size>LIMIT:raise ValueError('GeoSampler export exceeds 64 MiB.')
    with zipfile.ZipFile(source) as archive:
        if sum(entry.file_size for entry in archive.infolist())>LIMIT:raise ValueError('Expanded GeoSampler export exceeds 64 MiB.')
    dropped=next(axis for axis in range(3) if axis not in axes);datasets=[];plane=None
    with np.load(source,allow_pickle=False) as archive:
        if '_gs_metadata' not in archive:raise ValueError('GeoSampler schema metadata is missing.')
        manifest=json.loads(str(archive['_gs_metadata'].item()))
        if manifest.get('schema_version')!=2 or not isinstance(manifest.get('datasets'),list):raise ValueError('Only GeoSampler schema version 2 is supported.')
        known=[record['label'] for record in manifest['datasets']]
        if len(known)!=len(set(known)):raise ValueError('Duplicate GeoSampler dataset labels.')
        if labels is not None and not set(labels)<=set(known):raise ValueError('A selected dataset label is missing.')
        for record in manifest['datasets']:
            if labels is not None and record['label'] not in labels:continue
            points=real_array(archive[record['array']])
            if points.ndim!=2 or points.shape[1]!=3 or not 1<=len(points)<=100000 or not np.isfinite(points).all():raise ValueError('GeoSampler coordinates must be finite (N,3) arrays.')
            scale=max(float(np.ptp(points,axis=0).max()),1.)
            if plane is None:plane=float(points[0,dropped])
            if not np.allclose(points[:,dropped],plane,rtol=0,atol=scale*1e-9):raise ValueError('Selected data is not on one axis-aligned plane; provide an explicit problem-specific reduction.')
            arrays={'points':points[:,axes]}
            if record.get('normals'):
                normals=real_array(archive[record['normals']])
                if normals.shape!=points.shape or not np.isfinite(normals).all() or not np.allclose(np.linalg.norm(normals,axis=1),1,atol=1e-5) or not np.allclose(normals[:,dropped],0,atol=1e-8):raise ValueError('Selected normals are not unit vectors within the chosen plane. No orientation is inferred.')
                arrays['normals']=normals[:,axes]
            if record.get('entity_tags'):
                tags=np.asarray(archive[record['entity_tags']])
                if tags.shape!=(len(points),) or not np.issubdtype(tags.dtype,np.integer):raise ValueError('Invalid entity tag rows.')
                arrays['entity_tags']=tags
            datasets.append((record['label'],arrays))
    if not datasets:raise ValueError('Select at least one exported dataset.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    report={'source':str(source.absolute()),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'schema_version':2,'source_axes':list(axes),'dropped_axis':dropped,'plane_coordinate':plane,
            'coordinate_units':manifest.get('coordinate_units','unspecified'),'datasets':[]}
    with tempfile.TemporaryDirectory(dir=destination.parent,prefix='.margam-') as directory:
        staging=Path(directory)/'prepared';staging.mkdir()
        for index,(label,arrays) in enumerate(datasets):
            name=f'dataset-{index:03d}.npz';np.savez_compressed(staging/name,**arrays)
            report['datasets'].append({'label':label,'file':name,'coordinates':'points','normals':'normals' if 'normals' in arrays else '','values':''})
        (staging/'mapping.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        if destination.exists():raise FileExistsError('Output directory appeared during preparation; no files were replaced.')
        os.rename(staging,destination)
    return report
