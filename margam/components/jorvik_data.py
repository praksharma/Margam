"""Read Jorvik schema-v2 NetCDF snapshots; never infer geometry or normals."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .prepared_data import LIMIT, real_array


def open_export(path):
    import xarray as xr
    path = Path(path)
    if path.stat().st_size > LIMIT:
        raise ValueError('Jorvik export exceeds 64 MiB.')
    return xr.open_dataset(path, engine='scipy', decode_cf=False)


def records(path):
    with open_export(path) as dataset:
        if dataset.nbytes > LIMIT:
            raise ValueError('Expanded Jorvik export exceeds 64 MiB.')
        try:
            manifest = json.loads(dataset.attrs['metadata_json'])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError('Jorvik export metadata is missing or invalid.') from error
        if not isinstance(manifest,dict) or manifest.get('schema_version') != 2:
            raise ValueError('Only Jorvik sample schema version 2 is supported.')
        result = manifest.get('datasets')
        if not isinstance(result, list) or not 1 <= len(result) <= 65:
            raise ValueError('Choose an export with 1–65 labelled datasets.')
        labels = set()
        for record in result:
            if not isinstance(record, dict) or not isinstance(record.get('label'), str) or not record['label'] or record['label'] in labels:
                raise ValueError('Dataset labels must be non-empty and unique.')
            labels.add(record['label'])
            if not isinstance(record.get('array'),str) or record['array'] not in dataset or not isinstance(record.get('metadata'),dict):
                raise ValueError('Dataset array or metadata is missing.')
            region=record['metadata'].get('boundary_region',{})
            if not isinstance(region,dict) or 'id' in region and (not isinstance(region['id'],str) or not 1<=len(region['id'])<=512):
                raise ValueError('Invalid boundary region identity.')
            if record.get('normals') and not isinstance(record['normals'],str):raise ValueError('Invalid normals variable name.')
            if record['metadata'].get('normal_status')=='available' and not record.get('normals'):
                raise ValueError('Available boundary normals are missing their numeric array.')
        return result


def read_array(path, specification):
    """Mapping syntax is an explicit NetCDF variable followed by two source axes."""
    try:
        name, axes_text = specification.rsplit('|', 1)
        axes = tuple(int(part) for part in axes_text.split(','))
    except (ValueError, AttributeError) as error:
        raise ValueError('NetCDF mappings need variable|axis,axis.') from error
    if len(axes) != 2 or len(set(axes)) != 2 or not set(axes) <= {0,1,2}:
        raise ValueError('Choose two distinct source axes.')
    with open_export(path) as dataset:
        if name not in dataset:
            raise ValueError('A mapped NetCDF variable is missing.')
        variable = dataset[name]
        if variable.ndim != 2 or variable.shape[1] != 3 or not 1 <= variable.shape[0] <= 100000:
            raise ValueError('Jorvik arrays must have shape (N,3), at most 100000 rows.')
        array = real_array(variable.values)
    if not np.isfinite(array).all():
        raise ValueError('Jorvik arrays must be finite.')
    return array, axes


def inspect_selection(path, selections, axes=(0,1)):
    """Validate roles, planar reduction, verified normals and explicit boundary names."""
    from .training_data import valid_boundary_name
    exported = records(path)
    known = {record['label']: record for record in exported}
    if not isinstance(selections,dict) or not set(selections) <= set(known):
        raise ValueError('A selected dataset is missing.')
    interior = []; boundaries = {}; identity = set(); source_identity = None; plane = None; all_points = []
    suffix = '|' + ','.join(map(str, axes))
    for label, choice in selections.items():
        role, alias = choice
        if role == 'ignore': continue
        if role not in ('interior','boundary'):
            raise ValueError('Choose Interior, Boundary or Ignore for each dataset.')
        record = known[label]
        metadata = record['metadata']
        current_identity=(metadata.get('document'),metadata.get('source_hash'))
        if not all(current_identity) or source_identity is not None and current_identity!=source_identity:
            raise ValueError('Selected datasets must belong to the same source geometry.')
        source_identity=current_identity
        if metadata.get('entity_dimension') != (2 if role=='interior' else 1):
            raise ValueError('Planar 2D needs surface interiors and edge boundaries.')
        key = metadata.get('boundary_region',{}).get('id') or 'label:' + label
        if key in identity: raise ValueError('Choose only one sample dataset per region identity.')
        identity.add(key)
        points, checked_axes = read_array(path, record['array'] + suffix)
        dropped = next(axis for axis in range(3) if axis not in checked_axes)
        if plane is None: plane = float(points[0,dropped])
        if not np.allclose(points[:,dropped],plane,rtol=0,atol=max(float(np.ptp(points,axis=0).max()),1.)*1e-9):
            raise ValueError('Selected samples must lie on one axis-aligned plane. No 3D reduction is inferred.')
        mapping = dict(path=str(path),coordinates=record['array']+suffix,normals='',values='')
        if role == 'interior':
            interior.append(mapping)
        else:
            if not valid_boundary_name(alias) or alias in boundaries:
                raise ValueError('Boundary names must be unique: start with a letter, then letters, digits, _ or - (64 characters maximum).')
            if record.get('normals'):
                if metadata.get('normal_status') != 'available' or metadata.get('normal_kind') != 'outward_conormal':
                    raise ValueError('Planar boundaries require verified outward conormals when normals are supplied.')
                normals, _ = read_array(path, record['normals']+suffix)
                if normals.shape != points.shape or not np.allclose(normals[:,dropped],0,atol=1e-8) or not np.allclose(np.linalg.norm(normals,axis=1),1,atol=1e-5):
                    raise ValueError('Boundary normals must be unit vectors in the selected plane.')
                mapping['normals'] = record['normals'] + suffix
            boundaries[alias] = dict(mapping=mapping,identity=key,label=label)
        all_points.append(points[:,checked_axes])
    if len(interior) != 1 or not boundaries or len(boundaries)>64:
        raise ValueError('Select exactly one interior dataset and 1–64 named boundaries.')
    if sum(len(read_array(path, source['mapping']['coordinates'])[0]) for source in boundaries.values()) > 100000:
        raise ValueError('Boundary datasets exceed 100000 points in total.')
    points = np.concatenate(all_points)
    lower, upper = points.min(axis=0), points.max(axis=0)
    if np.any(lower>=upper): raise ValueError('Selected points need nonzero extent in both problem coordinates.')
    return dict(interior=interior[0],boundaries=boundaries,bounds=[lower.tolist(),upper.tolist()],
                axes=list(axes),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
