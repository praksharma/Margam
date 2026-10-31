"""Persist accepted file bytes separately from numerical source mappings."""
from copy import deepcopy
from ..components.prepared_data import validate_mapping,file_fingerprint
from ..components.evaluation_data import validate_source
from ..components.training_data import valid_boundary_name


def source_slot(kind,boundary=None):
    if boundary is not None:return 'boundary:'+boundary
    if kind in ('evaluation_points','reference_data'):return 'source_data'
    if kind=='reference':return 'reference_data'
    return 'interior_data'


def current_source(settings,slot):
    if slot.startswith('boundary:'):return settings.get('parameters',{}).get('boundary_data',{}).get(slot[9:],{})
    if slot=='interior_data':return settings.get('parameters',{}).get(slot,{})
    return settings.get(slot,{})


def validate_fingerprints(kind,records):
    if not isinstance(records,dict) or len(records)>64:raise ValueError('Invalid source fingerprints.')
    for slot,record in records.items():
        valid=(slot=='source_data' and kind in ('evaluation_points','reference_data') or
               slot=='reference_data' and kind=='reference' or
               slot=='interior_data' and kind in ('interior','interior_data','rar','importance') or
               isinstance(slot,str) and slot.startswith('boundary:') and kind in ('boundary','boundary_data') and valid_boundary_name(slot[9:]))
        if not valid or not isinstance(record,dict) or set(record)!={'source','sha256'}:raise ValueError('Invalid source fingerprint binding.')
        digest=record['sha256']
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Invalid source SHA-256 fingerprint.')
        if slot=='source_data':validate_source(record['source'],reference=kind=='reference_data')
        else:validate_mapping(record['source'])


def accepted_fingerprint(workspace,source):
    from pathlib import Path
    logical=workspace.portable_document(deepcopy(source))
    resolved=workspace.resolve_source(logical,create=False)
    digest,_=file_fingerprint(Path(resolved['path']))
    return dict(source=logical,sha256=digest)
