"""Restricted tensor/primitive job snapshots and explicit reference declarations."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import os
import zipfile

LIMIT=256*1024*1024


def cpu_tree(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    if type(value) in (str,int,float,bool,type(None)):return value
    raise ValueError('Jobs transport tensors and primitives only, never callbacks or Python objects.')


class LimitedWriter:
    def __init__(self,stream):self.stream=stream;self.count=0;self.exceeded=False
    def write(self,data):
        if self.count+len(data)>LIMIT:
            self.exceeded=True;raise ValueError('Job snapshot exceeds the 256 MiB transfer limit.')
        self.count+=len(data);return self.stream.write(data)
    def flush(self):return self.stream.flush()


def save_packet(path,data):
    import torch
    path=Path(path);temporary=path.with_suffix('.partial')
    try:
        with temporary.open('wb') as stream:
            writer=LimitedWriter(stream)
            try:torch.save(cpu_tree(data),writer)
            except Exception:
                if writer.exceeded:raise ValueError('Job snapshot exceeds the 256 MiB transfer limit; no result was adopted.') from None
                raise
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def load_packet(path):
    import torch
    path=Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size>LIMIT:raise ValueError('Missing or oversized job snapshot.')
    with zipfile.ZipFile(path) as archive:
        files=archive.infolist()
        if len(files)>65536 or sum(item.file_size for item in files)>LIMIT:raise ValueError('Expanded job snapshot exceeds transfer bounds.')
    return torch.load(path,map_location='cpu',weights_only=True)


def encode_config(config):return asdict(config)


def decode_config(data):
    from ..templates.poisson import PoissonConfig,BoundarySampling
    data=deepcopy(data)
    data['boundary_sampling']={name:BoundarySampling(**value) for name,value in data.get('boundary_sampling',{}).items()}
    config=PoissonConfig(**data);config.validate();return config


def encode_request(request):
    if request.reference is not None and request.reference.spec is None:
        raise ValueError('Isolated desktop jobs require a data-only reference declaration; arbitrary Python callbacks belong to the research API.')
    return dict(identifier=request.identifier,kind=request.kind,resolution=request.resolution,
        reference=deepcopy(request.reference.spec) if request.reference else None,
        point_data=asdict(request.point_data) if request.point_data else None,
        reference_data=asdict(request.reference_data) if request.reference_data else None)


def decode_request(data,config):
    from ..components.outputs import OutputRequest,ReferenceSource
    from ..components.evaluation_data import PointData
    from ..components.expressions import Expression
    from ..components.prepared_data import reference_values
    from ..templates.poisson import make_poisson_reference
    from functools import partial
    reference=None;spec=data['reference']
    if spec is not None:
        kind=spec['kind']
        if kind=='preset':reference=make_poisson_reference(config)
        elif kind=='expression':
            expressions=[Expression(part,len(config.bounds[0]),target=True,fields=config.field_names,coordinates=config.coordinate_names) for part in spec['texts']]
            def evaluate(points):
                import torch
                values=torch.cat([expr(torch.tensor(points),source=config.source,viscosity=config.viscosity) for expr in expressions],dim=1)
                return values[:,0].tolist() if len(expressions)==1 else values.tolist()
            reference=ReferenceSource('User reference expression',evaluate)
        elif kind=='mapping':reference=ReferenceSource('Imported reference fields',partial(reference_values,spec['source'],components=len(config.field_names)))
        else:raise ValueError('Unknown reference declaration.')
    return OutputRequest(data['identifier'],data['kind'],data['resolution'],reference,
        PointData(**data['point_data']) if data['point_data'] is not None else None,
        PointData(**data['reference_data']) if data['reference_data'] is not None else None)
