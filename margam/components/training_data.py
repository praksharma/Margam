"""Prepared planar training sets; labels carry no implied box normal or topology."""
import re
import numpy as np
from .prepared_data import read_points


def valid_boundary_name(name):
    return isinstance(name,str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}',name) is not None and name not in ('pde','total')


def read_training_sets(config,*,check_physics=True):
    """Read and validate each source once, before constructing its constraints."""
    dimension=len(config.bounds[0]);lower,upper=map(np.asarray,config.bounds)
    def read(source,interior=False):
        rows=read_points(source,dimension)
        if interior and (rows[1] is not None or rows[2] is not None):raise ValueError('Interior data accepts coordinates only.')
        tolerance=np.maximum((upper-lower)*1e-7,1e-12)
        if ((rows[0]<lower-tolerance)|(rows[0]>upper+tolerance)).any():raise ValueError('Prepared training points lie outside the coordinate bounds.')
        return rows
    interior=read(config.interior_data,True)
    boundaries={name:read(source) for name,source in config.boundary_data.items()}
    if sum(len(value[0]) for value in boundaries.values())>100000:raise ValueError('Prepared boundaries support at most 100000 points in total.')
    if not check_physics:return interior,boundaries
    from .expressions import Expression
    for name,(_,normals,values) in boundaries.items():
        target=config.bc_targets.get(name,'');operator=config.bc_operators.get(name,'')
        needs=config.bc_types[name]=='Neumann' or config.bc_types[name]=='Custom' and Expression(operator,dimension).uses_normal or bool(target) and Expression(target,dimension,target=True).uses_normal
        if needs and normals is None:raise ValueError(name+': map explicit outward unit normals for this boundary operator/target.')
        if values is not None and target:raise ValueError(name+': choose imported target values or a target expression, not both.')
    return interior,boundaries
