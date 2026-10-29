"""Explicit PDE point selection; constraints keep their independent coordinates."""
from copy import deepcopy


def boundary_rows(config,sets):
    rows=[]
    for name,points in sets.items():
        if name in ('pressure_gauge','mean_velocity'):continue
        rows.extend(deepcopy(points))
        if name in ('periodic_x','periodic_y'):
            axis=0 if name=='periodic_x' else 1
            for point in points:
                other=list(point);other[axis]=config.x_max if axis==0 else config.y_max;rows.append(other)
    return rows


def collocation_rows(config,interior,sets):
    mode=config.pde_points
    if mode=='interior':return interior
    boundary=boundary_rows(config,sets)
    if not boundary:raise ValueError('Boundary PDE collocation needs at least one boundary sample set.')
    return boundary if mode=='boundary' else [*interior,*boundary]
