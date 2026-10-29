"""Manufactured scalar Poisson problem on imported triangular point sets."""
from pathlib import Path
from .poisson import PoissonConfig


def prepared_triangle_config(data_directory=None):
    directory=Path(data_directory) if data_directory is not None else Path('assets/training/triangle')
    def source(name,normals='',values=''):return dict(path=str(directory/name),coordinates='x,y',normals=normals,values=values)
    c=PoissonConfig(shape='Prepared2D',source=-4.,points=55,boundary_points=12,iterations=1000,lr=.005)
    c.interior_data=source('interior.csv')
    c.boundary_data={'base':source('base.csv',values='value'),'upright':source('upright.csv',normals='nx,ny'),'slope':source('slope.csv',normals='nx,ny',values='value')}
    c.bc_types.update(base='Dirichlet',upright='Neumann',slope='Neumann')
    c.bc_values.update(base=0.,upright=0.,slope=0.)
    return c
