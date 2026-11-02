"""Observation-driven example; no separate imposed boundary/initial losses."""
from pathlib import Path
from .poisson import PoissonConfig


def inverse_example_config():
    path=Path(__file__).resolve().parents[2]/'assets/training/inverse_burgers/observations.csv'
    return PoissonConfig(shape='SpaceTime1D',x_min=-1.,x_max=1.,y_min=0.,y_max=1.,method='inverse_burgers',layers=(32,32,32),iterations=1000,lr=.001,interior_data=dict(path=str(path),coordinates='x,t',values='u',normals=''))
