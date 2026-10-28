"""Concrete Scale starter cases; shared physics, independently chosen networks."""
from copy import deepcopy
from .expert_examples import case_config
from ..recipes.scale import DEFAULTS


def scale_example_config(key):
    if key not in ('scale_allen_cahn','scale_cavity_3200'):raise ValueError('Unknown Scale example.')
    c=case_config('expert_allen_cahn' if key=='scale_allen_cahn' else 'expert_cavity')
    for name,value in DEFAULTS.items():setattr(c,name,tuple(value) if name=='layers' else deepcopy(value))
    c.iterations=2000;c.points=256;c.boundary_points=32;c.scale_settings['gamma']=c.viscosity
    if key=='scale_cavity_3200':c.scale_settings.update(tau_sc=.06,tau_alpha=1.);c.boundary_weight=15.;c.pde_weight=3.;c.lr=.0005
    else:c.boundary_weight=100.
    return c
