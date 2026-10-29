"""Concrete PDE families; boundary conditions and reference sources remain separate."""
from dataclasses import dataclass
import math
from .poisson import FLOW_RESIDUALS


@dataclass(frozen=True)
class PDEPreset:
    label: str
    equation: str
    shapes: tuple
    forcing: bool = True


PREDEFINED_PDES = {
    'kdv_1d':PDEPreset('Korteweg–de Vries · 1D + time','uₜ + c u uₓ + ν uₓₓₓ = 0',('SpaceTime1D',),False),
    'burgers_1d': PDEPreset('Burgers · 1D + time','uₜ + u uₓ − ν uₓₓ = 0',('SpaceTime1D',),False),
    'poisson': PDEPreset('Poisson', 'Δu + c + f(x,y) = 0', ('Interval','Rectangle','Prepared2D')),
    'helmholtz': PDEPreset('Helmholtz', 'Δu + c u = f(x,y), c = k²', ('Interval','Rectangle','Prepared2D')),
    'reaction_diffusion': PDEPreset('Reaction–diffusion', '−Δu + c u = f(x,y)', ('Interval','Rectangle','Prepared2D')),
    'advection_diffusion_1d': PDEPreset('Advection–diffusion · 1D', 'c uₓ − ν uₓₓ = f(x)', ('Interval',)),
    'incompressible_2d': PDEPreset('Steady incompressible flow · 2D', 'Momentum x/y + ∂u/∂x + ∂v/∂y = 0', ('Rectangle',),False),
}


def resolve_predefined(config):
    """Derive equations only; never choose BCs, samples, networks or exact values."""
    if config.pde_preset not in PREDEFINED_PDES:raise ValueError('Choose an implemented predefined PDE.')
    forcing='('+config.pde_forcing+')'
    config.physics='scalar'
    if config.pde_preset.startswith('expert_'):
        config.physics='paper_case';config.pde_expression='';return config
    if config.pde_preset=='kdv_1d':config.pde_expression='D(u,t)+c*u*D(u,x)+nu*D(D(D(u,x),x),x)'
    elif config.pde_preset=='burgers_1d':config.pde_expression='D(u,t)+u*D(u,x)-nu*D(D(u,x),x)'
    elif config.pde_preset=='poisson':
        config.pde_expression='' if config.pde_forcing.strip()=='0' else 'laplace(u)+c+'+forcing
    elif config.pde_preset=='helmholtz':config.pde_expression='laplace(u)+c*u-'+forcing
    elif config.pde_preset=='reaction_diffusion':config.pde_expression='-laplace(u)+c*u-'+forcing
    elif config.pde_preset=='advection_diffusion_1d':config.pde_expression='c*D(u,x)-('+repr(config.viscosity)+')*D(D(u,x),x)-'+forcing
    else:
        config.physics='incompressible_2d';config.flow_residuals=FLOW_RESIDUALS;config.pde_expression=''
    return config


def validate_predefined(config):
    from ..components.expressions import Expression
    preset=PREDEFINED_PDES[config.pde_preset]
    if config.shape not in preset.shapes:raise ValueError(preset.label+' supports '+', '.join(preset.shapes)+'.')
    if preset.forcing:Expression(config.pde_forcing,len(config.bounds[0]),target=True)
    if config.pde_preset=='advection_diffusion_1d' and config.viscosity<=0:raise ValueError('Advection–diffusion requires positive diffusivity ν.')

    if config.pde_preset=='burgers_1d' and (not math.isfinite(config.viscosity) or config.viscosity<=0):raise ValueError('Burgers requires positive finite viscosity ν.')
    if config.pde_preset=='helmholtz' and config.source<0:raise ValueError('Helmholtz requires non-negative c = k².')

from .expert_examples import CASES
for _key,_case in CASES.items():PREDEFINED_PDES[_key]=PDEPreset(_case['label'].replace(' · Re=3200','') if _key=='expert_cavity' else _case['label'],_case['equation'],(_case['shape'],),False)
