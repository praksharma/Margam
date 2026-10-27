"""Concrete PDE families; boundary conditions and reference sources remain separate."""
from dataclasses import dataclass
from .poisson import FLOW_RESIDUALS


@dataclass(frozen=True)
class PDEPreset:
    label: str
    equation: str
    shapes: tuple
    forcing: bool = True


PREDEFINED_PDES = {
    'poisson': PDEPreset('Poisson', 'Δu + c + f(x,y) = 0', ('Interval','Rectangle')),
    'reaction_diffusion': PDEPreset('Reaction–diffusion', '−Δu + c u = f(x,y)', ('Interval','Rectangle')),
    'advection_diffusion_1d': PDEPreset('Advection–diffusion · 1D', 'c uₓ − ν uₓₓ = f(x)', ('Interval',)),
    'incompressible_2d': PDEPreset('Steady incompressible flow · 2D', 'Momentum x/y + ∂u/∂x + ∂v/∂y = 0', ('Rectangle',),False),
}


def resolve_predefined(config):
    """Derive equations only; never choose BCs, samples, networks or exact values."""
    if config.pde_preset not in PREDEFINED_PDES:raise ValueError('Choose an implemented predefined PDE.')
    forcing='('+config.pde_forcing+')'
    config.physics='scalar'
    if config.pde_preset=='poisson':
        config.pde_expression='' if config.pde_forcing.strip()=='0' else 'laplace(u)+c+'+forcing
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
