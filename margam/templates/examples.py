"""Small verified runnable problems; example choices are distinct from PDE presets."""
from dataclasses import dataclass
from .poisson import PoissonConfig


@dataclass(frozen=True)
class ExampleProblem:
    label: str
    description: str
    reference: str


EXAMPLE_PROBLEMS = {
    'interval_poisson':ExampleProblem('Poisson · interval','Constant-source Poisson; zero endpoints.','x*(1-x)'),
    'reaction_diffusion':ExampleProblem('Reaction–diffusion · interval','Positive reaction coefficient; sine manufactured forcing.','sin(pi*x)'),
    'advection_diffusion':ExampleProblem('Advection–diffusion · interval','Unit advection, diffusivity 0.1, unequal Dirichlet endpoints.','x'),
    'poisson_sine_2d':ExampleProblem('Poisson · 2D sine forcing','Manufactured rectangular solution; zero edges.','sin(pi*x)*sin(pi*y)'),
}


def example_config(identifier):
    if identifier not in EXAMPLE_PROBLEMS:raise ValueError('Unknown example problem.')
    config=PoissonConfig(shape='Interval',points=32,iterations=1500,lr=.005,layers=(16,16),boundary_weight=10.,pde_preset='poisson')
    if identifier=='reaction_diffusion':
        config.pde_preset='reaction_diffusion';config.source=1.;config.pde_forcing='(pi*pi+1)*sin(pi*x)';config.method='hard_dirichlet'
    elif identifier=='advection_diffusion':
        config.pde_preset='advection_diffusion_1d';config.source=1.;config.viscosity=.1;config.pde_forcing='1';config.bc_values['right']=1.
    elif identifier=='poisson_sine_2d':
        config.shape='Rectangle';config.points=144;config.boundary_points=24;config.layers=(32,32);config.iterations=2500
        config.source=0.;config.pde_forcing='2*pi*pi*sin(pi*x)*sin(pi*y)';config.method='hard_dirichlet'
    return config
