"""Small verified runnable problems; example choices are distinct from PDE presets."""
from dataclasses import dataclass
from .poisson import PoissonConfig


@dataclass(frozen=True)
class ExampleProblem:
    label: str
    description: str
    reference: str


EXAMPLE_PROBLEMS = {
    'expert_burgers':ExampleProblem('Burgers — Expert PINN · 1D + time','Expert’s Guide v1 pipeline; Margam label, Fourier/RWF, causal training and gradient-norm balancing.',''),
    'burgers_1d':ExampleProblem('Burgers — 1D + time','Original PINN problem: ν=0.01/π, sine initial profile, zero endpoint BCs; current Adam defaults.',''),
    'lra_poisson':ExampleProblem('Poisson — LRA-PINN','Gradient-statistic constraint balancing; coefficient cap 100; existing FCNN and explicit reference.','x*(1-x)'),
    'self_adaptive_poisson':ExampleProblem('Poisson — SA-PINN','Pointwise squared PDE masks with Adam ascent; explicit reference.','x*(1-x)'),
    'interval_poisson':ExampleProblem('Poisson — baseline','Constant-source Poisson; zero endpoints.','x*(1-x)'),
    'reaction_diffusion':ExampleProblem('Reaction–diffusion','Positive reaction coefficient; sine manufactured forcing.','sin(pi*x)'),
    'advection_diffusion':ExampleProblem('Advection–diffusion','Unit advection, diffusivity 0.1, unequal Dirichlet endpoints.','x'),
    'helmholtz_lra_2d':ExampleProblem('2D Helmholtz — Fourier LRA-PINN','LRA paper sine modes (1,4) on [-1,1]²; existing Fourier-feature network, soft zero Dirichlet boundaries.','sin(pi*x)*sin(4*pi*y)'),
    'poisson_sine_2d':ExampleProblem('2D Poisson — sinusoidal forcing','Manufactured rectangular solution; zero edges.','sin(pi*x)*sin(pi*y)'),
}


def _example_config(identifier):
    if identifier not in EXAMPLE_PROBLEMS:raise ValueError('Unknown example problem.')
    if identifier.startswith('rba_'):
        from .rba_examples import rba_example_config
        return rba_example_config(identifier)
    if identifier.startswith('gpinn_'):
        from .gpinn_examples import gpinn_example_config
        return gpinn_example_config(identifier)
    if identifier.startswith('scale_'):
        from .scale_examples import scale_example_config
        return scale_example_config(identifier)
    if identifier in ('expert_allen_cahn','expert_advection','expert_stokes','expert_ks','expert_cavity','expert_torus','expert_cylinder'):
        from .expert_examples import case_config
        return case_config(identifier)
    if identifier=='expert_burgers':
        from .burgers import burgers_config
        from ..recipes.expert import DEFAULTS
        from copy import deepcopy
        config=burgers_config()
        for key,value in DEFAULTS.items():setattr(config,key,tuple(value) if key=='layers' else deepcopy(value))
        return config
    if identifier=='burgers_1d':
        from .burgers import burgers_config
        return burgers_config()
    config=PoissonConfig(shape='Interval',points=32,iterations=1500,lr=.005,layers=(16,16),boundary_weight=10.,pde_preset='poisson')
    if identifier=='lra_poisson':config.method='lra';config.lr=.001;config.boundary_weight=1.;config.iterations=2000;config.lra_max=100.
    elif identifier=='self_adaptive_poisson':config.method='self_adaptive'
    elif identifier=='reaction_diffusion':
        config.pde_preset='reaction_diffusion';config.source=1.;config.pde_forcing='(pi*pi+1)*sin(pi*x)';config.method='hard_dirichlet'
    elif identifier=='advection_diffusion':
        config.pde_preset='advection_diffusion_1d';config.source=1.;config.viscosity=.1;config.pde_forcing='1';config.bc_values['right']=1.
    elif identifier=='helmholtz_lra_2d':
        config.shape='Rectangle';config.x_min=config.y_min=-1.;config.x_max=config.y_max=1.
        config.pde_preset='helmholtz';config.source=1.;config.pde_forcing='(1-17*pi*pi)*sin(pi*x)*sin(4*pi*y)'
        config.points=1024;config.boundary_points=64;config.layers=(32,32);config.iterations=5000
        config.method='lra';config.lr=.001;config.boundary_weight=1.;config.lra_max=1e4;config.network_seed=7
        config.network_type='fourier';config.network_options={'encoding':'axis','frequencies':[1.,2.,3.,4.],'scale':1.}
    elif identifier=='poisson_sine_2d':
        config.shape='Rectangle';config.points=144;config.boundary_points=24;config.layers=(32,32);config.iterations=2500
        config.source=0.;config.pde_forcing='2*pi*pi*sin(pi*x)*sin(pi*y)';config.method='hard_dirichlet'
    return config

from .expert_examples import CASES
for _key,_case in CASES.items():EXAMPLE_PROBLEMS[_key]=ExampleProblem(_case['label'],'Expert’s Guide example; see its case notes and labelled initial-field variations.','')

from .scale_examples import SCALE_EXAMPLES
for _key,(_label,_description) in SCALE_EXAMPLES.items():EXAMPLE_PROBLEMS[_key]=ExampleProblem(_label,_description,'')

from .gpinn_examples import GPINN_EXAMPLES
for _key,(_label,_description,_reference) in GPINN_EXAMPLES.items():EXAMPLE_PROBLEMS[_key]=ExampleProblem(_label,_description,_reference)

from .rba_examples import RBA_EXAMPLES
for _key,(_label,_description,_reference) in RBA_EXAMPLES.items():EXAMPLE_PROBLEMS[_key]=ExampleProblem(_label,_description,_reference)


def example_config(identifier):
    config=_example_config(identifier)
    config.pde_points='both'
    return config
