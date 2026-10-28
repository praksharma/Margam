"""Original PINN Burgers problem; current Margam network/training defaults."""
import math
from .poisson import PoissonConfig,BoundarySampling


def burgers_config():
    config=PoissonConfig(shape='SpaceTime1D',x_min=-1.,x_max=1.,y_min=0.,y_max=1.,
        pde_preset='burgers_1d',source=0.,viscosity=.01/math.pi,points=1024,sampling='random',
        boundary_points=64,layers=(32,32,32),iterations=2000,lr=.001,boundary_weight=10.)
    config.bc_targets={'bottom':'-sin(pi*x)'}
    config.boundary_sampling={side:BoundarySampling(64,'random',8+i) for i,side in enumerate(config.sides)}
    return config
