"""Steady heat demo: unit conductivity, constant source, zero edge temperature."""
from .poisson import PoissonConfig


def heat_demo_config():
    """A configured baseline example, not a new PINN method or transient solver."""
    return PoissonConfig(shape='Rectangle', points=144, boundary_points=24,
                         layers=(32,32), iterations=2500, lr=.005,
                         source=2., boundary_weight=10.)
