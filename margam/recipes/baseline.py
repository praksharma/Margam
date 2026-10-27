"""Baseline PINN recipe: fixed samples, replaceable dense network, full-batch Adam.

The compact block and expanded graph are two representations of this definition.
Physics, sampling, BCs and result requests connect externally. The numerical
implementation remains margam.methods.pinn.PINN; shared components stay shared.
"""
NAME = 'Baseline PINN'
NETWORK_FIELDS = ('layers', 'network_seed')
TRAINING_FIELDS = ('method', 'iterations', 'lr', 'device', 'pde_weight', 'boundary_weight')
PARAMETERS = NETWORK_FIELDS + TRAINING_FIELDS
EXPOSED = ('iterations', 'lr', 'device')
COMPONENTS = (
    ('network', 'Neural network', (-240, 320), NETWORK_FIELDS),
    ('method', 'PINN method', (0, 0), TRAINING_FIELDS),
)
CONNECTIONS = (('network', 'method', 'network'),)
INPUTS = (('pde', 'PDE residual', 'residual'), ('boundary', 'Boundary conditions', 'constraints'))
OUTPUT = ('output', 'PINN run', 'pinn_run')
DEFAULTS = dict(layers=[16, 16], network_seed=7, method='baseline', iterations=1000,
                lr=.01, device='cpu', pde_weight=1., boundary_weight=1.)


def split(parameters):
    """Materialise both components from the very same stored recipe settings."""
    from copy import deepcopy
    if not isinstance(parameters,dict) or set(parameters) != set(PARAMETERS) or parameters['method'] != 'baseline':
        raise ValueError('The baseline recipe requires baseline training and complete component settings.')
    return {kind: {key: deepcopy(parameters[key]) for key in fields}
            for kind, _, _, fields in COMPONENTS}


def combine(network, training):
    result = {**network, **training}
    split(result)
    return result
