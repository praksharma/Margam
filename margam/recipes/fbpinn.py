"""One FBPINN recipe shared by compact and expanded graph representations."""
from copy import deepcopy
from . import baseline
from ..components.fbpinn_config import fbpinn_defaults

NAME = 'Finite Basis PINN'
ROOT = 'fbpinn_method'
NETWORK_FIELDS = baseline.NETWORK_FIELDS
ADAPTIVE_FIELDS = ('fbpinn_settings',)
TRAINING_FIELDS = baseline.TRAINING_FIELDS + ADAPTIVE_FIELDS
PARAMETERS = NETWORK_FIELDS + TRAINING_FIELDS
EXPOSED = ('iterations', 'lr', 'device')
COMPONENTS = (
    ('network', 'Neural network', (-240, 320), NETWORK_FIELDS),
    (ROOT, 'FBPINN trainer', (0, 0), TRAINING_FIELDS),
)
CONNECTIONS = (('network', ROOT, 'network'),)
INPUTS = baseline.INPUTS
OUTPUT = baseline.OUTPUT
DEFAULTS = dict(
    deepcopy(baseline.DEFAULTS),
    method='fbpinn',
    lr=0.005,
    fbpinn_settings=fbpinn_defaults(),
)


def split(parameters):
    if not isinstance(parameters, dict) or set(parameters) != set(PARAMETERS) or parameters['method'] != 'fbpinn':
        raise ValueError('FBPINN recipe requires complete network/trainer settings.')
    return {kind: {key: deepcopy(parameters[key]) for key in fields} for kind, _, _, fields in COMPONENTS}


def combine(network, training):
    result = {**network, **training}
    split(result)
    return result
