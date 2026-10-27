"""SA-PINN recipe: replaceable network and pointwise Adam min/max trainer."""
from copy import deepcopy
from . import baseline

NAME = 'Self-adaptive PINN'
ROOT = 'sa_method'
ADAPTIVE_FIELDS = ('sa_lr','sa_initial','sa_max','sa_boundary')
NETWORK_FIELDS = baseline.NETWORK_FIELDS
TRAINING_FIELDS = baseline.TRAINING_FIELDS + ADAPTIVE_FIELDS
PARAMETERS = NETWORK_FIELDS + TRAINING_FIELDS
EXPOSED = baseline.EXPOSED + ('sa_lr',)
COMPONENTS = (
    ('network','Neural network',(-240,320),NETWORK_FIELDS),
    (ROOT,'Self-adaptive trainer',(0,0),TRAINING_FIELDS),
)
CONNECTIONS = (('network',ROOT,'network'),)
INPUTS = baseline.INPUTS
OUTPUT = baseline.OUTPUT
DEFAULTS = dict(deepcopy(baseline.DEFAULTS),method='self_adaptive',sa_lr=.005,sa_initial=1.,sa_max=1000.,sa_boundary=False)


def split(parameters):
    if not isinstance(parameters,dict) or set(parameters)!=set(PARAMETERS) or parameters['method']!='self_adaptive':
        raise ValueError('The self-adaptive recipe requires complete SA-PINN settings.')
    return {kind:{key:deepcopy(parameters[key]) for key in fields} for kind,_,_,fields in COMPONENTS}


def combine(network,training):
    result={**network,**training};split(result);return result
