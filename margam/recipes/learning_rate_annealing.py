"""One LRA recipe definition for compact and expanded network/trainer forms."""
from copy import deepcopy
from . import baseline

NAME='Learning-rate annealing PINN'
ROOT='lra_method'
ADAPTIVE_FIELDS=('lra_alpha','lra_interval','lra_epsilon','lra_max')
NETWORK_FIELDS=baseline.NETWORK_FIELDS
TRAINING_FIELDS=baseline.TRAINING_FIELDS+ADAPTIVE_FIELDS
PARAMETERS=NETWORK_FIELDS+TRAINING_FIELDS
EXPOSED=baseline.EXPOSED+('lra_alpha','lra_interval')
COMPONENTS=(('network','Neural network',(-240,320),NETWORK_FIELDS),(ROOT,'LRA trainer',(0,0),TRAINING_FIELDS))
CONNECTIONS=(('network',ROOT,'network'),)
INPUTS=baseline.INPUTS
OUTPUT=baseline.OUTPUT
DEFAULTS=dict(deepcopy(baseline.DEFAULTS),method='lra',lr=.001,lra_alpha=.9,lra_interval=10,lra_epsilon=1e-12,lra_max=1e6)


def split(parameters):
    if not isinstance(parameters,dict) or set(parameters)!=set(PARAMETERS) or parameters['method']!='lra':
        raise ValueError('The LRA recipe requires complete learning-rate annealing settings.')
    return {kind:{key:deepcopy(parameters[key]) for key in fields} for kind,_,_,fields in COMPONENTS}


def combine(network,training):
    result={**network,**training};split(result);return result
