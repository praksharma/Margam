"""One gradient-enhanced recipe shared by compact and expanded representations."""
from copy import deepcopy
from . import baseline
from ..components.gpinn_config import gpinn_defaults
NAME='Gradient-enhanced PINN'
ROOT='gpinn_method'
NETWORK_FIELDS=baseline.NETWORK_FIELDS
ADAPTIVE_FIELDS=('gpinn_weights',)
TRAINING_FIELDS=baseline.TRAINING_FIELDS+ADAPTIVE_FIELDS
PARAMETERS=NETWORK_FIELDS+TRAINING_FIELDS
EXPOSED=baseline.EXPOSED
COMPONENTS=(('network','Neural network',(-240,320),NETWORK_FIELDS),(ROOT,'gPINN trainer',(0,0),TRAINING_FIELDS))
CONNECTIONS=(('network',ROOT,'network'),)
INPUTS=baseline.INPUTS
OUTPUT=baseline.OUTPUT
DEFAULTS=dict(deepcopy(baseline.DEFAULTS),method='gpinn',lr=.001,gpinn_weights=gpinn_defaults())


def split(parameters):
    if not isinstance(parameters,dict) or set(parameters)!=set(PARAMETERS) or parameters['method']!='gpinn':raise ValueError('gPINN recipe requires complete network/trainer settings.')
    return {kind:{key:deepcopy(parameters[key]) for key in fields} for kind,_,_,fields in COMPONENTS}


def combine(network,training):
    result={**network,**training};split(result);return result
