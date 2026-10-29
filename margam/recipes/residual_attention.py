"""One RBA recipe shared by compact and expanded graph representations."""
from copy import deepcopy
from . import baseline
from ..components.rba_config import rba_defaults
NAME='Residual-based attention PINN'
ROOT='rba_method'
NETWORK_FIELDS=baseline.NETWORK_FIELDS
ADAPTIVE_FIELDS=('rba_settings',)
TRAINING_FIELDS=baseline.TRAINING_FIELDS+ADAPTIVE_FIELDS
PARAMETERS=NETWORK_FIELDS+TRAINING_FIELDS
EXPOSED=baseline.EXPOSED
COMPONENTS=(('network','Neural network',(-240,320),NETWORK_FIELDS),(ROOT,'RBA trainer',(0,0),TRAINING_FIELDS))
CONNECTIONS=(('network',ROOT,'network'),)
INPUTS=baseline.INPUTS
OUTPUT=baseline.OUTPUT
DEFAULTS=dict(deepcopy(baseline.DEFAULTS),method='rba',lr=.001,rba_settings=rba_defaults())


def split(parameters):
    if not isinstance(parameters,dict) or set(parameters)!=set(PARAMETERS) or parameters['method']!='rba':raise ValueError('RBA recipe requires complete network/trainer settings.')
    return {kind:{key:deepcopy(parameters[key]) for key in fields} for kind,_,_,fields in COMPONENTS}


def combine(network,training):
    result={**network,**training};split(result);return result
