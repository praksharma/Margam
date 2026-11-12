"""One DSGNAR recipe shared by compact and expanded graph representations."""
from copy import deepcopy
from . import baseline
from ..components.dsgnar_config import dsgnar_defaults
NAME='DSGNAR PINN'
ROOT='dsgnar_method'
NETWORK_FIELDS=baseline.NETWORK_FIELDS
ADAPTIVE_FIELDS=('dsgnar_settings',)
TRAINING_FIELDS=baseline.TRAINING_FIELDS+ADAPTIVE_FIELDS
PARAMETERS=NETWORK_FIELDS+TRAINING_FIELDS
EXPOSED=('iterations','device')
COMPONENTS=(('network','Neural network',(-240,320),NETWORK_FIELDS),(ROOT,'DSGNAR trainer',(0,0),TRAINING_FIELDS))
CONNECTIONS=(('network',ROOT,'network'),)
INPUTS=baseline.INPUTS
OUTPUT=baseline.OUTPUT
DEFAULTS=dict(deepcopy(baseline.DEFAULTS),method='dsgnar',lr=.001,dsgnar_settings=dsgnar_defaults())


def split(parameters):
    if not isinstance(parameters,dict) or set(parameters)!=set(PARAMETERS) or parameters['method']!='dsgnar':raise ValueError('DSGNAR recipe requires complete network/trainer settings.')
    return {kind:{key:deepcopy(parameters[key]) for key in fields} for kind,_,_,fields in COMPONENTS}


def combine(network,training):
    result={**network,**training};split(result);return result
