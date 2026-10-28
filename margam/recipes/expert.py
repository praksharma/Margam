"""Expert PINN is Margam's name for the Expert's Guide v1 recommended pipeline."""
from copy import deepcopy
import math
from . import baseline
from ..components.expert_config import expert_defaults

NAME='Expert PINN'
ROOT='expert_method'
NETWORK_FIELDS=baseline.NETWORK_FIELDS
ADAPTIVE_FIELDS=('expert_settings',)
TRAINING_FIELDS=baseline.TRAINING_FIELDS+ADAPTIVE_FIELDS
PARAMETERS=NETWORK_FIELDS+TRAINING_FIELDS
EXPOSED=baseline.EXPOSED
COMPONENTS=(('network','Neural network',(-240,320),NETWORK_FIELDS),(ROOT,'Expert trainer',(0,0),TRAINING_FIELDS))
CONNECTIONS=(('network',ROOT,'network'),)
INPUTS=baseline.INPUTS
OUTPUT=baseline.OUTPUT
DEFAULTS=dict(deepcopy(baseline.DEFAULTS),method='expert',lr=.001,expert_settings=expert_defaults(),
              network_type='fourier',network_options=dict(encoding='gaussian',features=32,scale=1/(2*math.pi),weight_factorization=True,factor_mean=1.,factor_std=.1),layers=[32,32,32])


def split(parameters):
    if not isinstance(parameters,dict) or set(parameters)!=set(PARAMETERS) or parameters['method']!='expert':raise ValueError('The Expert recipe requires complete network/trainer settings.')
    return {kind:{key:deepcopy(parameters[key]) for key in fields} for kind,_,_,fields in COMPONENTS}


def combine(network,training):
    result={**network,**training};split(result);return result
