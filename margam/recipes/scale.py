"""One replaceable network plus the independent Scale-PINN correction trainer."""
from copy import deepcopy
from . import baseline
from ..components.scale_config import scale_defaults
NAME='Scale-PINN'
ROOT='scale_method'
NETWORK_FIELDS=baseline.NETWORK_FIELDS
ADAPTIVE_FIELDS=('scale_settings',)
TRAINING_FIELDS=baseline.TRAINING_FIELDS+ADAPTIVE_FIELDS
PARAMETERS=NETWORK_FIELDS+TRAINING_FIELDS
EXPOSED=baseline.EXPOSED
COMPONENTS=(('network','Neural network',(-240,320),NETWORK_FIELDS),(ROOT,'Scale trainer',(0,0),TRAINING_FIELDS))
CONNECTIONS=(('network',ROOT,'network'),)
INPUTS=baseline.INPUTS
OUTPUT=baseline.OUTPUT
DEFAULTS=dict(deepcopy(baseline.DEFAULTS),method='scale',lr=.001,scale_settings=scale_defaults(),network_type='affine_fourier',network_options=dict(input_transform='',features=32,cycles=1.,normalize_inputs=False,activation='silu'),layers=[32,32,32])


def split(parameters):
    if not isinstance(parameters,dict) or set(parameters)!=set(PARAMETERS) or parameters['method']!='scale':raise ValueError('The Scale recipe requires complete network/trainer settings.')
    return {kind:{key:deepcopy(parameters[key]) for key in fields} for kind,_,_,fields in COMPONENTS}


def combine(network,training):
    result={**network,**training};split(result);return result
