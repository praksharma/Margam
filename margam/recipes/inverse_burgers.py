"""Two trainable Burgers coefficients; fixed observed field data, replaceable NN."""
from copy import deepcopy
from .baseline import NETWORK_FIELDS,TRAINING_FIELDS
NAME='Inverse Burgers PINN'
ROOT='inverse_method'
PARAMETERS=NETWORK_FIELDS+TRAINING_FIELDS
EXPOSED=('iterations','lr','device','pde_weight','boundary_weight')
COMPONENTS=(('network','Neural network',(-240,320),NETWORK_FIELDS),('inverse_method','Inverse trainer',(0,0),TRAINING_FIELDS))
CONNECTIONS=(('network','inverse_method','network'),)
INPUTS=(('pde','Inverse Burgers residual','inverse_residual'),('observations','Observed fields','observations'))
OUTPUT=('output','PINN run','pinn_run')
DEFAULTS=dict(layers=[32,32,32],network_seed=7,method='inverse_burgers',iterations=1000,lr=.001,device='cpu',pde_weight=1.,boundary_weight=1.,network_type='fcnn',network_options={},loss_formulation='mean')

def split(parameters):
    if not isinstance(parameters,dict) or set(parameters)!=set(PARAMETERS) or parameters['method']!='inverse_burgers':raise ValueError('Inverse recipe requires its complete network/trainer settings.')
    return {kind:{key:deepcopy(parameters[key]) for key in fields} for kind,_,_,fields in COMPONENTS}

def combine(network,training):
    result={**network,**training};split(result);return result
