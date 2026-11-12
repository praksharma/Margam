"""Build the selected concrete pointwise architecture; old FCNN stays unchanged."""
from copy import deepcopy
import torch
from .FCNN import FullyConnected
from ..components.network_config import validate_network


def _build_network(config):
    options=validate_network(config.network_type,config.network_options,config.layers)
    inputs=len(config.bounds[0]);outputs=len(config.field_names);bounds=config.bounds;layers=config.layers
    seed=config.seed if config.network_seed is None else config.network_seed
    torch.manual_seed(seed)
    kind=config.network_type
    if kind=='fcnn':network=FullyConnected([inputs,*layers,outputs],'tanh','Xavier normal',input_bounds=bounds)
    elif kind in ('fourier','fourier_mlp'):
        from .fourier import FourierNetwork
        network=FourierNetwork(inputs,outputs,layers,bounds,options,angular=kind=='fourier_mlp')
    elif kind=='affine_fourier':
        from .affine_fourier import AffineFourierNetwork
        network=AffineFourierNetwork(inputs,outputs,layers,bounds,options,config.coordinate_names)
    elif kind=='highway_fourier':
        from .highway import HighwayFourierNetwork
        network=HighwayFourierNetwork(inputs,outputs,layers,bounds,options)
    elif kind=='modified_fourier':
        from .modified_fourier import ModifiedFourierNetwork
        network=ModifiedFourierNetwork(inputs,outputs,layers,bounds,options)
    elif kind=='multiscale_fourier':
        from .multiscale_fourier import MultiscaleFourierNetwork
        network=MultiscaleFourierNetwork(inputs,outputs,layers,bounds,options)
    elif kind=='dgm':
        from .dgm import DGMNetwork
        network=DGMNetwork(inputs,outputs,layers,bounds,options)
    elif kind=='siren':
        from .siren import SirenNetwork
        network=SirenNetwork(inputs,outputs,layers,bounds,options)
    elif kind in ('fourier_filter','gabor_filter'):
        from .multiplicative_filter import MultiplicativeFilterNetwork
        network=MultiplicativeFilterNetwork(inputs,outputs,layers,bounds,options,gabor=kind=='gabor_filter')
    else:raise ValueError('Unsupported pointwise neural network.')
    if options.get('weight_factorization',False):
        from .weight_factorization import factorize_linear_layers
        factorize_linear_layers(network,options['factor_mean'],options['factor_std'])
    else:
        # Disabled options must preserve historical architecture checkpoint specs.
        options={key:value for key,value in options.items() if key not in ('weight_factorization','factor_mean','factor_std')}
    network._margam_spec=deepcopy(dict(type=kind,options=options,layers=list(layers),inputs=inputs,outputs=outputs,bounds=[list(row) for row in bounds],seed=seed))
    if config.method=='inverse_burgers':
        from ..methods.inverse_burgers import InverseBurgersNetwork
        network=InverseBurgersNetwork(network,config.inverse_settings)
    if config.method=='dsgnar':network.to(dtype=getattr(torch,config.dsgnar_settings['precision']))
    return network


def build_network(config):
    if config.method!='dsgnar':return _build_network(config)
    previous=torch.get_default_dtype()
    try:
        torch.set_default_dtype(getattr(torch,config.dsgnar_settings['precision']))
        return _build_network(config)
    finally:torch.set_default_dtype(previous)
