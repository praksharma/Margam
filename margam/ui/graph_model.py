"""Concrete scalar graph assembly; core parameters belong to their block instances."""
from copy import deepcopy
from dataclasses import asdict
from ..templates.poisson import PoissonConfig,BoundarySampling

PARAMETERS = {
    'domain':('shape','x_min','x_max','y_min','y_max'),
    'interior':('points','sampling','seed','interior_data'),
    'boundary':('boundary_points','boundary_sampling','boundary_data'),
    'pde':('source','pde_expression'),
    'bc':('bc_types','bc_values','bc_targets','bc_operators'),
    'network':('layers','network_seed'),
    'method':('iterations','lr','device','pde_weight','boundary_weight'),
}
OWNERS={attribute:kind for kind,attributes in PARAMETERS.items() for attribute in attributes}


def encode_parameter(attribute,value):
    if attribute=='boundary_sampling':return {side:asdict(recipe) if isinstance(recipe,BoundarySampling) else deepcopy(recipe) for side,recipe in value.items()}
    if attribute=='layers':return list(value)
    return deepcopy(value)


def parameters_for(kind,config=None):
    config=config or PoissonConfig()
    result={key:encode_parameter(key,getattr(config,key)) for key in PARAMETERS.get(kind,())}
    if kind=='network' and result['network_seed'] is None:result['network_seed']=config.seed
    return result


def overlay(config,kind,parameters):
    config=deepcopy(config)
    for key in PARAMETERS.get(kind,()):
        value=deepcopy(parameters[key])
        if key=='boundary_sampling':value={side:BoundarySampling(**recipe) for side,recipe in value.items()}
        if key=='layers':value=tuple(value)
        setattr(config,key,value)
    return config


def assemble(nodes,wires,*,require_all=True):
    """Follow one method's typed dependencies; no implicit hidden model inputs."""
    methods=[node for node in nodes if node.kind=='method']
    if len(methods)!=1:raise ValueError('Select a graph with exactly one PINN method root for this scalar executor.')
    incoming={(wire.target.parentItem().identifier,wire.target.name):wire.source.parentItem() for wire in wires}
    def source(node,port,kind):
        result=incoming.get((node.identifier,port))
        if result is None or result.kind!=kind:raise ValueError('Missing compatible '+port+' input on '+node.title)
        return result
    method=methods[0]
    pde=source(method,'pde','pde');bc=source(method,'boundary','bc');network=source(method,'network','network')
    interior=source(pde,'input','interior');boundary=source(bc,'input','boundary')
    domain=source(interior,'input','domain')
    if source(boundary,'input','domain') is not domain:raise ValueError('Interior and boundary sets must use the same geometry instance for this scalar problem.')
    used=[domain,interior,boundary,pde,bc,network,method]
    if require_all and any(node.kind in PARAMETERS and node not in used for node in nodes):raise ValueError('Unconnected core blocks are not executable; remove them or wire a single scalar problem.')
    config=PoissonConfig()
    for node in used:config=overlay(config,node.kind,node.settings['parameters'])
    return config
