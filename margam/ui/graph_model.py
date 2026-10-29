"""Concrete supported-physics graph assembly; core parameters belong to their block instances."""
from copy import deepcopy
from dataclasses import asdict
from ..templates.poisson import PoissonConfig,BoundarySampling
from ..recipes import baseline, self_adaptive, learning_rate_annealing, expert, scale, gradient_enhanced, residual_attention, COMPACT_RECIPES, TRAINER_RECIPES

PARAMETERS = {
    'domain':('shape','x_min','x_max','y_min','y_max','t_min','t_max','cylinder_x','cylinder_y','cylinder_radius'),
    'interior':('points','sampling','seed','interior_data','adaptive_sampling'),
    'rar':('points','sampling','seed','interior_data','adaptive_sampling'),
    'importance':('points','sampling','seed','interior_data','adaptive_sampling'),
    'boundary':('boundary_points','boundary_sampling','boundary_data'),
    'predefined_pde':('pde_preset','pde_forcing','source','viscosity','fourth_order'),
    'pde':('source','pde_expression','physics','viscosity','flow_residuals'),
    'bc':('bc_types','bc_values','bc_targets','bc_operators','flow_boundary','pressure_point','pressure_value','pressure_weight','startup_time'),
    'network':baseline.NETWORK_FIELDS,
    'method':baseline.TRAINING_FIELDS,
    'recipe':baseline.PARAMETERS,
    'sa_method':self_adaptive.TRAINING_FIELDS,
    'sa_recipe':self_adaptive.PARAMETERS,
    'lra_method':learning_rate_annealing.TRAINING_FIELDS,
    'lra_recipe':learning_rate_annealing.PARAMETERS,
    'rba_method':residual_attention.TRAINING_FIELDS,
    'rba_recipe':residual_attention.PARAMETERS,
    'gpinn_method':gradient_enhanced.TRAINING_FIELDS,
    'gpinn_recipe':gradient_enhanced.PARAMETERS,
    'scale_method':scale.TRAINING_FIELDS,
    'scale_recipe':scale.PARAMETERS,
    'expert_method':expert.TRAINING_FIELDS,
    'expert_recipe':expert.PARAMETERS,
}
PARAMETERS['interior_data']=PARAMETERS['interior']
PARAMETERS['boundary_data']=PARAMETERS['boundary']

OWNERS={attribute:kind for kind,attributes in PARAMETERS.items() if kind not in ('predefined_pde','sa_method','lra_method','expert_method','scale_method','gpinn_method','rba_method',*COMPACT_RECIPES) for attribute in attributes}
OWNERS.update({key:'interior' for key in PARAMETERS['interior']})
OWNERS.update({key:'boundary' for key in PARAMETERS['boundary']})
OWNERS.update(fourth_order='pde',pde_preset='pde',pde_forcing='pde',**{key:'method' for key in (*self_adaptive.ADAPTIVE_FIELDS,*learning_rate_annealing.ADAPTIVE_FIELDS,*expert.ADAPTIVE_FIELDS,*scale.ADAPTIVE_FIELDS,*gradient_enhanced.ADAPTIVE_FIELDS,*residual_attention.ADAPTIVE_FIELDS)})

def role(kind):return {'rba_recipe':'method','rba_method':'method','gpinn_recipe':'method','gpinn_method':'method','interior_data':'interior','boundary_data':'boundary','scale_recipe':'method','scale_method':'method','expert_recipe':'method','expert_method':'method','rar':'interior','importance':'interior','predefined_pde':'pde','recipe':'method','sa_recipe':'method','sa_method':'method','lra_method':'method','lra_recipe':'method'}.get(kind,kind)


def encode_parameter(attribute,value):
    if attribute=='boundary_sampling':return {side:asdict(recipe) if isinstance(recipe,BoundarySampling) else deepcopy(recipe) for side,recipe in value.items()}
    if attribute in ('layers','flow_residuals','pressure_point'):return list(value)
    return deepcopy(value)


def parameters_for(kind,config=None):
    config=config or PoissonConfig()
    result={key:encode_parameter(key,getattr(config,key)) for key in PARAMETERS.get(kind,())}
    if kind in ('interior','interior_data'):result['adaptive_sampling']={}
    if kind in ('rar','importance'):
        from ..components.adaptive_sampling import sampling_defaults
        if result['adaptive_sampling'].get('method')!=kind:result['adaptive_sampling']=sampling_defaults(kind)
    if kind=='predefined_pde' and not result['pde_preset']:result['pde_preset']='poisson'
    if kind in COMPACT_RECIPES:return deepcopy(COMPACT_RECIPES[kind].DEFAULTS)
    if kind=='sa_method':result['method']='self_adaptive'
    if kind=='rba_method':result['method']='rba'
    if kind=='gpinn_method':result['method']='gpinn'
    if kind=='scale_method':result['method']='scale'
    if kind=='expert_method':result['method']='expert'
    if kind=='lra_method':result['method']='lra'
    if kind=='network' and result['network_seed'] is None:result['network_seed']=config.seed
    return result


def overlay(config,kind,parameters):
    config=deepcopy(config)
    for key in PARAMETERS.get(kind,()):
        value=deepcopy(parameters[key])
        if key=='boundary_sampling':value={side:BoundarySampling(**recipe) for side,recipe in value.items()}
        if key in ('layers','flow_residuals','pressure_point'):value=tuple(value)
        setattr(config,key,value)
    if kind=='predefined_pde':
        from ..templates.predefined_pdes import resolve_predefined
        resolve_predefined(config)
    elif kind=='pde':config.pde_preset=''
    if config.shape=='Prepared2D':
        for name in config.sides:
            config.bc_types.setdefault(name,'Dirichlet');config.bc_values.setdefault(name,0.)
    return config


def assemble(nodes,wires,*,require_all=True):
    """Follow one method's typed dependencies; no implicit hidden model inputs."""
    methods=[node for node in nodes if role(node.kind)=='method']
    if len(methods)!=1:raise ValueError('Select a graph with exactly one PINN method root for this executor.')
    incoming={(wire.target.parentItem().identifier,wire.target.name):wire.source.parentItem() for wire in wires}
    def source(node,port,kind):
        result=incoming.get((node.identifier,port))
        if result is None or role(result.kind)!=kind:raise ValueError('Missing compatible '+port+' input on '+node.title)
        return result
    method=methods[0]
    pde=source(method,'pde','pde');bc=source(method,'boundary','bc')
    network=source(method,'network','network') if method.kind in TRAINER_RECIPES else None
    if method.kind in COMPACT_RECIPES:COMPACT_RECIPES[method.kind].split(method.settings['parameters'])
    interior=source(pde,'input','interior');boundary=source(bc,'input','boundary')
    domain=source(interior,'input','domain')
    if source(boundary,'input','domain') is not domain:raise ValueError('Interior and boundary sets must use the same geometry instance for this problem.')
    used=[domain,interior,boundary,pde,bc]+([network] if network else [])+[method]
    if require_all and any(node.kind in PARAMETERS and node not in used for node in nodes):raise ValueError('Unconnected core blocks are not executable; remove them or wire a single supported problem.')
    config=PoissonConfig()
    for node in used:config=overlay(config,node.kind,node.settings['parameters'])
    if interior.kind=='interior_data' and not config.interior_data:raise ValueError('Map a coordinate source on Interior data.')
    if boundary.kind=='boundary_data' and not config.boundary_data:raise ValueError('Map at least one named set on Boundary data.')
    if config.shape=='Prepared2D' and (interior.kind!='interior_data' or boundary.kind!='boundary_data'):raise ValueError('Prepared 2D uses Interior data and Boundary data blocks.')
    return config
