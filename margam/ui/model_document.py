"""Versioned, data-only model documents for the current Poisson builder."""
from dataclasses import asdict, fields
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile

from ..templates.poisson import PoissonConfig, BoundarySampling, FLOW_RESIDUALS, flow_boundary_defaults
from ..components.sampling import SAMPLERS
from .ports import INPUTS, OUTPUTS

MAX_BYTES = 4 * 1024 * 1024
SIDES = {'left', 'right', 'bottom', 'top'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def decode_document(data):
    """Validate editor structure; incomplete/physically invalid drafts are allowed."""
    require(isinstance(data, dict) and set(data) == {'format', 'version', 'config', 'nodes', 'wires'}, 'Invalid Margam model document.')
    require(data['format'] == 'margam-model' and type(data['version']) is int and data['version'] in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16), 'Unsupported Margam model format/version.')
    c = data['config']
    if isinstance(c,dict) and data['version'] in (1,2,3,4,5,6,7,8,9,10):
        c=dict(c)
        for key,value in (('pde_expression',''),('bc_targets',{}),('bc_operators',{}),('interior_data',{}),('boundary_data',{}),('network_seed',None),('method','baseline'),('physics','scalar'),('viscosity',.01),('flow_residuals',FLOW_RESIDUALS),('flow_boundary',flow_boundary_defaults()),('pressure_point',(0.,0.)),('pressure_value',0.),('pressure_weight',1.),('pde_preset',''),('pde_forcing','0'),('sa_lr',.005),('sa_initial',1.),('sa_max',1000.),('sa_boundary',False),('network_type','fcnn'),('network_options',{}),('lra_alpha',.9),('lra_interval',10),('lra_epsilon',1e-12),('lra_max',1e6)):c.setdefault(key,value)
    if isinstance(c,dict) and data['version']<13:c=dict(c,adaptive_sampling=c.get('adaptive_sampling',{}))
    if isinstance(c,dict) and data['version']<14:c=dict(c,loss_formulation=c.get('loss_formulation','mean'))
    if isinstance(c,dict) and data['version']<16:
        from ..components.expert_config import expert_defaults
        c=dict(c,expert_settings=c.get('expert_settings',expert_defaults()))
    require(isinstance(c, dict) and set(c) == {f.name for f in fields(PoissonConfig)}, 'Invalid template settings.')
    require(c['shape'] in ('Interval', 'Rectangle','SpaceTime1D') and c['sampling'] in SAMPLERS and c['device'] in ('cpu', 'auto', 'cuda'), 'Unsupported template option.')
    for key in ('x_min','x_max','y_min','y_max','source','lr','pde_weight','boundary_weight'):
        require(number(c[key]), f'{key} must be finite numeric data.')
    for key,lower,upper in (('points',1,100000),('boundary_points',1,10000),
                            ('seed',0,2**31-1),('iterations',1,100000)):
        require(type(c[key]) is int and lower <= c[key] <= upper, f'Invalid {key}.')
    require(isinstance(c['layers'], (list,tuple)) and len(c['layers']) <= 256 and all(type(v) is int and abs(v) <= 1000000 for v in c['layers']), 'Invalid network layers.')
    from ..components.network_config import network_options
    network_options(c['network_type'],c['network_options'])
    for key in ('bc_types','bc_values'):
        require(isinstance(c[key],dict) and set(c[key]) == SIDES, 'Boundary settings need all four named sides.')
    require(all(v in ('Dirichlet','Neumann','Custom') for v in c['bc_types'].values()), 'Unsupported boundary condition.')
    require(all(number(v) for v in c['bc_values'].values()), 'Boundary targets must be finite.')
    require(isinstance(c['pde_expression'],str) and len(c['pde_expression'])<=512, 'Invalid PDE expression.')
    for key in ('bc_targets','bc_operators'):
        require(isinstance(c[key],dict) and set(c[key])<=SIDES and all(isinstance(v,str) and len(v)<=512 for v in c[key].values()), 'Invalid boundary expression data.')
    from ..components.prepared_data import validate_mapping
    if c['interior_data']:validate_mapping(c['interior_data'])
    require(isinstance(c['boundary_data'],dict) and set(c['boundary_data'])<=SIDES,'Invalid imported boundary mappings.')
    for source in c['boundary_data'].values():validate_mapping(source)
    require(c['network_seed'] is None or type(c['network_seed']) is int and 0<=c['network_seed']<2**31,'Invalid network seed.')
    require(c['method'] in ('baseline','hard_dirichlet','self_adaptive','lra','expert'),'Unknown PINN method.')
    require(all(number(c[key]) and c[key]>0 for key in ('sa_lr','sa_initial','sa_max')) and c['sa_initial']<=c['sa_max'] and type(c['sa_boundary']) is bool,'Invalid adaptive settings.')
    require(number(c['lra_alpha']) and 0<c['lra_alpha']<=1 and type(c['lra_interval']) is int and 1<=c['lra_interval']<=100000 and number(c['lra_epsilon']) and 0<c['lra_epsilon']<=1 and number(c['lra_max']) and 1<=c['lra_max']<=1e12,'Invalid LRA settings.')
    require(c['physics'] in ('scalar','incompressible_2d') and number(c['viscosity']) and number(c['pressure_value']) and number(c['pressure_weight']),'Invalid flow settings.')
    require(isinstance(c['flow_residuals'],(list,tuple)) and len(c['flow_residuals'])==3 and all(isinstance(text,str) and len(text)<=512 for text in c['flow_residuals']),'Invalid flow residuals.')
    require(isinstance(c['pressure_point'],(list,tuple)) and len(c['pressure_point'])==2 and all(number(v) for v in c['pressure_point']),'Invalid pressure gauge point.')
    require(isinstance(c['flow_boundary'],dict) and set(c['flow_boundary'])==SIDES,'Flow boundaries need named sets.')
    for components in c['flow_boundary'].values():
        require(isinstance(components,dict) and set(components)<={'u','v','p'},'Invalid boundary component mapping.')
        for spec in components.values():
            require(isinstance(spec,dict) and set(spec)=={'type','target','operator'} and spec['type'] in ('None','Dirichlet','Neumann','Custom') and all(isinstance(spec[key],str) and len(spec[key])<=512 for key in ('target','operator')),'Invalid component boundary settings.')
    from ..templates.predefined_pdes import PREDEFINED_PDES
    require(isinstance(c['pde_preset'],str) and (not c['pde_preset'] or c['pde_preset'] in PREDEFINED_PDES) and isinstance(c['pde_forcing'],str) and len(c['pde_forcing'])<=384,'Invalid predefined PDE settings.')
    from ..components.adaptive_sampling import validate_adaptive_sampling
    validate_adaptive_sampling(c['adaptive_sampling'])
    from ..components.losses import LOSS_FORMULATIONS
    require(isinstance(c['loss_formulation'],str) and c['loss_formulation'] in LOSS_FORMULATIONS,'Unknown loss formulation.')
    from ..components.expert_config import validate_expert_settings
    validate_expert_settings(c['expert_settings'])
    recipes = c['boundary_sampling']
    require(isinstance(recipes,dict) and set(recipes) <= SIDES, 'Invalid boundary sampling sets.')
    parsed = {}
    for side, recipe in recipes.items():
        require(isinstance(recipe,dict) and set(recipe) == {'points','method','seed'}, 'Invalid boundary recipe.')
        require(recipe['method'] in SAMPLERS and type(recipe['points']) is int and 1 <= recipe['points'] <= 10000 and type(recipe['seed']) is int and 0 <= recipe['seed'] < 2**31, 'Unsupported boundary recipe settings.')
        parsed[side] = BoundarySampling(**recipe)
    values = dict(c, layers=tuple(c['layers']), boundary_sampling=parsed,flow_residuals=tuple(c['flow_residuals']),pressure_point=tuple(c['pressure_point']))
    config = PoissonConfig(**values)
    nodes, wires = deepcopy(data['nodes']), deepcopy(data['wires'])
    require(isinstance(nodes,list) and len(nodes) <= 2000, 'Invalid block list (maximum 2000).')
    require(isinstance(wires,list) and len(wires) <= 10000, 'Invalid wire list (maximum 10000).')
    by_id = {}
    for node in nodes:
        require(isinstance(node,dict) and set(node) == {'id','kind','title','position','settings'}, 'Invalid block record.')
        identifier = node['id']
        require(isinstance(identifier,str) and 0 < len(identifier) <= 128 and identifier not in by_id, 'Block identifiers must be unique non-empty strings.')
        require(isinstance(node['kind'],str) and node['kind'] in INPUTS, 'Unknown block kind.')
        require(isinstance(node['title'],str) and 0 < len(node['title'].strip()) <= 256, 'Invalid block name.')
        require(isinstance(node['position'],list) and len(node['position']) == 2 and all(number(v) and abs(v) <= 1e9 for v in node['position']), 'Invalid block position.')
        settings = node['settings']
        require(isinstance(settings,dict) and {'resolution','history_limit'} <= set(settings) <= {'resolution','history_limit','reference_expression','reference_data','parameters'}, 'Invalid output settings.')
        require(type(settings['resolution']) is int and 2 <= settings['resolution'] <= 201 and type(settings['history_limit']) is int and 25 <= settings['history_limit'] <= 100000, 'Unsupported output settings.')
        require(isinstance(settings.get('reference_expression',''),str) and len(settings.get('reference_expression',''))<=512,'Invalid reference expression.')
        if settings.get('reference_data'):validate_mapping(settings['reference_data'])
        from ..recipes import COMPACT_RECIPES
        if data['version']<10 and node['kind'] in ('network',*COMPACT_RECIPES) and isinstance(settings.get('parameters'),dict):
            parameters=dict(settings['parameters'])
            parameters.setdefault('network_type','fcnn');parameters.setdefault('network_options',{})
            node['settings']=settings=dict(settings,parameters=parameters)
        from ..recipes import TRAINER_RECIPES
        if data['version']<14 and node['kind'] in (*COMPACT_RECIPES,*TRAINER_RECIPES) and isinstance(settings.get('parameters'),dict):
            parameters=dict(settings['parameters']);parameters.setdefault('loss_formulation','mean');node['settings']=settings=dict(settings,parameters=parameters)
        if node['kind'] in COMPACT_RECIPES:
            require('parameters' in settings,'Recipe blocks require component settings.')
            COMPACT_RECIPES[node['kind']].split(settings['parameters'])
            require(type(settings['parameters']['network_seed']) is int,'Recipe network requires an explicit seed.')
        if node['kind']=='expert_method':require(isinstance(settings.get('parameters'),dict) and settings['parameters'].get('method')=='expert','Invalid Expert trainer settings.')
        if node['kind']=='lra_method':require(isinstance(settings.get('parameters'),dict) and settings['parameters'].get('method')=='lra','Invalid LRA trainer settings.')
        if node['kind']=='sa_method':require(isinstance(settings.get('parameters'),dict) and settings['parameters'].get('method')=='self_adaptive','Invalid SA trainer settings.')
        if 'parameters' in settings:
            from .graph_model import PARAMETERS
            parameters=settings['parameters']
            if data['version']<6 and isinstance(parameters,dict):
                from .graph_model import parameters_for
                parameters={**parameters_for(node['kind'],config),**parameters};node['settings']=settings=dict(settings,parameters=parameters)
            if data['version']<5 and node['kind']=='method' and isinstance(parameters,dict):parameters=dict(parameters,method=parameters.get('method','baseline'));settings=dict(settings,parameters=parameters);node['settings']=settings
            if data['version']<13 and node['kind']=='interior' and isinstance(parameters,dict):
                parameters=dict(parameters,adaptive_sampling={});node['settings']=settings=dict(settings,parameters=parameters)
            require(isinstance(parameters,dict) and set(parameters)==set(PARAMETERS.get(node['kind'],())),'Invalid instance parameter fields.')
            if node['kind']=='method':require(parameters['method'] in ('baseline','hard_dirichlet'),'Use the self-adaptive trainer for SA-PINN settings.')
            if node['kind'] in ('rar','importance'):
                validate_adaptive_sampling(parameters['adaptive_sampling'])
                require(parameters['adaptive_sampling'].get('method')==node['kind'],'Adaptive sampler block/method mismatch.')
            if node['kind']=='interior':require(parameters['adaptive_sampling']=={},'Use a dedicated adaptive sampler block.')
            if node['kind']=='network':require(type(parameters['network_seed']) is int,'Network instances require an explicit seed.')
            trial=asdict(config);trial.update(parameters)
            decode_document({'format':'margam-model','version':16,'config':trial,'nodes':[],'wires':[]})
        by_id[identifier] = node
    occupied = set()
    for wire in wires:
        require(isinstance(wire,dict) and set(wire) == {'source','target','input'}, 'Invalid wire record.')
        require(all(isinstance(wire[k],str) for k in wire), 'Wire endpoints must be strings.')
        require(wire['source'] in by_id and wire['target'] in by_id and wire['source'] != wire['target'], 'Unknown or self-connected wire endpoint.')
        source = OUTPUTS[by_id[wire['source']]['kind']]
        targets = {spec.name:spec for spec in INPUTS[by_id[wire['target']]['kind']]}
        target = targets.get(wire['input'])
        require(source is not None and target is not None and source.data_type == target.data_type, 'Incompatible saved wire ports.')
        key = (wire['target'], wire['input'])
        require(key not in occupied, 'A saved input has multiple sources.')
        occupied.add(key)
    return config, nodes, wires


def capture_model(window):
    return {'format':'margam-model', 'version':16, 'config':asdict(window.config),
            'nodes':[{'id':n.identifier,'kind':n.kind,'title':n.title,
                      'position':[n.pos().x(),n.pos().y()],'settings':deepcopy(n.settings)}
                     for n in sorted(window.scene.nodes(),key=lambda n:n.identifier)],
            'wires':[{'source':w.source.parentItem().identifier,'target':w.target.parentItem().identifier,'input':w.target.name}
                     for w in window.scene.wires]}


def unique_keys(pairs):
    result = {}
    for key,value in pairs:
        require(key not in result, 'Duplicate JSON object key.')
        result[key] = value
    return result


def read_model(path):
    with open(path,'rb') as stream:
        content = stream.read(MAX_BYTES+1)
    require(len(content) <= MAX_BYTES, 'Model file exceeds 4 MiB.')
    try:
        data = json.loads(content, object_pairs_hook=unique_keys)
        decode_document(data)
    except (TypeError, KeyError, OverflowError, RecursionError, UnicodeError) as error:
        raise ValueError('Invalid model data.') from error
    return data


def write_model(path, data):
    """Replace only after a complete validated write beside the destination."""
    decode_document(data)
    content = (json.dumps(data,indent=2,allow_nan=False)+'\n').encode('utf-8')
    require(len(content) <= MAX_BYTES, 'Model document exceeds 4 MiB.')
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.margam-',suffix='.tmp',delete=False) as stream:
            temporary = stream.name
            stream.write(content);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None and os.path.exists(temporary):os.unlink(temporary)
