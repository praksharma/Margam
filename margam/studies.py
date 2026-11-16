"""Bounded Cartesian/paired design-point previews; no execution, sampling or file writes."""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import itertools
import json
import math
from types import SimpleNamespace

MAX_AXES=8
MAX_VALUES=64
MAX_TRIALS=512
FIELDS={'points':'Interior sample budget','boundary_points':'Default boundary sample budget',
        'seed':'Sampling seed','network_seed':'Network seed','layers':'Hidden layers',
        'iterations':'Training updates / attempts','lr':'Adam learning rate',
        'pde_weight':'PDE loss coefficient','boundary_weight':'Boundary loss coefficient',
        'network_type':'Network architecture (family defaults)'}
INTEGER={'points','boundary_points','seed','network_seed','iterations'}


def catalogue(document):
    """List actual supported block-held controls, using IDs rather than Qt objects."""
    from .ui.graph_model import PARAMETERS
    rows=[]
    for node in document['nodes']:
        parameters=node['settings'].get('parameters',{})
        for key in PARAMETERS.get(node['kind'],()):
            if key not in FIELDS or key not in parameters:continue
            if key=='iterations' and parameters.get('expert_settings',{}).get('windows',1)>1:continue
            if key=='lr' and (node['kind'].startswith('dsgnar') or parameters.get('method')=='dsgnar'):continue
            if key in ('points','seed') and (node['kind'] in ('interior_data','observations') or parameters.get('interior_data')):continue
            rows.append(dict(node_id=node['id'],parameter=key,title=node['title'],label=FIELDS[key],value=deepcopy(parameters[key])))
    return sorted(rows,key=lambda row:(row['title'],row['node_id'],row['parameter']))


def values_for(parameter,mode,text,*,allow_duplicates=False):
    if not isinstance(parameter,str) or not isinstance(mode,str) or parameter not in FIELDS or mode not in ('fixed','candidates','range'):raise ValueError('Choose a supported parameter and mode.')
    if not isinstance(text,str) or len(text)>4096:raise ValueError('Keep each value specification within 4096 characters.')
    def finite(value):
        try:return type(value) in (int,float) and math.isfinite(value)
        except OverflowError:return False
    def reject(value):raise ValueError('Use finite JSON numbers, not '+value+'.')
    try:value=json.loads(text,parse_constant=reject)
    except (json.JSONDecodeError,RecursionError) as error:raise ValueError('Enter JSON values, for example [1000,2000] or [0.001,0.003,0.001].') from error
    if parameter=='network_type':
        from .components.network_config import NETWORK_NAMES
        if mode=='range':raise ValueError('Network architecture uses Fixed/Candidates, not numeric ranges.')
        values=[value] if mode=='fixed' else value
        if not isinstance(values,list) or not 1<=len(values)<=MAX_VALUES or any(not isinstance(kind,str) or kind not in NETWORK_NAMES for kind in values):raise ValueError('Choose supported architecture keys: '+', '.join(NETWORK_NAMES))
        if not allow_duplicates and len(set(values))!=len(values):raise ValueError('Remove duplicate architecture choices.')
        return list(values)
    if mode=='range':
        if parameter=='layers':raise ValueError('Use fixed/candidate lists of hidden widths; ranges are scalar only.')
        if not isinstance(value,list) or len(value)!=3 or any(not finite(item) for item in value):
            raise ValueError('A range is [start, stop, step], with finite numbers and an inclusive reachable stop.')
        try:
            start,stop,step=map(lambda item:Decimal(str(item)),value)
            if not step or (stop-start)*step<0:raise ValueError('Range step must be nonzero and point toward stop.')
            count=int((stop-start)//step)+1
            if count>MAX_VALUES:raise ValueError('A parameter can have at most 64 values.')
            values=[int(start+i*step) if parameter in INTEGER and (start+i*step)==(start+i*step).to_integral_value() else float(start+i*step) for i in range(count)]
        except (InvalidOperation,OverflowError) as error:raise ValueError('Range numbers are too large.') from error
    elif mode=='fixed':values=[value]
    else:
        if not isinstance(value,list):raise ValueError('Candidate mode needs a JSON list.')
        values=value
    if not 1<=len(values)<=MAX_VALUES:raise ValueError('A parameter needs 1 to 64 values.')
    for item in values:
        if parameter=='layers':
            valid=isinstance(item,list) and 1<=len(item)<=16 and all(type(width) is int for width in item)
        elif parameter in INTEGER:valid=type(item) is int or parameter=='network_seed' and item is None
        else:valid=finite(item)
        if not valid:raise ValueError('Use integer values for counts/seeds, finite numeric coefficients, and lists of integers for hidden layers. Only network seed can be null.')
    if parameter not in INTEGER and parameter!='layers':values=[float(item) for item in values]
    normalized=[json.dumps(item,sort_keys=True,allow_nan=False) for item in values]
    if not allow_duplicates and len(set(normalized))!=len(normalized):raise ValueError('Remove duplicate candidate values.')
    return deepcopy(values)


def config_from_document(document,scene=None):
    """Use the editor's real structural decoder and typed graph assembler."""
    from .ui.model_document import decode_document
    from .ui.graph_model import assemble
    decode_document(document)
    nodes={record['id']:SimpleNamespace(kind=record['kind'],title=record['title'],identifier=record['id'],
        settings=deepcopy(record['settings']),scene=lambda:scene) for record in document['nodes']}
    wires=[SimpleNamespace(source=SimpleNamespace(parentItem=lambda n=nodes[row['source']]:n),
        target=SimpleNamespace(parentItem=lambda n=nodes[row['target']]:n,name=row['input'])) for row in document['wires']]
    return assemble(list(nodes.values()),wires)


def resolve_axes(document,axes,combination='cartesian'):
    """Validate and resolve the bounded finite grid without assembling trials."""
    if combination not in ('cartesian','paired'):raise ValueError('Choose Cartesian or paired combinations.')
    if not isinstance(axes,list) or not 1<=len(axes)<=MAX_AXES:raise ValueError('Choose 1 to 8 parameter rows.')
    available={(row['node_id'],row['parameter']):row for row in catalogue(document)}
    choices=[];seen=set();total=1
    for axis in axes:
        if not isinstance(axis,dict) or set(axis)!={'node_id','parameter','mode','text'}:raise ValueError('Invalid design parameter row.')
        if not isinstance(axis['node_id'],str) or not isinstance(axis['parameter'],str):raise ValueError('Block IDs and parameter names must be strings.')
        key=(axis['node_id'],axis['parameter'])
        if key not in available:raise ValueError('A chosen block/parameter no longer exists or is not supported. Choose it again.')
        if key in seen:raise ValueError('Each block parameter can appear only once.')
        seen.add(key);values=values_for(axis['parameter'],axis['mode'],axis['text'],allow_duplicates=combination=='paired');total*=len(values)
        if combination=='cartesian' and total>MAX_TRIALS:raise ValueError('Preview is limited to 512 design points. Reduce candidate lists/ranges.')
        choices.append((available[key],values))
    if combination=='paired':
        lengths={len(values) for _,values in choices if len(values)>1}
        if len(lengths)>1:raise ValueError('Paired candidate/range lists must have matching lengths; single values repeat across rows.')
        total=max((len(values) for _,values in choices),default=1)
    return choices,total


def combinations(choices,mode='cartesian'):
    if mode=='cartesian':return itertools.product(*(values for _,values in choices))
    if mode!='paired':raise ValueError('Choose Cartesian or paired combinations.')
    count=max((len(values) for _,values in choices),default=1)
    return (tuple(values[0] if len(values)==1 else values[index] for _,values in choices) for index in range(count))


def architecture_defaults(document,axes,existing=None,combination='cartesian'):
    """Freeze only the families actually selected, preserving opened snapshots."""
    from .components.network_config import network_options
    choices,_=resolve_axes(document,axes,combination);result={}
    for axis,values in choices:
        if axis['parameter']!='network_type':continue
        for kind in values:
            options=deepcopy(existing[kind]) if existing is not None and kind in existing else network_options(kind,{})
            network_options(kind,options) # Validate without replacing frozen values.
            result[kind]=options
    return result


def apply_choices(document,choices,combination,*,network_defaults=None):
    """One materialization rule shared by preview and portable/native execution.

    An explicit family axis always selects that family's defaults, even when it
    equals the base family. Without this axis, custom base options are untouched.
    """
    from .components.network_config import network_options
    candidate=deepcopy(document);nodes={node['id']:node for node in candidate['nodes']};selected=[];networks=[]
    for (axis,_),value in zip(choices,combination):
        parameters=nodes[axis['node_id']]['settings']['parameters'];parameters[axis['parameter']]=deepcopy(value)
        selected.append(dict(node_id=axis['node_id'],parameter=axis['parameter'],value=deepcopy(value)))
        if axis['parameter']=='network_type':
            parameters['network_options']=deepcopy(network_defaults[value]) if network_defaults is not None and value in network_defaults else network_options(value,{})
            networks.append(dict(node_id=axis['node_id'],architecture=value,options=deepcopy(parameters['network_options'])))
    return candidate,selected,networks


def preview(document,axes,*,scene=None,network_defaults=None,combination='cartesian'):
    """Enumerate bounded Cartesian/paired choices; invalid trials retain a reason, no metric.

    Config validation is structural/numerical preflight, not execution/data/output
    qualification. This function never constructs networks or samples points.
    """
    from .components.resources import architecture_size
    choices,total=resolve_axes(document,axes,combination)
    defaults=architecture_defaults(document,axes,network_defaults,combination)
    trials=[]
    for index,values in enumerate(combinations(choices,combination),1):
        candidate,settings,networks=apply_choices(document,choices,values,network_defaults=defaults)
        row=dict(design_point=f'DP{index:04d}',choices=settings,status='Invalid',reason='',updates=None,parameters=None,networks=networks)
        try:
            config=config_from_document(candidate,scene);config.validate()
            count=architecture_size(config)['parameters']
            updates=config.expert_settings['windows']*config.expert_settings['window_steps'] if config.method=='expert' and config.is_time_dependent and config.expert_settings['windows']>1 else config.iterations
            row.update(status='Ready for execution validation',updates=updates,parameters=count)
        except (ValueError,TypeError,KeyError,OSError,OverflowError) as error:row['reason']=str(error)
        trials.append(row)
    valid=[row for row in trials if row['updates'] is not None]
    return dict(trials=trials,total=total,valid=len(valid),invalid=total-len(valid),
        total_updates=sum(row['updates'] for row in valid),maximum_parameters=max((row['parameters'] for row in valid),default=0))
