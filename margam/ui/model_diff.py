"""Read-only definition differences; stable block IDs, no physics or file execution."""
from copy import deepcopy
from dataclasses import dataclass
import json
from .model_document import decode_document
from .graph_model import PARAMETERS, parameters_for

ABSENT=object()


@dataclass(frozen=True)
class Difference:
    category: str
    node_id: str
    block: str
    field: str
    before: object
    after: object


def value_text(value,pretty=False):
    if value is ABSENT:return '(absent)'
    return json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2 if pretty else None)


def normalise(document):
    config,records,wires=decode_document(deepcopy(document))
    for record in records:
        if record['kind'] in PARAMETERS and 'parameters' not in record['settings']:
            record['settings']['parameters']=parameters_for(record['kind'],config)
    return {r['id']:r for r in records},{(w['target'],w['input']):w['source'] for w in wires}


def compare_models(before,after,*,include_presentation=False):
    """Compare migrated block-owned definitions, ignoring the inspector config cache.

    IDs match blocks. Independent recreations intentionally appear added/removed.
    File references are compared as declared: no import, hashing or path resolution.
    """
    left,old_wires=normalise(before);right,new_wires=normalise(after);changes=[]
    def add(category,identifier,field,a,b):
        node=right.get(identifier,left.get(identifier))
        changes.append(Difference(category,identifier,node['title'],field,a,b))
    def walk(identifier,path,a,b):
        if isinstance(a,dict) and isinstance(b,dict):
            for key in sorted(set(a)|set(b)):
                escaped=key.replace('~','~0').replace('/','~1')
                walk(identifier,path+'/'+escaped,a.get(key,ABSENT),b.get(key,ABSENT))
            return
        if a is not ABSENT and b is not ABSENT and a==b and (type(a) is type(b) or type(a) in (int,float) and type(b) in (int,float)):return
        category='Data metadata' if path.startswith('/settings/data_fingerprints') or path.startswith('/settings/jorvik') else 'Setting'
        if path in ('/title','/position','/settings/annotation_text'):category='Presentation'
        add(category,identifier,path,a,b)
    for identifier in sorted(set(left)|set(right)):
        a,b=left.get(identifier),right.get(identifier)
        if a is None or b is None:
            if (a or b)['kind']=='annotation' and not include_presentation:continue
            add('Added block' if a is None else 'Removed block',identifier,'/block',a if a is not None else ABSENT,b if b is not None else ABSENT)
            continue
        if a['kind']!=b['kind']:add('Block type',identifier,'/kind',a['kind'],b['kind'])
        a_settings=deepcopy(a['settings']);b_settings=deepcopy(b['settings'])
        if not include_presentation:
            a_settings.pop('annotation_text',None);b_settings.pop('annotation_text',None)
        walk(identifier,'/settings',a_settings,b_settings)
        if include_presentation:
            walk(identifier,'/title',a['title'],b['title']);walk(identifier,'/position',a['position'],b['position'])
    for identifier,port in sorted(set(old_wires)|set(new_wires)):
        a,b=old_wires.get((identifier,port),ABSENT),new_wires.get((identifier,port),ABSENT)
        if a==b:continue
        def endpoint(source,nodes):
            return ABSENT if source is ABSENT else dict(id=source,kind=nodes[source]['kind'],title=nodes[source]['title'])
        add('Connection',identifier,'/inputs/'+port,endpoint(a,left),endpoint(b,right))
    return changes
