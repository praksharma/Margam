"""Shared executable-graph and output contracts for desktop and headless jobs."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from ..ui.ports import INPUTS,OUTPUTS,compatible
from ..ui.graph_model import assemble,role
from ..components.outputs import OutputRequest,ReferenceSource
from ..components.output_settings import OUTPUT_KINDS,output_enabled
from ..components.expressions import Expression
from ..components.prepared_data import reference_values
from ..templates.poisson import make_poisson_reference

CORE_KINDS={'domain','interior','boundary','pde','bc','network','method'}


class ExecutableGraph:
    def connection_error(self, source, target, ignore=None):
        if source is None or target is None:
            return 'Choose an output and a compatible input.'
        if source.scene() is not self or target.scene() is not self:
            return 'Both ports must belong to the current model.'
        if not source.output or target.output:
            return 'Connect an output to an input.'
        if source.parentItem() is target.parentItem():
            return 'A block cannot connect to itself.'
        if not compatible(source.data_type,target.data_type):
            return f'{source.label} cannot feed {target.label}. Choose a compatible input.'
        for wire in self.wires:
            if wire is ignore:continue
            if wire.target is target:
                return f'{target.label} already has a source. Delete its wire before reconnecting.'
        return None

    def validate_graph(self):
        nodes = self.nodes()
        if any('enabled' in n.settings and n.kind not in OUTPUT_KINDS for n in nodes):raise ValueError('Only optional result blocks support an enabled flag; required physics cannot be disabled.')
        if any(n.kind in ('inverse_recipe','inverse_method') for n in nodes):
            for wire in self.wires:
                error=self.connection_error(wire.source,wire.target,ignore=wire)
                if error:raise ValueError(error)
            assemble(nodes,self.wires);return
        # The executable dependencies determine which point sources are required.
        destinations=set()
        for wire in self.wires:
            error=self.connection_error(wire.source,wire.target,ignore=wire)
            if error:raise ValueError(error)
            source,target = wire.source.parentItem(),wire.target.parentItem()
            if role(source.kind) in CORE_KINDS and role(target.kind) in CORE_KINDS:
                allowed_edges={('domain','interior'),('domain','boundary'),('interior','pde'),('boundary','pde'),('boundary','bc'),('pde','method'),('bc','method'),('network','method')}
                if (role(source.kind),role(target.kind)) not in allowed_edges:raise ValueError('Unsupported core connection.')
            else:
                allowed = role(source.kind) == 'method' and target.kind in OUTPUT_KINDS
                allowed = allowed and wire.target.name == ('model' if target.kind=='comparison' else 'input')
                allowed = allowed or (source.kind in ('reference','reference_data') and target.kind=='comparison' and wire.target.name=='reference')
                allowed = allowed or (source.kind=='evaluation_points' and target.kind in ('solution','comparison','export','paraview') and wire.target.name=='points')
                allowed = allowed or (source.kind=='comparison' and target.kind in ('export','paraview') and wire.target.name=='comparison')
                if not allowed:raise ValueError('Unsupported output connection: '+source.title+' → '+target.title)
                key=(target.identifier,wire.target.name)
                if key in destinations:raise ValueError('Each output input accepts only one source.')
                destinations.add(key)
        for kind in ('domain','boundary','pde','bc','method'):
            if sum(role(node.kind)==kind for node in nodes)!=1:raise ValueError(f'The graph needs exactly one {kind} block.')
        try:assemble(nodes,self.wires)
        except ValueError as error:raise ValueError('Missing connections or invalid graph: '+str(error)) from error

    def compile_config(self):
        self.validate_graph()
        return assemble(self.nodes(),self.wires)

    def output_requests(self, config):
        self.validate_graph()
        from ..components.evaluation_data import read_point_data,align_reference
        from ..components.outputs import evaluation_points
        cache={};sources={}
        def resolve_source(source):
            workspace=getattr(self,'project_workspace',None)
            return workspace().resolve_source(source,create=False) if workspace is not None else deepcopy(source)
        def snapshot(node,reference=False):
            if node.identifier not in sources:sources[node.identifier]=read_point_data(resolve_source(node.settings.get('source_data',{})),config,reference=reference)
            return sources[node.identifier]
        def request(node):
            if not output_enabled(node):return None
            if node.identifier in cache:return cache[node.identifier]
            incoming={wire.target.name:wire.source.parentItem() for wire in self.wires if wire.target.parentItem() is node}
            if not incoming:return None
            if 'comparison' in incoming:
                if node.kind not in ('export','paraview') or set(incoming)!={'comparison'}:raise ValueError('Choose a trainer/points connection OR compared fields for export, not both.')
                if not output_enabled(incoming['comparison']):raise ValueError(node.title+' depends on disabled Comparison '+incoming['comparison'].title+'. Enable it or disable this export.')
                parent=request(incoming['comparison'])
                if parent is None:raise ValueError('Connect Comparison to a trained model and reference first.')
                value=OutputRequest(node.identifier,node.kind,parent.resolution,parent.reference,parent.point_data,parent.reference_data)
                cache[node.identifier]=value;return value
            model_port='model' if node.kind=='comparison' else 'input'
            if model_port not in incoming:raise ValueError(node.title+' needs a trained-model connection.')
            point_data=snapshot(incoming['points']) if 'points' in incoming else None
            reference=None;reference_data=None
            if node.kind=='comparison':
                if 'reference' not in incoming:raise ValueError('Comparison needs both a trained-model connection and a Reference connection.')
                source=incoming['reference']
                if source.kind=='reference_data':reference_data=snapshot(source,True)
                else:
                    text=source.settings.get('reference_expression','');data=source.settings.get('reference_data',{})
                    if text and data:raise ValueError('Choose an expression or imported reference data, not both.')
                    if text:
                        texts=[part.strip() for part in text.split(';')]
                        if len(texts)!=len(config.field_names):raise ValueError('Provide one reference expression per field, separated by semicolons.')
                        expressions=[Expression(part,len(config.bounds[0]),target=True,fields=config.field_names,coordinates=config.coordinate_names) for part in texts]
                        def evaluate(points,expressions=expressions):
                            import torch
                            values=torch.cat([expr(torch.tensor(points,dtype=torch.float64),source=config.source,viscosity=config.viscosity) for expr in expressions],dim=1)
                            return values[:,0].tolist() if len(expressions)==1 else values.tolist()
                        reference=ReferenceSource('User reference expression',evaluate,dict(kind='expression',texts=texts))
                    elif data:
                        from functools import partial
                        reference=ReferenceSource('Imported reference fields',partial(reference_values,resolve_source(data),components=len(config.field_names)),dict(kind='mapping',source=resolve_source(data)))
                    else:reference=replace(make_poisson_reference(config),spec=dict(kind='preset'))
            if node.kind=='monitor':
                if not 25<=node.settings['history_limit']<=100000:raise ValueError('Monitor history limit must be between 25 and 100000.')
            elif point_data is None and config.shape=='Prepared2D':raise ValueError('Connect Evaluation points to '+node.title+'; a bounding-box grid is not this prepared domain.')
            elif point_data is None and node.settings['resolution']**len(config.bounds[0])>100000:raise ValueError('Reduce evaluation resolution: at most 100000 locations are supported.')
            if node.kind=='comparison' and incoming['reference'].kind=='reference' and incoming['reference'].settings.get('reference_data'):
                coordinates=point_data.points if point_data is not None else evaluation_points(config,node.settings['resolution'])
                try:reference.evaluate(coordinates)
                except ValueError as error:raise ValueError(node.title+': '+str(error)) from error
            if reference_data is not None:
                coordinates=point_data.points if point_data is not None else evaluation_points(config,node.settings['resolution'])
                align_reference(reference_data,coordinates,point_data.point_ids if point_data is not None else None,config)
            value=OutputRequest(node.identifier,node.kind,node.settings['resolution'],reference,point_data,reference_data);cache[node.identifier]=value;return value
        return [value for node in self.nodes() if node.kind in OUTPUT_KINDS if (value:=request(node)) is not None]



class DocumentGraph(ExecutableGraph):
    """Data-only nodes/ports with the same contract, not a Qt scene."""
    def __init__(self,document,workspace,device):
        from ..ui.model_document import decode_document
        _,records,connections=decode_document(document)
        self.project_workspace=lambda:workspace;self.execution_device=lambda:device
        self._nodes=[];by_id={}
        for record in records:
            node=SimpleNamespace(kind=record['kind'],title=record['title'],identifier=record['id'],settings=deepcopy(record['settings']),scene=lambda:self)
            def port(spec,output,node=node):
                return SimpleNamespace(name=spec.name,label=spec.label,data_type=spec.data_type,output=output,parentItem=lambda:node,scene=lambda:self)
            node.inputs={spec.name:port(spec,False) for spec in INPUTS[node.kind]}
            node.output=port(OUTPUTS[node.kind],True) if OUTPUTS[node.kind] else None
            self._nodes.append(node);by_id[node.identifier]=node
        self.wires=[SimpleNamespace(source=by_id[row['source']].output,target=by_id[row['target']].inputs[row['input']]) for row in connections]
    def nodes(self):return list(self._nodes)


def prepare_graph(graph):
    """The shared execution gate, including actual mapped data and outputs."""
    config=graph.compile_config();config.validate()
    if config.method=='inverse_burgers':
        from ..components.inverse_config import read_observations
        read_observations(config)
    if config.shape=='Prepared2D':
        from ..components.training_data import read_training_sets
        read_training_sets(config)
    if config.method!='inverse_burgers' and config.shape!='Prepared2D':
        from ..components.prepared_data import read_points
        if config.interior_data:read_points(config.interior_data,len(config.bounds[0]))
        for source in config.boundary_data.values():read_points(source,len(config.bounds[0]))
    return config,graph.output_requests(config)
