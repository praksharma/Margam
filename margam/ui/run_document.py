"""Explicit results plus baseline checkpoint; restricted torch loading, CPU mapped."""
from pathlib import Path
import os
import tempfile
import zipfile
from .model_document import capture_model, decode_document, require, number

LIMIT=256*1024*1024


def validate_run(data):
    require(isinstance(data,dict) and set(data)=={'format','version','model','result','monitors','checkpoint'},'Invalid run archive.')
    require(data['format']=='margam-run' and type(data['version']) is int and data['version'] in (1,2),'Unsupported run version.')
    config,nodes,_=decode_document(data['model']);kinds={n['id']:n['kind'] for n in nodes}
    result=data['result']
    require(isinstance(result,dict) and set(result)=={'outputs','steps','cancelled','device'},'Invalid run result.')
    steps=result['steps']
    require(type(steps) is int and 0<=steps<=100000 and type(result['cancelled']) is bool and isinstance(result['device'],str),'Invalid run status.')
    require(isinstance(result['outputs'],dict),'Invalid run outputs.')
    for identifier,output in result['outputs'].items():
        require(kinds.get(identifier) in ('solution','comparison','export','paraview') and isinstance(output,dict),'Unknown run output.')
        require(set(output)>={'points','values','size','steps','reference','rmse'} and set(output)<={'points','values','size','steps','reference','rmse','reference_name','max_error','fields','rmse_by_field','max_error_by_field'},'Invalid output metadata.')
        size=output['size'];require(type(size) is int and 2<=size<=201,'Invalid output resolution.')
        points,values=output['points'],output['values']
        require(isinstance(points,list) and isinstance(values,list) and len(points)==len(values)==size**len(config.bounds[0]),'Invalid output point count.')
        vector=config.physics=='incompressible_2d'
        require(output.get('fields',['u'])==list(config.field_names),'Invalid output field contract.')
        finite_values=lambda rows:all(isinstance(row,list) and len(row)==3 and all(number(v) for v in row) for row in rows) if vector else all(number(v) for v in rows)
        require(all(isinstance(row,list) and len(row)==len(config.bounds[0]) and all(number(v) for v in row) for row in points) and finite_values(values),'Invalid output coordinates/values.')
        require(output['steps']==steps,'Output steps do not match the run.')
        reference=output['reference']
        require(reference is None or isinstance(reference,list) and len(reference)==len(values) and finite_values(reference),'Invalid reference values.')
        require(output['rmse'] is None or number(output['rmse']),'Invalid error metric.')
        if 'max_error' in output:require(number(output['max_error']),'Invalid maximum error.')
        if 'reference_name' in output:require(isinstance(output['reference_name'],str),'Invalid reference label.')
        for key in ('rmse_by_field','max_error_by_field'):
            if key in output:require(isinstance(output[key],dict) and set(output[key])=={'u','v','p'} and all(number(v) for v in output[key].values()),'Invalid component metrics.')
    monitors=data['monitors'];require(isinstance(monitors,dict),'Invalid monitor histories.')
    names={'pde','total',*config.constraint_names}
    def check_record(record):return isinstance(record,dict) and set(record)==names and all(number(v) for v in record.values())
    for identifier,records in monitors.items():
        require(kinds.get(identifier)=='monitor' and isinstance(records,list) and len(records)<=100000,'Invalid monitor records.')
        previous=0
        for row in records:
            require(isinstance(row,(tuple,list)) and len(row)==2 and type(row[0]) is int and previous<row[0]<=steps and check_record(row[1]),'Invalid monitor record.')
            previous=row[0]
    checkpoint=data['checkpoint']
    if checkpoint is not None:
        require(isinstance(checkpoint,dict) and {'network','optimiser','iter','loss_history','training_history'}<=set(checkpoint)<={'network','optimiser','data_fingerprint','iter','loss_history','training_history','adaptive','network_spec','lra'} and checkpoint['iter']==steps,'Invalid checkpoint metadata.')
        require(('lra' in checkpoint)==(config.method=='lra'),'Checkpoint LRA state/method mismatch.')
        require(('adaptive' in checkpoint)==(config.method=='self_adaptive'),'Checkpoint method/adaptive state mismatch.')
        if config.method=='self_adaptive':
            import torch
            adaptive=checkpoint['adaptive']
            require(isinstance(adaptive,dict) and set(adaptive)=={'settings','weights','optimiser'},'Invalid adaptive checkpoint.')
            adaptive_names={'pde',*(config.constraint_names if config.sa_boundary else ())}
            require(isinstance(adaptive['weights'],dict) and set(adaptive['weights'])==adaptive_names,'Invalid adaptive terms.')
            require(all(isinstance(v,torch.Tensor) and v.ndim==2 and v.shape[1]==1 and 1<=v.shape[0]<=100000 for v in adaptive['weights'].values()),'Invalid adaptive point counts.')
        require(isinstance(checkpoint['loss_history'],list) and len(checkpoint['loss_history'])==steps and all(check_record(v) for v in checkpoint['loss_history']),'Invalid checkpoint losses.')
        require(isinstance(checkpoint['training_history'],list) and len(checkpoint['training_history'])==steps and all(number(v) for v in checkpoint['training_history']),'Invalid checkpoint total losses.')
        if data['version']==2:require(isinstance(checkpoint.get('data_fingerprint'),str),'Checkpoint data fingerprint is missing.')
    return config


def capture_run(window):
    result={key:value for key,value in window.result.items() if key!='config'}
    result['outputs']={identifier:{key:value for key,value in output.items() if key!='config'} for identifier,output in result['outputs'].items()}
    checkpoint=window.last_run.model.training_state() if window.last_run is not None and window.last_run.model is not None else window.loaded_checkpoint
    return {'format':'margam-run','version':2,'model':capture_model(window),'result':result,
            'monitors':window.monitor_records,'checkpoint':checkpoint}


def save_run(path,data):
    import torch
    validate_run(data);path=Path(path);temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.margam-',delete=False) as stream:
            temporary=stream.name;torch.save(data,stream);stream.flush();os.fsync(stream.fileno())
        require(Path(temporary).stat().st_size<=LIMIT,'Run exceeds 256 MiB.')
        os.replace(temporary,path)
    finally:
        if temporary is not None and os.path.exists(temporary):os.unlink(temporary)


def load_run(path):
    import torch
    require(Path(path).stat().st_size<=LIMIT,'Run exceeds 256 MiB.')
    with zipfile.ZipFile(path) as archive:
        require(sum(entry.file_size for entry in archive.infolist())<=LIMIT,'Expanded run exceeds 256 MiB.')
    data=torch.load(path,map_location='cpu',weights_only=True)
    validate_run(data)
    if data['checkpoint'] is not None:
        # Validate the architecture/Adam payload without opening external input
        # files: archived predictions remain viewable if those files moved.
        # Actual resume still reconstructs data and verifies the original digest.
        from ..components import PINNProblem,PointConstraint
        from ..methods import PINN
        from ..nn.factory import build_network
        config=decode_document(data['model'])[0]
        dimension=len(config.bounds[0])
        network=build_network(config)
        if config.method=='hard_dirichlet':
            from ..methods.hard_dirichlet import BoxDirichletNetwork
            network=BoxDirichletNetwork(network,*config.bounds,config.bc_values['left'])
        adaptive=data['checkpoint'].get('adaptive',{}).get('weights',{})
        def dummy(name):return torch.zeros((len(adaptive[name]) if name in adaptive else 1,dimension))
        problem=PINNProblem(lambda x,u:u,dummy('pde'),[PointConstraint(name,dummy(name),0.) for name in config.constraint_names])
        if config.method=='self_adaptive':
            from ..methods.self_adaptive import SelfAdaptivePINN
            model=SelfAdaptivePINN(problem,network,weights=config.loss_weights).compile(lr=config.lr,device='cpu',adaptive_lr=config.sa_lr,initial_weight=config.sa_initial,max_weight=config.sa_max,adapt_boundary=config.sa_boundary)
        elif config.method=='lra':
            from ..methods.learning_rate_annealing import LearningRateAnnealingPINN
            model=LearningRateAnnealingPINN(problem,network,weights=config.loss_weights).compile(lr=config.lr,device='cpu',alpha=config.lra_alpha,update_interval=config.lra_interval,epsilon=config.lra_epsilon,max_weight=config.lra_max)
        else:model=PINN(problem,network).compile(lr=config.lr,device='cpu')
        if 'data_fingerprint' not in data['checkpoint']:
            require(data['version']==1 and not config.interior_data and not config.boundary_data,'Legacy archives cannot resume imported data without a fingerprint.')
            from ..templates.poisson import build_pinn
            from copy import deepcopy
            legacy=deepcopy(config);legacy.device='cpu'
            data['checkpoint']['data_fingerprint']=build_pinn(legacy).data_fingerprint()
        state=dict(data['checkpoint'],data_fingerprint=model.data_fingerprint())
        model.load_training_state(state)
    return data
