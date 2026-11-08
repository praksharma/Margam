"""Explicit research checkpoints for fixed data and stateless loss code."""
from copy import deepcopy
import json
import math
import torch


def canonical(value):
    return json.dumps(value,sort_keys=True,allow_nan=False)


def signature(trainer):
    trainer._require_compiled()
    from .trainer import Trainer
    if trainer.checkpoint_id is None:
        raise NotImplementedError('Set a versioned checkpoint_id identifying your physics, network and loss code before checkpointing.')
    if trainer.step is not None or type(trainer).update is not Trainer.update or type(trainer).training_step is not Trainer.training_step:
        raise NotImplementedError('Custom update code must implement its own complete checkpoint state, including extra optimizers/schedules.')
    if trainer.sampling_controller is not None:
        raise NotImplementedError('Research checkpoints currently require fixed point sets.')
    if type(trainer.optimizer) not in (torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD):
        raise NotImplementedError('Research checkpoints currently support Adam, AdamW and SGD; custom optimizers own their state contract.')
    losses=trainer.loss.values() if isinstance(trainer.loss,dict) else (trainer.loss,)
    if any(isinstance(loss,torch.nn.Module) and loss.state_dict() for loss in losses):
        raise NotImplementedError('Stateful loss modules require their own checkpoint implementation.')
    if any(not isinstance(value,torch.Tensor) for value in trainer.network.state_dict().values()):
        raise NotImplementedError('Research checkpoints require a tensor-only network state_dict.')
    trainer.check_schedule()
    groups=[{k:v for k,v in group.items() if k!='params'} for group in trainer.optimizer.state_dict()['param_groups']]
    value=dict(checkpoint_id=trainer.checkpoint_id,
        trainer_class=type(trainer).__module__+'.'+type(trainer).__qualname__,
        network_class=type(trainer.network).__module__+'.'+type(trainer.network).__qualname__,
        optimizer_class=type(trainer.optimizer).__module__+'.'+type(trainer.optimizer).__qualname__,
        groups=groups,weights=trainer.weights,loss_formulation=trainer.loss_formulation,
        measures=trainer.loss_measures,data_fingerprint=trainer.data_fingerprint(),
        device=str(trainer.collocation_point_sample.device))
    if trainer.schedule is not None:
        value['schedule']=dict(kind='step_decay',every=trainer.schedule.every,gamma=trainer.schedule.gamma,base_lrs=list(trainer._base_lrs))
        for group,rate in zip(groups,trainer._base_lrs):group['lr']=rate
    # Capture metadata as JSON primitives, never executable objects.
    return json.loads(canonical(value))


def rng_state(trainer):
    return dict(cpu=torch.get_rng_state().clone(),
        cuda=torch.cuda.get_rng_state(trainer.device).cpu().clone() if trainer.device.type=='cuda' else None)


def set_rng(trainer,state):
    torch.set_rng_state(state['cpu'].cpu())
    if state['cuda'] is not None:torch.cuda.set_rng_state(state['cuda'].cpu(),trainer.device)


def capture(trainer):
    return deepcopy(dict(format='margam-research-training',version=2 if trainer.schedule is not None else 1,signature=signature(trainer),
        network=trainer.network.state_dict(),optimizer=trainer.optimizer.state_dict(),
        iter=trainer.iter,loss_history=trainer.loss_history,training_history=trainer.training_history,
        rng=rng_state(trainer)))


def require(value,message):
    if not value:raise ValueError(message)


def validate_optimizer(trainer,state,steps):
    require(isinstance(state,dict) and set(state)=={'state','param_groups'},'Invalid optimizer state.')
    current=dict(param_groups=deepcopy(trainer.optimizer.state_dict()['param_groups']));groups=state['param_groups']
    if trainer.schedule is not None:
        for group,rate in zip(current['param_groups'],trainer.scheduled_rates(steps)):group['lr']=rate
    require(isinstance(groups,list) and len(groups)==len(current['param_groups']),'Optimizer parameter groups differ.')
    for a,b in zip(groups,current['param_groups']):
        require(isinstance(a,dict) and set(a)==set(b) and canonical(a)==canonical(b),'Optimizer settings or parameter identities differ.')
    ids=[key for group in groups for key in group['params']]
    parameters=[p for group in trainer.optimizer.param_groups for p in group['params']]
    require(len(ids)==len(set(ids)) and isinstance(state['state'],dict) and set(state['state'])<=set(ids),'Invalid optimizer parameter IDs.')
    expected=dict(zip(ids,parameters));is_sgd=type(trainer.optimizer) is torch.optim.SGD
    for key,values in state['state'].items():
        require(isinstance(values,dict),'Invalid optimizer parameter state.')
        allowed={'momentum_buffer'} if is_sgd else {'step','exp_avg','exp_avg_sq','max_exp_avg_sq'}
        required={'momentum_buffer'} if is_sgd else {'step','exp_avg','exp_avg_sq'}
        require(required<=set(values)<=allowed,'Invalid optimizer state fields.')
        for name,value in values.items():
            require(isinstance(value,torch.Tensor) and torch.isfinite(value).all().item(),'Non-finite or invalid optimizer tensor.')
            if name=='step':
                require(value.numel()==1 and float(value.item()).is_integer() and 0<=value.item()<=steps,'Invalid optimizer step count.')
            else:
                require(value.shape==expected[key].shape and value.dtype==expected[key].dtype,'Optimizer tensor shape/dtype mismatch.')


def restore(trainer,state):
    expected_signature=signature(trainer)
    keys={'format','version','signature','network','optimizer','iter','loss_history','training_history','rng'}
    require(isinstance(state,dict) and set(state)==keys and state['format']=='margam-research-training' and type(state['version']) is int and state['version']==(2 if trainer.schedule is not None else 1),'Invalid research checkpoint.')
    require(canonical(state['signature'])==canonical(expected_signature),'Research checkpoint identity, data, objective, device or optimizer settings differ.')
    steps=state['iter'];history=state['loss_history'];totals=state['training_history']
    require(type(steps) is int and steps>=0 and isinstance(history,list) and isinstance(totals,list) and len(history)==len(totals)==steps,'Checkpoint history/step count mismatch.')
    required={'pde','total',*(c.name for c,_,_ in trainer._constraints),*('weighted:'+c.name for c,_,_ in trainer._constraints)}
    names=set(history[0]) if history and isinstance(history[0],dict) else required
    require(required<=names,'Missing research loss history terms.')
    for record,total in zip(history,totals):
        require(isinstance(record,dict) and set(record)==names and all(isinstance(k,str) for k in record),'Inconsistent research loss history.')
        require(all(type(v) in (float,int) and math.isfinite(v) for v in (*record.values(),total)) and total==record['total'],'Invalid research loss history values.')
    if trainer.schedule is not None:
        for step,record in enumerate(history):
            for index,rate in enumerate(trainer.scheduled_rates(step)):
                require(type(record.get(f'learning_rate:{index}')) in (int,float) and record[f'learning_rate:{index}']==rate,'Scheduled learning-rate history differs.')
    network=state['network'];expected=trainer.network.state_dict()
    require(isinstance(network,dict) and set(network)==set(expected),'Network state fields differ.')
    for key,value in network.items():
        require(isinstance(value,torch.Tensor) and value.shape==expected[key].shape and value.dtype==expected[key].dtype and torch.isfinite(value).all().item(),'Network shape/dtype or finite-value mismatch.')
    validate_optimizer(trainer,state['optimizer'],steps)
    rng=state['rng'];current_rng=rng_state(trainer)
    require(isinstance(rng,dict) and set(rng)=={'cpu','cuda'},'Invalid Torch RNG state.')
    for key,value in rng.items():
        expected=current_rng[key]
        require(value is None if expected is None else isinstance(value,torch.Tensor) and value.dtype==torch.uint8 and value.shape==expected.shape,'RNG state differs from the configured device.')
    # Roll back even if a native module/optimizer rejects an otherwise valid state.
    old_network=deepcopy(trainer.network.state_dict());old_optimizer=deepcopy(trainer.optimizer.state_dict())
    try:
        trainer.network.load_state_dict(network);trainer.optimizer.load_state_dict(deepcopy(state['optimizer']));set_rng(trainer,rng)
    except Exception:
        trainer.network.load_state_dict(old_network);trainer.optimizer.load_state_dict(old_optimizer);set_rng(trainer,current_rng)
        raise
    trainer.iter=steps;trainer.loss_history=deepcopy(history);trainer.training_history=list(totals)
    return trainer
