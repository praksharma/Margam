"""Validate DSGNAR tensor/primitive checkpoints before restoring state."""
import math
from copy import deepcopy
import torch
from .dsgnar_config import validate_dsgnar_settings


def validate_state(state,settings,initial_weights,configuration=None):
    def require(ok,message='Invalid DSGNAR checkpoint.'):
        if not ok:raise ValueError(message)
    def number(value):return type(value) in (int,float) and math.isfinite(value)
    payload=state.get('dsgnar') if isinstance(state,dict) else None
    fields={'version','settings','initial_weights','weights','radius','target_ratio','generator','diagnostics','lambda_history','converged','configuration'}
    require(isinstance(payload,dict) and set(payload)==fields and payload['version']==1)
    validate_dsgnar_settings(payload['settings'])
    require(payload['settings']==settings and payload['initial_weights']==initial_weights,'DSGNAR settings/initial coefficients differ from this checkpoint.')
    if configuration is not None:require(payload['configuration']==configuration,'DSGNAR physics/configuration differs from this checkpoint.')
    require(isinstance(payload['configuration'],dict))
    names=tuple(initial_weights)
    def weights(value):return isinstance(value,dict) and set(value)==set(names) and all(number(v) and v>0 for v in value.values())
    require(weights(payload['weights']) and number(payload['radius']) and 0<payload['radius']<=settings['max_radius'])
    require(payload['target_ratio'] in (settings['target_ratio'],settings['stage2_ratio']))
    require(type(payload['converged']) is bool and payload['converged']==(payload['radius']<settings['min_radius']))
    rng=payload['generator']
    require(isinstance(rng,torch.Tensor) and rng.dtype==torch.uint8 and rng.ndim==1)
    try:torch.Generator(device='cpu').set_state(rng.cpu())
    except (RuntimeError,TypeError) as error:raise ValueError('Invalid DSGNAR sketch RNG.') from error
    count=state.get('iter');require(type(count) is int and 0<=count<=100000)
    history=payload['diagnostics'];lambdas=payload['lambda_history']
    require(isinstance(history,list) and len(history)==count and isinstance(lambdas,list) and len(lambdas)==count)
    require(all(number(v) and v>=0 for v in lambdas))
    from ..methods.dsgnar import next_ratio,reweight
    expected_weights=deepcopy(initial_weights);radius=settings['initial_radius'];ratio=settings['target_ratio']
    loss_history=state.get('loss_history');totals=state.get('training_history')
    require(isinstance(loss_history,list) and len(loss_history)==count and isinstance(totals,list) and len(totals)==count)
    for index,(row,record) in enumerate(zip(history,loss_history)):
        require(isinstance(row,dict) and set(row)=={'step','accepted','radius','next_radius','regularization','ratio','target_ratio','objective_before','objective_after','weights'})
        require(row['step']==index+1 and type(row['accepted']) is bool and weights(row['weights']) and row['weights']==expected_weights)
        require(all(number(row[key]) for key in ('radius','next_radius','regularization','ratio','target_ratio','objective_before','objective_after')))
        require(row['regularization']==lambdas[index] and row['regularization']>=0 and row['target_ratio']==ratio)
        require(0<row['radius']<=settings['max_radius'] and row['next_radius']==(row['radius'] if row['accepted'] else radius/3))
        require(row['objective_before']>=0 and row['objective_after']>=0 and (row['objective_after']<row['objective_before'] and row['ratio']>0 if row['accepted'] else row['objective_after']==row['objective_before']))
        require(isinstance(record,dict) and set(record)==set(names)|{'total'}|{'weighted:'+n for n in names if n!='pde'} and all(number(v) and v>=0 for v in record.values()))
        objective=sum(expected_weights[n]*record[n] for n in names)
        require(math.isclose(objective,row['objective_before'],rel_tol=1e-12,abs_tol=1e-30) and record['total']==totals[index]==row['objective_before'])
        require(all(math.isclose(record['weighted:'+n],expected_weights[n]*record[n],rel_tol=1e-6,abs_tol=1e-30) for n in names if n!='pde'))
        expected_weights=reweight(expected_weights,{n:expected_weights[n]*record[n] for n in names},settings['reweight_exponent'])
        radius=row['next_radius'];ratio=next_ratio(lambdas[:index+1],settings,ratio)
    require(payload['weights']==expected_weights and payload['radius']==radius and payload['target_ratio']==ratio,'DSGNAR trajectory state does not match its history.')
    return payload
