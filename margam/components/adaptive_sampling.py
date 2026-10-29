"""Residual-driven interior sampling for concrete interval/rectangle workflows.

RAR grows the active set. Importance sampling draws with replacement and uses
inverse probabilities to estimate uniform candidate-pool MSE. No Qt dependency.
"""
from copy import deepcopy
import math
from .sampling import sample_box


def sampling_defaults(method):
    if method not in ('rar','importance'):raise ValueError('Unknown adaptive sampler.')
    common=dict(method=method,interval=100,candidates=1024,chunk=256)
    return dict(common,batch=16,maximum=10000) if method=='rar' else dict(common,power=2.,mixture=.9)


def validate_adaptive_sampling(settings):
    if not isinstance(settings,dict):raise ValueError('Adaptive sampling settings must be a dictionary.')
    if not settings:return
    method=settings.get('method')
    expected=sampling_defaults(method)
    if set(settings)!=set(expected):raise ValueError('Incomplete adaptive sampling settings.')
    for key,limit in (('interval',100000),('candidates',100000),('batch',100000),('maximum',100000),('chunk',4096)):
        if key not in settings:continue
        if type(settings[key]) is not int or not 1<=settings[key]<=limit:raise ValueError(f'Adaptive sampling {key} must be between 1 and {limit}.')
    for key,low,high in (('power',.1,4.),('mixture',0.,.99)):
        if key not in settings:continue
        if type(settings[key]) not in (int,float) or not math.isfinite(settings[key]) or not low<=settings[key]<=high:raise ValueError(f'Invalid adaptive sampling {key}.')
    if method=='rar' and settings['batch']>settings['candidates']:raise ValueError('RAR additions cannot exceed the candidate pool.')


def importance_probabilities(scores,power,mixture):
    """Positive probability mixture; uniform when every residual is zero."""
    import torch
    maximum=scores.max().item()
    scaled=(scores/maximum).pow(power) if maximum else torch.ones_like(scores)
    return (1-mixture)/len(scores)+mixture*scaled/scaled.sum()


class AdaptiveSampler:
    def __init__(self,model,settings,bounds,seed):
        import torch
        validate_adaptive_sampling(settings)
        if not settings:raise ValueError('Choose an adaptive sampling method.')
        if hasattr(model,'adaptive_weights'):raise ValueError('Adaptive sampling cannot yet move SA-PINN pointwise masks. Use a baseline, hard-BC or LRA trainer.')
        if not callable(model.problem.residual_target) and torch.as_tensor(model.problem.residual_target).ndim!=0:
            raise ValueError('Adaptive sampling needs a scalar or coordinate-callable residual target.')
        if settings['method']=='importance' and not isinstance(model.metric,torch.nn.MSELoss):
            raise ValueError('Importance sampling currently requires MSE.')
        lower,upper=bounds
        sample_box(lower,upper,1,'random',seed)
        if type(seed) is not int or not 0<=seed<2**31:raise ValueError('Invalid adaptive sampler seed.')
        if settings['method']=='rar' and len(model.collocation_point_sample)>settings['maximum']:
            raise ValueError('RAR maximum must cover the initial interior set.')
        initial=model.collocation_point_sample
        if initial.shape[1]!=len(lower) or not torch.isfinite(initial).all() or ((initial<initial.new_tensor(lower))|(initial>initial.new_tensor(upper))).any():raise ValueError('Adaptive initial points must lie in the configured interval/rectangle.')
        self.settings=deepcopy(settings);self.bounds=(list(lower),list(upper));self.seed=seed
        self.initial_count=len(model.collocation_point_sample)
        self.boundary_anchors=None
        if getattr(model.problem,'boundary_anchor_count',0):
            self.boundary_anchors=model.collocation_point_sample[-model.problem.boundary_anchor_count:].detach().clone()
        self.initial_fingerprint=model.data_fingerprint()
        self.weights=None;self.history=[]

    def scores(self,model,points):
        import torch
        scores=[]
        # Autograd must remain enabled: residuals can contain spatial derivatives.
        with torch.enable_grad():
            for chunk in points.split(self.settings['chunk']):
                x=chunk.detach().requires_grad_(True)
                values=model.problem.residual(x,model.network(x))
                target=model._target(model.problem.residual_target,x)
                values,target=model._aligned(values,target,len(x))
                error=values-target
                if not torch.isfinite(error).all():raise FloatingPointError('Non-finite candidate PDE residual; sampling was not changed.')
                score=error.detach().to(dtype=torch.float64).square().mean(dim=1).sqrt()
                if not torch.isfinite(score).all():raise FloatingPointError('Candidate residual score overflow; sampling was not changed.')
                scores.append(score.cpu())
        return torch.cat(scores)

    def before_step(self,model):
        import torch
        s=self.settings
        # Update before the next optimiser step, after each completed interval.
        if not model.iter or model.iter%s['interval']:return
        if self.history and self.history[-1]['step']==model.iter:return
        old=model.collocation_point_sample
        if s['method']=='rar' and len(old)>=s['maximum']:return
        event_seed=(self.seed+model.iter)%2**31
        candidates=model._points(sample_box(*self.bounds,s['candidates'],'random',event_seed))
        scores=self.scores(model,candidates)
        if s['method']=='rar':
            existing={tuple(row) for row in old.detach().cpu().tolist()}
            indices=[];candidate_rows=candidates.detach().cpu().tolist()
            for index in torch.argsort(scores,descending=True,stable=True).tolist():
                row=tuple(candidate_rows[index])
                if row in existing:continue
                existing.add(row);indices.append(index)
                if len(indices)>=min(s['batch'],s['maximum']-len(old)):break
            points=torch.cat((old,candidates[indices])) if indices else old
            weights=None
        else:
            probabilities=importance_probabilities(scores,s['power'],s['mixture'])
            generator=torch.Generator(device='cpu').manual_seed(event_seed)
            count=self.initial_count-(len(self.boundary_anchors) if self.boundary_anchors is not None else 0)
            indices=torch.multinomial(probabilities,count,replacement=True,generator=generator)
            points=candidates[indices.to(candidates.device)]
            weights=(1/(len(scores)*probabilities[indices])).to(device=model.device,dtype=model.dtype).unsqueeze(1)
            if self.boundary_anchors is not None:
                points=torch.cat((points,self.boundary_anchors));weights=torch.cat((weights,weights.new_ones((len(self.boundary_anchors),1))))
        labels=model._target(model.problem.residual_target,points)
        if not torch.isfinite(labels).all():raise FloatingPointError('Non-finite adaptive PDE targets; sampling was not changed.')
        model.collocation_point_sample=points.detach();model.collocation_point_labels=labels
        self.weights=weights
        self.history.append(dict(step=model.iter,points=len(points),mean_residual=scores.mean().item(),max_residual=scores.max().item()))

    def state(self,model):
        return deepcopy(dict(settings=self.settings,bounds=self.bounds,seed=self.seed,initial_count=self.initial_count,
                             points=model.collocation_point_sample,weights=self.weights,history=self.history))

    def validate_state(self,model,payload,steps):
        import torch
        keys={'settings','bounds','seed','initial_count','points','weights','history'}
        if not isinstance(payload,dict) or set(payload)!=keys or payload['settings']!=self.settings or payload['bounds']!=self.bounds or payload['seed']!=self.seed or payload['initial_count']!=self.initial_count:
            raise ValueError('Adaptive sampling checkpoint settings differ from this recipe.')
        points=payload['points'];weights=payload['weights'];history=payload['history']
        if not isinstance(points,torch.Tensor) or points.ndim!=2 or points.shape[1]!=len(self.bounds[0]) or not 1<=len(points)<=100000 or points.dtype!=model.dtype or not torch.isfinite(points).all():raise ValueError('Invalid adaptive checkpoint coordinates.')
        if ((points<points.new_tensor(self.bounds[0]))|(points>points.new_tensor(self.bounds[1]))).any():raise ValueError('Adaptive checkpoint points lie outside the domain.')
        if not isinstance(history,list):raise ValueError('Invalid adaptive sampling history.')
        previous=0;count=self.initial_count
        for event in history:
            if not isinstance(event,dict) or set(event)!={'step','points','mean_residual','max_residual'} or type(event['step']) is not int or event['step']!=previous+self.settings['interval'] or event['step']>=steps:
                raise ValueError('Invalid adaptive sampling update phase.')
            if type(event['points']) is not int:raise ValueError('Invalid adaptive point count.')
            if self.settings['method']=='rar':
                if not count<=event['points']<=min(count+self.settings['batch'],self.settings['maximum']):raise ValueError('Invalid RAR growth history.')
            elif event['points']!=self.initial_count:raise ValueError('Invalid importance sampling budget.')
            if any(type(event[key]) not in (int,float) or not math.isfinite(event[key]) or event[key]<0 for key in ('mean_residual','max_residual')) or event['mean_residual']>event['max_residual']+1e-12:raise ValueError('Invalid residual statistics.')
            previous=event['step'];count=event['points']
        expected_events=(steps-1)//self.settings['interval'] if steps else 0
        if len(history)!=expected_events and not (self.settings['method']=='rar' and count>=self.settings['maximum']):raise ValueError('Missing adaptive sampling history.')
        if len(points)!=count:raise ValueError('Adaptive point count does not match history.')
        if self.boundary_anchors is not None:
            n=len(self.boundary_anchors);start=self.initial_count-n if self.settings['method']=='rar' else len(points)-n
            if not torch.equal(points[start:start+n].to(self.boundary_anchors.device),self.boundary_anchors):raise ValueError('Adaptive checkpoint changed fixed boundary PDE anchors.')
            if weights is not None and not torch.equal(weights[-n:],torch.ones_like(weights[-n:])):raise ValueError('Boundary PDE anchors need unit importance weights.')
        if self.settings['method']=='importance' and history:
            if not isinstance(weights,torch.Tensor) or weights.shape!=(count,1) or weights.dtype!=model.dtype or not torch.isfinite(weights).all() or (weights<=0).any() or (weights>1/(1-self.settings['mixture'])+1e-4).any():raise ValueError('Invalid importance weights.')
        elif weights is not None:raise ValueError('Unexpected importance weights.')
        return points.to(model.device).detach().clone(),None if weights is None else weights.to(model.device).detach().clone(),deepcopy(history)


def configure_adaptive_sampling(model,config):
    if config.adaptive_sampling:
        if config.pde_points=='both':
            from ..templates.poisson import sample_problem
            from .collocation import boundary_rows
            _,sets=sample_problem(config);model.problem.boundary_anchor_count=len(boundary_rows(config,sets))
        model.sampling_controller=AdaptiveSampler(model,config.adaptive_sampling,config.bounds,config.seed)
    return model
