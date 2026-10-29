"""Independent Scale-PINN equations (4), (6), (7) and Algorithm 1; no JAX source."""
from copy import deepcopy
from dataclasses import asdict,replace
import math
import torch
from .pinn import PINN
from ..components.scale_config import validate_scale_problem


from ..components.differential import derivative


def spatial_laplacian(x,values,axes):
    return torch.cat([sum((derivative(derivative(values[:,i:i+1],x,axis),x,axis) for axis in axes),x.new_zeros((len(x),1))) for i in range(values.shape[1])],1)


class ScalePINN(PINN):
    def __init__(self,problem,network,*,config):
        super().__init__(problem,network,weights=config.loss_weights)
        self.config=deepcopy(config);self.options=deepcopy(config.scale_settings);self._scale_fingerprint=None

    def compile(self,optimiser_string='adam',lr=.001,metrics_string='MSE',device='auto',*,optimiser=None):
        validate_scale_problem(self.config)
        if optimiser_string!='adam' or metrics_string!='MSE' or optimiser is not None:raise ValueError('Scale-PINN currently uses Adam/MSE.')
        super().compile(optimiser_string,lr,metrics_string,device)
        self.previous=deepcopy(self.network).eval()
        for parameter in self.previous.parameters():parameter.requires_grad_(False)
        configuration=asdict(self.config)
        for key in ('iterations','device'):configuration.pop(key)
        self.signature=dict(algorithm='scale-pinn-v1-sequential-correction',configuration=configuration,lr=lr,weights=deepcopy(self.weights))
        self.diagnostics=[];self._fresh_samples(0)
        # Canonical CPU target evaluation makes archived data identity device-independent.
        sample,labels,constraints=self.collocation_point_sample,self.collocation_point_labels,self._constraints
        def target(source,points):return torch.as_tensor(source(points) if callable(source) else source,dtype=self.dtype,device='cpu').detach()
        try:
            self.collocation_point_sample=sample.detach().cpu();self.collocation_point_labels=target(self.problem.residual_target,self.collocation_point_sample)
            self._constraints=[(condition,points.detach().cpu(),target(condition.target,points.detach().cpu())) for condition,points,_ in constraints]
            self._scale_fingerprint=super().data_fingerprint()
        finally:self.collocation_point_sample,self.collocation_point_labels,self._constraints=sample,labels,constraints
        return self

    def data_fingerprint(self):return self._scale_fingerprint if self._scale_fingerprint is not None else super().data_fingerprint()

    def _fresh_samples(self,phase):
        if not self.options['fresh']:return
        if hasattr(self.problem,'refresh_samples'):points,conditions=self.problem.refresh_samples(phase)
        else:
            from ..templates.poisson import sample_problem
            c=deepcopy(self.config);c.seed=(c.seed+104729*phase)%2**31
            c.boundary_sampling={side:replace(c.boundary_recipe(side),seed=(c.boundary_recipe(side).seed+104729*phase)%2**31) for side in c.sides}
            points,sets=sample_problem(c);by_name={c.constraint_name(side):rows for side,rows in sets.items()}
            conditions=[]
            for condition,_,_ in self._constraints:
                rows=by_name.get(condition.name,by_name.get(condition.name.split(':')[0]))
                conditions.append(replace(condition,points=rows) if rows is not None else condition)
        self.collocation_point_sample=self._points(points);self.collocation_point_labels=self._target(self.problem.residual_target,self.collocation_point_sample)
        self._constraints=[(condition,self._points(condition.points),self._target(condition.target,self._points(condition.points))) for condition in conditions]

    def correction(self,x,values):
        if self.options['strength']==0:return torch.zeros_like(values)
        axes=[i for i,name in enumerate(self.config.coordinate_names) if name!='t']
        previous=self.previous(x)
        if self.config.field_names==('u','v','p'):
            # Momentum receives velocity smoothing; continuity receives pressure change only.
            current=torch.cat((values[:,:2]/self.options['tau_sc']-self.options['gamma']/self.options['tau_alpha']*spatial_laplacian(x,values[:,:2],axes),values[:,2:3]/self.options['tau_sc']),1)
            old=torch.cat((previous[:,:2]/self.options['tau_sc']-self.options['gamma']/self.options['tau_alpha']*spatial_laplacian(x,previous[:,:2],axes),previous[:,2:3]/self.options['tau_sc']),1)
        else:
            current=values/self.options['tau_sc']-self.options['gamma']/self.options['tau_alpha']*spatial_laplacian(x,values,axes)
            old=previous/self.options['tau_sc']-self.options['gamma']/self.options['tau_alpha']*spatial_laplacian(x,previous,axes)
        return self.options['strength']*(current-old.detach())

    def compute_losses(self):
        # Original BC/IC operators and weights remain untouched.
        if self.options['strength']==0:
            terms=super().compute_losses();value=terms['pde'].detach().item();self._latest=dict(physical_pde=value,corrected_pde=value,correction_rms=0.);return terms
        samples=self.loss_samples();terms={name:self._term(*sample) for name,sample in samples.items() if name!='pde'}
        x=self.collocation_point_sample.detach().requires_grad_(True);values=self.network(x)
        physical=self.problem.residual(x,values);correction=self.correction(x,values)
        if physical.shape!=correction.shape:raise ValueError('Scale correction and physical residual components must match.')
        corrected,target=self._aligned(physical+correction,self.collocation_point_labels,len(x))
        physical,target_physical=self._aligned(physical,self.collocation_point_labels,len(x))
        # Shared component-mean MSE; a flow example's PDE weight 3 gives the paper sum.
        terms={'pde':self._term(corrected,target,len(x)),**terms}
        self._latest=dict(physical_pde=(physical-target_physical).square().mean().detach().item(),corrected_pde=terms['pde'].detach().item(),correction_rms=correction.square().mean().sqrt().detach().item())
        return terms

    def train(self,iterations,display_every=1):
        self._require_compiled()
        if type(iterations) is not int or iterations<0 or type(display_every) is not int or display_every<0:raise ValueError('Invalid Scale training update/display count.')
        self.network.train()
        for _ in range(iterations):
            self._fresh_samples(self.iter);self.optimiser.zero_grad(set_to_none=True);terms=self.compute_losses()
            total=sum(value*self.weights.get(name,1.) for name,value in terms.items())
            if not torch.isfinite(total):raise FloatingPointError('Non-finite Scale loss; no update was taken.')
            total.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in self.network.parameters()):raise FloatingPointError('Non-finite Scale gradient; no update was taken.')
            # Store w^k BEFORE the optimizer produces w^(k+1), never an EMA or initial solution.
            self.previous.load_state_dict(self.network.state_dict());self.optimiser.step();self.iter+=1
            record={name:value.detach().item() for name,value in terms.items()};record['total']=total.detach().item()
            self.loss_history.append(record);self.training_history.append(record['total']);self.diagnostics.append(dict(step=self.iter,**self._latest))
            self.PDE_loss=terms['pde'].detach();self.BC_loss=sum(value.detach() for name,value in terms.items() if name!='pde');self.total_loss=total.detach()
            if display_every and self.iter%display_every==0:print(f'Scale update {self.iter}: {record["total"]:.6g}')
        return self.training_history

    def scale_summary(self):return deepcopy(self.diagnostics[-1]) if self.diagnostics else dict(step=0,physical_pde=None,corrected_pde=None,correction_rms=None)

    def training_state(self):
        state=super().training_state();state['scale']=deepcopy(dict(settings=self.signature,previous=self.previous.state_dict(),diagnostics=self.diagnostics));return state

    def load_training_state(self,state):
        payload=state.get('scale') if isinstance(state,dict) else None
        if isinstance(payload,dict) and isinstance(payload.get('settings'),dict) and isinstance(payload['settings'].get('configuration'),dict):
            from ..components.gpinn_config import gpinn_defaults
            payload=deepcopy(payload);payload['settings']['configuration'].setdefault('gpinn_weights',gpinn_defaults())
        if not isinstance(payload,dict) or set(payload)!={'settings','previous','diagnostics'} or payload['settings']!=self.signature:raise ValueError('Scale-PINN checkpoint configuration differs.')
        expected=self.previous.state_dict();previous=payload['previous'];records=payload['diagnostics'];steps=state.get('iter')
        if not isinstance(previous,dict) or set(previous)!=set(expected) or any(not isinstance(value,torch.Tensor) or value.shape!=expected[name].shape or value.dtype!=expected[name].dtype or not torch.isfinite(value).all() for name,value in previous.items()):raise ValueError('Invalid Scale previous-iterate network.')
        if type(steps) is not int or not isinstance(records,list) or len(records)!=steps or any(not isinstance(row,dict) or set(row)!={'step','physical_pde','corrected_pde','correction_rms'} or row['step']!=i+1 or any(type(row[key]) not in (int,float) or not math.isfinite(row[key]) or row[key]<0 for key in ('physical_pde','corrected_pde','correction_rms')) for i,row in enumerate(records)):raise ValueError('Invalid Scale diagnostic history.')
        super().load_training_state({key:value for key,value in state.items() if key!='scale'})
        self.previous.load_state_dict(previous);self.diagnostics=deepcopy(records);self._fresh_samples(max(self.iter-1,0));return self
