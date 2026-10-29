"""Independent residual-based attention: detached recurrence, no second optimizer."""
from copy import deepcopy
import math
import torch
from .pinn import PINN
from ..components.losses import weighted_loss
from ..components.rba_config import validate_rba_settings


class ResidualAttentionPINN(PINN):
    def __init__(self,problem,network,*,settings,weights=None,configuration=None):
        super().__init__(problem,network,weights=weights);validate_rba_settings(settings)
        self.settings=deepcopy(settings);self.configuration=deepcopy(configuration or {})
        for key in ('iterations','device'):self.configuration.pop(key,None)

    def compile(self,optimiser_string='adam',lr=.001,metrics_string='MSE',device='auto',*,optimiser=None):
        if optimiser_string!='adam' or metrics_string!='MSE' or optimiser is not None or self.loss_formulation!='mean':raise ValueError('RBA currently uses Adam and mean squared losses.')
        if any(c.name=='pde_weighted' for c in self.problem.constraints):raise ValueError('pde_weighted is reserved for RBA diagnostics.')
        self._rba_fingerprint=None
        super().compile(optimiser_string,lr,metrics_string,device)
        self.attention=torch.zeros((len(self.collocation_point_sample),1),device=self.device,dtype=self.dtype)
        self.fixed_fingerprint=super().data_fingerprint()
        saved=(self.collocation_point_sample,self.collocation_point_labels,self._constraints)
        def target(source,points):return torch.as_tensor(source(points) if callable(source) else source,dtype=self.dtype,device='cpu').detach()
        try:
            self.collocation_point_sample=saved[0].detach().cpu();self.collocation_point_labels=target(self.problem.residual_target,self.collocation_point_sample)
            self._constraints=[(c,points.detach().cpu(),target(c.target,points.detach().cpu())) for c,points,_ in saved[2]]
            self._rba_fingerprint=super().data_fingerprint()
        finally:self.collocation_point_sample,self.collocation_point_labels,self._constraints=saved
        self.signature=dict(algorithm='anagnostopoulos2024-detached-pde-attention',settings=deepcopy(self.settings),weights=deepcopy(self.weights),configuration=self.configuration)
        return self

    def data_fingerprint(self):return self._rba_fingerprint if self._rba_fingerprint is not None else super().data_fingerprint()

    def candidate_attention(self,error):
        scores=error.detach().abs()
        if scores.shape!=self.attention.shape or not torch.isfinite(scores).all():raise FloatingPointError('RBA needs finite scalar residuals matching its fixed point set.')
        maximum=scores.max()
        normalized=torch.zeros_like(scores) if maximum.item()==0 else scores/maximum
        increment=1. if self.iter==0 and self.settings['initialization']=='normalized' else self.settings['rate']
        return (self.settings['decay']*self.attention+increment*normalized).detach()

    def train(self,iterations,display_every=1):
        self._require_compiled()
        if type(iterations) is not int or iterations<0 or type(display_every) is not int or display_every<0:raise ValueError('Iterations/display interval must be non-negative integers.')
        if self.loss_formulation!='mean' or self.sampling_controller is not None or super().data_fingerprint()!=self.fixed_fingerprint:raise ValueError('RBA point identities must remain fixed; resampling is not supported.')
        self.network.train()
        for _ in range(iterations):
            self.optimiser.zero_grad(set_to_none=True);samples=self.loss_samples();raw={name:self._term(*sample) for name,sample in samples.items()}
            values,target,rows=samples['pde'];values,target=self._aligned(values,target,rows)
            if values.shape[1]!=1:raise ValueError('RBA currently supports one scalar PDE residual.')
            candidate=self.candidate_attention(values-target)
            attention_loss=((values-target)*candidate).square().mean()
            total=weighted_loss(dict(raw,pde=attention_loss),self.weights)
            if any(not torch.isfinite(v).item() for v in [*raw.values(),attention_loss,total]):raise FloatingPointError('Non-finite RBA loss; no update was taken.')
            total.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.network.parameters()):raise FloatingPointError('Non-finite RBA gradient; no update was taken.')
            self.optimiser.step();self.attention=candidate;self.iter+=1
            record={name:value.detach().item() for name,value in raw.items()};record.update(pde_weighted=attention_loss.detach().item(),total=total.detach().item())
            self.loss_history.append(record);self.training_history.append(record['total'])
            self.PDE_loss=raw['pde'].detach();self.BC_loss=sum(value.detach() for name,value in raw.items() if name!='pde');self.total_loss=total.detach()
            if display_every and self.iter%display_every==0:print(f'Iteration: {self.iter} RBA loss: {record["total"]:.6g}')
        return self.training_history

    def attention_summary(self):
        return dict(step=self.iter,minimum=self.attention.min().item(),mean=self.attention.mean().item(),maximum=self.attention.max().item(),asymptotic_bound=self.settings['rate']/(1-self.settings['decay']),weighted_pde=self.loss_history[-1]['pde_weighted'] if self.loss_history else None)

    def training_state(self):
        state=super().training_state();state['rba']=deepcopy(dict(settings=self.signature,attention=self.attention));return state

    def load_training_state(self,state):
        self._require_compiled();payload=state.get('rba') if isinstance(state,dict) else None
        if not isinstance(payload,dict) or set(payload)!={'settings','attention'} or payload['settings']!=self.signature:raise ValueError('RBA checkpoint configuration differs.')
        value=payload['attention'];steps=state.get('iter');cap=max(1. if self.settings['initialization']=='normalized' else 0.,self.settings['rate']/(1-self.settings['decay']))
        # Conservative floating-point envelope; recurrence weights are never clamped.
        if not isinstance(value,torch.Tensor) or value.shape!=self.attention.shape or value.dtype!=self.dtype or value.requires_grad or not torch.isfinite(value).all() or (value<0).any() or (value>cap*1.1+32*torch.finfo(self.dtype).eps).any() or steps==0 and (value!=0).any():raise ValueError('Invalid RBA attention tensor/count/bounds.')
        names={'pde','pde_weighted','total',*(c.name for c,_,_ in self._constraints)}
        if not isinstance(state.get('loss_history'),list) or any(not isinstance(row,dict) or set(row)!=names or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in row.values()) for row in state['loss_history']):raise ValueError('Invalid RBA loss history.')
        super().load_training_state({key:v for key,v in state.items() if key!='rba'})
        self.attention=value.to(self.device).detach().clone();return self
