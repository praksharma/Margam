"""Wang/Teng/Perdikaris 2020 v1 Algorithm 1: max/mean gradient loss balancing.

No architecture changes or optimiser-rate decay. Scalar coefficients are updated
from raw (not already lambda-weighted) constraint gradients; see docs/lra-pinn.md.
"""
from copy import deepcopy
import math
import torch
from .pinn import PINN
from ..components.losses import weighted_loss


def _finite(value):return type(value) in (int,float) and math.isfinite(value)


class LearningRateAnnealingPINN(PINN):
    def compile(self,optimiser_string='adam',lr=.001,metrics_string='MSE',device='auto',*,
                alpha=.9,update_interval=10,epsilon=1e-12,max_weight=1e6,optimiser=None):
        if optimiser_string!='adam' or optimiser is not None or metrics_string!='MSE':
            raise ValueError('LRA currently uses MSE and full-batch Adam.')
        if not _finite(alpha) or not 0<alpha<=1 or type(update_interval) is not int or not 1<=update_interval<=100000 or not _finite(epsilon) or not 0<epsilon<=1 or not _finite(max_weight) or not 1<=max_weight<=1e12:
            raise ValueError('Invalid LRA smoothing, interval, gradient floor or coefficient cap.')
        super().compile(optimiser_string,lr,metrics_string,device)
        self._lra_parameters=tuple(p for p in self.network.parameters() if p.requires_grad)
        if not self._lra_parameters or self.weights.get('pde',1.)<=0:
            self._compiled=False;raise ValueError('LRA needs trainable parameters and a positive PDE weight.')
        self.lra_weights={condition.name:1. for condition,_,_ in self._constraints}
        self.lra_settings=dict(algorithm='wang2020v1-raw-max-mean',alpha=float(alpha),interval=update_interval,
                               epsilon=float(epsilon),maximum=float(max_weight),global_weights=deepcopy(self.weights),terms=list(self.lra_weights))
        self.lra_history=[]
        return self

    def _gradient_stats(self,loss):
        gradients=torch.autograd.grad(loss,self._lra_parameters,retain_graph=True,allow_unused=True) if loss.requires_grad else (None,)*len(self._lra_parameters)
        maximum=0.;absolute_sum=0.;count=0
        for parameter,gradient in zip(self._lra_parameters,gradients):
            count+=parameter.numel()  # Include zeros for unused trainable entries.
            if gradient is None:continue
            if not torch.isfinite(gradient).all():raise FloatingPointError('Non-finite LRA gradient statistics; no update was taken.')
            values=gradient.detach().abs()
            maximum=max(maximum,values.max().item());absolute_sum+=values.sum(dtype=torch.float64).item()
        return maximum,absolute_sum/count

    def _candidate_weights(self,terms):
        reference,_=self._gradient_stats(terms['pde']*self.weights.get('pde',1.))
        means={};targets={};weights={}
        for name,old in self.lra_weights.items():
            _,mean=self._gradient_stats(terms[name]*self.weights.get(name,1.));means[name]=mean
            # A zero/uninformative constraint gradient supplies no finite ratio.
            target=None if mean<=self.lra_settings['epsilon'] else min(reference/mean,self.lra_settings['maximum'])
            targets[name]=target
            weights[name]=old if target is None else (1-self.lra_settings['alpha'])*old+self.lra_settings['alpha']*target
        return weights,dict(step=self.iter+1,pde_max=reference,means=means,targets=targets,weights=deepcopy(weights))

    def train(self,iterations,display_every=1):
        self._require_compiled()
        if type(iterations) is not int or iterations<0 or type(display_every) is not int or display_every<0:
            raise ValueError('Iterations and display interval must be non-negative integers.')
        self.network.train()
        for _ in range(iterations):
            self._before_training_step()
            self.optimiser.zero_grad(set_to_none=True);terms=self.compute_losses()
            if any(not torch.isfinite(loss).item() for loss in terms.values()):
                raise FloatingPointError('Non-finite LRA physical loss; no update was taken.')
            weights=self.lra_weights;event=None
            if self.iter%self.lra_settings['interval']==0:weights,event=self._candidate_weights(terms)
            effective={name:self.weights.get(name,1.)*(weights[name] if name in weights else 1.) for name in terms}
            total=weighted_loss(terms,effective)
            if not torch.isfinite(total).item():raise FloatingPointError('Non-finite LRA objective; no update was taken.')
            total.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self._lra_parameters):
                raise FloatingPointError('Non-finite LRA network gradient; no update was taken.')
            self.optimiser.step()
            self.iter+=1;self.lra_weights=weights
            if event is not None:self.lra_history.append(event)
            record={name:loss.detach().item() for name,loss in terms.items()};record['total']=total.detach().item()
            self.loss_history.append(record);self.training_history.append(record['total'])
            self.PDE_loss=terms['pde'].detach();self.BC_loss=sum(loss.detach() for name,loss in terms.items() if name!='pde');self.total_loss=total.detach()
            if display_every and self.iter%display_every==0:print(f"Iteration: {self.iter}\t LRA loss: {record['total']:.6g}")
        return self.training_history

    def balancing_summary(self):
        latest=self.lra_history[-1] if self.lra_history else None
        return dict(step=latest['step'] if latest else None,pde_max=latest['pde_max'] if latest else None,
                    terms={name:dict(coefficient=value,effective_weight=value*self.weights.get(name,1.),
                                     mean_gradient=latest['means'][name] if latest else None)
                           for name,value in self.lra_weights.items()})

    def training_state(self):
        state=super().training_state()
        state['lra']=deepcopy(dict(settings=self.lra_settings,weights=self.lra_weights,history=self.lra_history))
        return state

    def load_training_state(self,state):
        self._require_compiled()
        if not isinstance(state,dict) or not isinstance(state.get('lra'),dict):raise ValueError('LRA checkpoint is missing its balancing state.')
        payload=state['lra']
        if set(payload)!={'settings','weights','history'} or payload['settings']!=self.lra_settings:
            raise ValueError('LRA checkpoint settings differ from this recipe.')
        names=set(self.lra_weights);history=payload['history'];steps=state.get('iter')
        if type(steps) is not int or not 0<=steps<=100000 or not isinstance(history,list) or len(history)!=((steps-1)//self.lra_settings['interval']+1 if steps else 0):
            raise ValueError('LRA history does not match its update phase.')
        previous={name:1. for name in names}
        def valid_weights(value):return isinstance(value,dict) and set(value)==names and all(_finite(v) and 0<=v<=self.lra_settings['maximum'] for v in value.values())
        for index,event in enumerate(history):
            if not isinstance(event,dict) or set(event)!={'step','pde_max','means','targets','weights'} or type(event['step']) is not int or event['step']!=1+index*self.lra_settings['interval'] or not _finite(event['pde_max']) or event['pde_max']<0:
                raise ValueError('Invalid LRA history event.')
            if not isinstance(event['means'],dict) or set(event['means'])!=names or not isinstance(event['targets'],dict) or set(event['targets'])!=names or not valid_weights(event['weights']):raise ValueError('Invalid LRA term data.')
            for name in names:
                mean=event['means'][name];target=event['targets'][name]
                if not _finite(mean) or mean<0:raise ValueError('Invalid LRA gradient statistic.')
                expected=None if mean<=self.lra_settings['epsilon'] else min(event['pde_max']/mean,self.lra_settings['maximum'])
                if expected is None:
                    if target is not None:raise ValueError('Invalid LRA zero-gradient handling.')
                    updated=previous[name]
                else:
                    if not _finite(target) or not math.isclose(target,expected,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Invalid LRA ratio.')
                    updated=(1-self.lra_settings['alpha'])*previous[name]+self.lra_settings['alpha']*target
                if not math.isclose(event['weights'][name],updated,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Invalid LRA moving average.')
            previous=event['weights']
        if not valid_weights(payload['weights']) or payload['weights']!=previous:raise ValueError('LRA final coefficients do not match history.')
        super().load_training_state({key:value for key,value in state.items() if key!='lra'})
        self.lra_weights=deepcopy(payload['weights']);self.lra_history=deepcopy(history)
        return self
