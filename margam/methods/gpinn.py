"""Independent Yu et al. (2022) gPINN: coordinate derivatives of PDE residuals."""
from copy import deepcopy
import math
import torch
from .pinn import PINN
from ..components.differential import derivative
from ..components.losses import weighted_loss


class GradientEnhancedPINN(PINN):
    def __init__(self,problem,network,*,coordinates,gradient_weights,weights=None,configuration=None):
        super().__init__(problem,network,weights=weights)
        from ..components.gpinn_config import validate_gpinn_weights
        validate_gpinn_weights(gradient_weights)
        if tuple(coordinates) not in (('x',),('x','y'),('x','t')):raise ValueError('gPINN currently uses scalar x, x/y or x/t coordinates.')
        self.coordinates=tuple(coordinates);self.gradient_weights=deepcopy(gradient_weights)
        self.physical_weights=deepcopy(self.weights)
        self.configuration=deepcopy(configuration or {})
        for key in ('iterations','device'):self.configuration.pop(key,None)

    def compile(self,optimiser_string='adam',lr=.001,metrics_string='MSE',device='auto',*,optimiser=None):
        if optimiser_string!='adam' or metrics_string!='MSE' or optimiser is not None:raise ValueError('gPINN currently uses Adam and mean squared losses.')
        if not callable(self.problem.residual_target) and torch.as_tensor(self.problem.residual_target).ndim!=0:raise ValueError('gPINN residual targets must be scalar or coordinate-callable so their derivatives are defined.')
        self.weights=deepcopy(self.physical_weights)
        super().compile(optimiser_string,lr,metrics_string,device)
        if self.collocation_point_sample.shape[1]!=len(self.coordinates):raise ValueError('gPINN coordinate names must match the point dimension.')
        self.directions={name:i for i,name in enumerate(self.coordinates) if self.gradient_weights[name]>0}
        for name in self.directions:
            term='pde_grad_'+name
            if term in self.loss_measures:raise ValueError('A boundary name conflicts with '+term)
            self.weights[term]=self.weights.get('pde',1.)*self.gradient_weights[name]
        self.signature=dict(algorithm='yu2022-scalar-residual-gradients',coordinates=list(self.coordinates),gradient_weights=deepcopy(self.gradient_weights),weights=deepcopy(self.weights),configuration=self.configuration)
        return self

    def pde_samples(self,x):
        residual=self.problem.residual(x,self.network(x))
        target=self.problem.residual_target(x) if callable(self.problem.residual_target) else self.problem.residual_target
        target=torch.as_tensor(target,dtype=self.dtype,device=self.device)
        residual,target=self._aligned(residual,target,len(x))
        if residual.shape[1]!=1:raise ValueError('gPINN currently supports one scalar residual.')
        error=residual-target
        terms={'pde':(residual,target,len(x))}
        for name,axis in self.directions.items():
            gradient=derivative(error,x,axis)
            terms['pde_grad_'+name]=(gradient,torch.zeros_like(gradient),len(x))
        return terms

    def compute_losses(self):
        if self.loss_formulation!='mean' or self.sampling_controller is not None and self.sampling_controller.settings['method']!='rar':raise ValueError('gPINN supports mean loss with fixed points or RAR only.')
        return super().compute_losses()

    def train(self,iterations,display_every=1):
        self._require_compiled()
        if type(iterations) is not int or iterations<0 or type(display_every) is not int or display_every<0:raise ValueError('Iterations/display interval must be non-negative integers.')
        self.network.train()
        for _ in range(iterations):
            self._before_training_step();self.optimiser.zero_grad(set_to_none=True);terms=self.compute_losses()
            if any(not torch.isfinite(value).item() for value in terms.values()):raise FloatingPointError('Non-finite gPINN loss; no optimizer update was taken.')
            total=weighted_loss(terms,self.weights)
            if not torch.isfinite(total).item():raise FloatingPointError('Non-finite gPINN objective; no optimizer update was taken.')
            total.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.network.parameters()):raise FloatingPointError('Non-finite gPINN gradient; no optimizer update was taken.')
            self.optimiser.step();self.iter+=1
            record={name:loss.detach().item() for name,loss in terms.items()};record['total']=total.detach().item()
            self.loss_history.append(record);self.training_history.append(record['total'])
            self.PDE_loss=terms['pde'].detach();self.BC_loss=sum(terms[c.name].detach() for c,_,_ in self._constraints);self.total_loss=total.detach()
            if display_every and self.iter%display_every==0:print(f'Iteration: {self.iter} gPINN loss: {record["total"]:.6g}')
        return self.training_history

    def training_state(self):
        state=super().training_state();state['gpinn']=deepcopy(self.signature);return state

    def load_training_state(self,state):
        if not isinstance(state,dict) or state.get('gpinn')!=self.signature:raise ValueError('gPINN checkpoint gradient weights, coordinates or configuration differ.')
        names={'pde','total',*('pde_grad_'+name for name in self.directions),*(c.name for c,_,_ in self._constraints)}
        history=state.get('loss_history')
        if not isinstance(history,list) or any(not isinstance(row,dict) or set(row)!=names or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in row.values()) for row in history):raise ValueError('Invalid gPINN loss history.')
        super().load_training_state({key:value for key,value in state.items() if key!='gpinn'});return self
