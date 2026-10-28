"""McClenny/Braga-Neto SA-PINN: pointwise squared masks and Adam min/max.

Full-batch fixed samples, mean-normalised MSE terms, optional adaptive BCs.
See docs/self-adaptive-pinn.md for the equations and implementation choices.
"""
from copy import deepcopy
import math
import torch
from .pinn import PINN
from ..components.losses import weighted_loss


class SelfAdaptivePINN(PINN):
    def compile(self,optimiser_string='adam',lr=1e-3,metrics_string='MSE',device='auto',*,
                adaptive_lr=.005,initial_weight=1.,max_weight=1000.,adapt_boundary=False,optimiser=None):
        if optimiser_string!='adam' or optimiser is not None or metrics_string!='MSE':
            raise ValueError('SA-PINN currently uses MSE and separate Adam descent/ascent optimisers.')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in (adaptive_lr,initial_weight,max_weight)) or initial_weight>max_weight:
            raise ValueError('SA-PINN requires positive finite rates/weights and initial weight <= cap.')
        if type(adapt_boundary) is not bool:raise ValueError('Adaptive boundary setting must be boolean.')
        super().compile(optimiser_string,lr,metrics_string,device)
        if max_weight>math.sqrt(torch.finfo(self.dtype).max):
            self._compiled=False;raise ValueError('Adaptive weight cap exceeds the squared-mask dtype range.')
        self.adaptive_settings=dict(lr=adaptive_lr,initial=initial_weight,maximum=max_weight,boundary=adapt_boundary,global_weights=deepcopy(self.weights))
        sizes={'pde':len(self.collocation_point_sample)}
        if adapt_boundary:sizes.update({condition.name:len(points) for condition,points,_ in self._constraints})
        self.adaptive_settings['terms']=list(sizes)
        self.adaptive_weights={name:torch.nn.Parameter(torch.full((rows,1),initial_weight,device=self.device,dtype=self.dtype)) for name,rows in sizes.items()}
        self.adaptive_optimiser=torch.optim.Adam(self.adaptive_weights.values(),lr=adaptive_lr,maximize=True)
        return self

    def point_errors(self):
        return {name:self.point_squared_errors(*sample) for name,sample in self.loss_samples().items()}

    def objective(self,errors):
        return weighted_loss({name:((error*self.adaptive_weights[name].square()).mean() if name in self.adaptive_weights else error.mean())*self.loss_scale(name)
                              for name,error in errors.items()},self.weights)

    def train(self,iterations,display_every=1):
        self._require_compiled()
        if type(iterations) is not int or iterations<0 or type(display_every) is not int or display_every<0:
            raise ValueError('Iterations and display interval must be non-negative integers.')
        self.network.train()
        for _ in range(iterations):
            self.optimiser.zero_grad(set_to_none=True);self.adaptive_optimiser.zero_grad(set_to_none=True)
            errors=self.point_errors();total=self.objective(errors)
            if not torch.isfinite(total).item():raise FloatingPointError('Non-finite SA-PINN loss; no update was taken.')
            total.backward()
            parameters=[*self.network.parameters(),*self.adaptive_weights.values()]
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError('Non-finite SA-PINN gradient; no update was taken.')
            # Both gradients are evaluated at the same pre-update network/masks.
            self.optimiser.step();self.adaptive_optimiser.step()
            with torch.no_grad():
                for weight in self.adaptive_weights.values():weight.clamp_(0,self.adaptive_settings['maximum'])
            self.iter+=1
            record={name:(error.detach().mean()*self.loss_scale(name)).item() for name,error in errors.items()}
            record['total']=total.detach().item()
            self.loss_history.append(record);self.training_history.append(record['total'])
            self.PDE_loss=errors['pde'].detach().mean()*self.loss_scale('pde')
            self.BC_loss=sum(error.detach().mean()*self.loss_scale(name) for name,error in errors.items() if name!='pde')
            self.total_loss=total.detach()
            if display_every and self.iter%display_every==0:print(f"Iteration: {self.iter}\t SA loss: {record['total']:.6g}")
        return self.training_history

    def adaptive_summary(self):
        return {name:dict(minimum=w.detach().min().item(),mean=w.detach().mean().item(),maximum=w.detach().max().item()) for name,w in self.adaptive_weights.items()}

    def training_state(self):
        state=super().training_state()
        state['adaptive']=deepcopy(dict(settings=self.adaptive_settings,weights={name:w.detach() for name,w in self.adaptive_weights.items()},optimiser=self.adaptive_optimiser.state_dict()))
        return state

    def load_training_state(self,state):
        self._require_compiled()
        if not isinstance(state,dict) or 'adaptive' not in state:raise ValueError('SA-PINN checkpoint is missing its adaptive state.')
        adaptive=state['adaptive']
        if not isinstance(adaptive,dict) or set(adaptive)!={'settings','weights','optimiser'} or adaptive['settings']!=self.adaptive_settings:
            raise ValueError('Adaptive checkpoint settings differ from this recipe.')
        weights=adaptive['weights']
        if not isinstance(weights,dict) or set(weights)!=set(self.adaptive_weights):raise ValueError('Invalid adaptive checkpoint terms.')
        for name,value in weights.items():
            expected=self.adaptive_weights[name]
            if not isinstance(value,torch.Tensor) or value.shape!=expected.shape or value.dtype!=expected.dtype or not torch.isfinite(value).all() or (value<0).any() or (value>self.adaptive_settings['maximum']).any():
                raise ValueError('Invalid adaptive weight tensor or sample count.')
        # Validate a staged optimiser before changing live network/mask state.
        staged_parameters=[torch.nn.Parameter(torch.zeros_like(w)) for w in self.adaptive_weights.values()]
        staged=torch.optim.Adam(staged_parameters,lr=self.adaptive_settings['lr'],maximize=True)
        payload=adaptive['optimiser'];expected=staged.state_dict()['param_groups']
        if not isinstance(payload,dict) or set(payload)!={'state','param_groups'} or len(payload['param_groups'])!=len(expected) or any(g!=e for g,e in zip(payload['param_groups'],expected)):
            raise ValueError('Adaptive optimiser settings do not match the recipe.')
        staged.load_state_dict(deepcopy(payload))
        if type(state.get('iter')) is not int or state['iter']<0:raise ValueError('Invalid adaptive step count.')
        if len(staged.state)!=(len(staged_parameters) if state['iter'] else 0):raise ValueError('Invalid adaptive optimiser state count.')
        for parameter,values in staged.state.items():
            if set(values)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('Invalid adaptive optimiser fields.')
            for name,value in values.items():
                if not isinstance(value,torch.Tensor) or not torch.isfinite(value).all():
                    raise ValueError('Invalid adaptive optimiser tensor.')
                if name=='step':
                    if value.numel()!=1 or value.item()!=state['iter']:raise ValueError('Adaptive optimiser step count differs.')
                elif value.shape!=parameter.shape or value.dtype!=parameter.dtype or (name=='exp_avg_sq' and (value<0).any()):
                    raise ValueError('Invalid adaptive optimiser moment.')
        super().load_training_state({k:v for k,v in state.items() if k!='adaptive'})
        with torch.no_grad():
            for name,weight in self.adaptive_weights.items():weight.copy_(weights[name])
        self.adaptive_optimiser.load_state_dict(deepcopy(payload))
        return self
