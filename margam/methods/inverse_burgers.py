"""Independent continuous-time identification from Raissi et al., Part II §2.1."""
from copy import deepcopy
import math
import torch
from .pinn import PINN


class InverseBurgersNetwork(torch.nn.Module):
    def __init__(self,network,initial):
        super().__init__();self.solution=network
        self.lambda_1=torch.nn.Parameter(torch.tensor(float(initial['lambda_1'])))
        self.log_lambda_2=torch.nn.Parameter(torch.tensor(float(initial['log_lambda_2'])))
        self._margam_spec=dict(type='inverse_burgers',options=deepcopy(initial),network=deepcopy(network._margam_spec))

    def forward(self,points):return self.solution(points)

    def estimated_parameters(self):return dict(lambda_1=self.lambda_1.detach().item(),lambda_2=self.log_lambda_2.detach().exp().item())


def burgers_residual(points,values,network):
    from ..components.differential import derivative
    return derivative(values,points,1)+network.lambda_1*values*derivative(values,points,0)-network.log_lambda_2.exp()*derivative(values,points,0,2)


class InverseBurgersPINN(PINN):
    def __init__(self,problem,network,*,weights,initial):
        super().__init__(problem,network,weights=weights);self.inverse_history=[]
        self.inverse_signature=dict(initial=deepcopy(initial),weights=deepcopy(weights),algorithm='continuous-time-burgers-identification-v1')
        def finite_gradient(gradient):
            if not torch.isfinite(gradient).all():raise FloatingPointError('Non-finite inverse gradient; no optimizer step was taken.')
            return gradient
        for parameter in network.parameters():parameter.register_hook(finite_gradient)

    def estimated_parameters(self):return self.network.estimated_parameters()

    def train(self,iterations,display_every=1):
        if type(iterations) is not int or iterations<0 or type(display_every) is not int or display_every<0:raise ValueError('Use non-negative integer updates/display interval.')
        for _ in range(iterations):
            super().train(1,display_every)
            values=self.estimated_parameters()
            if not all(math.isfinite(v) for v in values.values()) or values['lambda_2']<=0:raise FloatingPointError('Non-finite/underflowed physical coefficients after the inverse update.')
            self.inverse_history.append(dict(step=self.iter,**values))
        return self.training_history

    def training_state(self):
        state=super().training_state();state['inverse']=dict(settings=deepcopy(self.inverse_signature),history=deepcopy(self.inverse_history));return state

    def load_training_state(self,state):
        payload=state.get('inverse') if isinstance(state,dict) else None
        if not isinstance(payload,dict) or set(payload)!={'settings','history'} or payload['settings']!=self.inverse_signature:raise ValueError('Inverse settings/objective differ from this checkpoint.')
        history=payload['history'];steps=state.get('iter')
        if type(steps) is not int or not isinstance(history,list) or len(history)!=steps or any(not isinstance(row,dict) or set(row)!={'step','lambda_1','lambda_2'} or type(row['step']) is not int or row['step']!=i+1 or any(type(row[k]) not in (int,float) or not math.isfinite(row[k]) for k in ('lambda_1','lambda_2')) or row['lambda_2']<=0 for i,row in enumerate(history)):raise ValueError('Invalid inverse parameter history.')
        weights=state.get('network',{})
        for key in ('lambda_1','log_lambda_2'):
            value=weights.get(key)
            if not isinstance(value,torch.Tensor) or value.shape!=torch.Size([]) or not torch.isfinite(value):raise ValueError('Invalid learned coefficient checkpoint.')
        physical=dict(lambda_1=weights['lambda_1'].item(),lambda_2=weights['log_lambda_2'].exp().item())
        if not math.isfinite(physical['lambda_2']) or physical['lambda_2']<=0 or history and any(not math.isclose(history[-1][key],physical[key],rel_tol=1e-6,abs_tol=1e-9) for key in physical):raise ValueError('Coefficient history differs from the saved parameters.')
        baseline={key:value for key,value in state.items() if key!='inverse'}
        previous=super().training_state()
        try:super().load_training_state(baseline)
        except Exception:
            super().load_training_state(previous);raise
        self.inverse_history=deepcopy(history);return self


def build_inverse(config):
    import hashlib,json
    from ..components.inverse_config import validate_inverse,read_observations
    from ..components.prepared_data import file_fingerprint
    from pathlib import Path
    from ..components.problem import PINNProblem,PointConstraint
    from ..nn.factory import build_network
    validate_inverse(config);digest,_=file_fingerprint(Path(config.interior_data['path']));points,values=read_observations(config)
    if file_fingerprint(Path(config.interior_data['path']))[0]!=digest:raise ValueError('Observation file changed while preparing training data.')
    network=build_network(config)
    problem=PINNProblem(lambda x,u:burgers_residual(x,u,network),torch.tensor(points),[PointConstraint('observations',torch.tensor(points),torch.tensor(values))],data_identity=json.dumps(dict(sha256=digest,initial=config.inverse_settings,weights=config.loss_weights),sort_keys=True))
    return InverseBurgersPINN(problem,network,weights=config.loss_weights,initial=config.inverse_settings).compile(lr=config.lr,device=config.device)
