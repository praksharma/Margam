"""Reconstruct prediction weights without reading training files or optimizing."""
from copy import deepcopy
import torch


class CheckpointPredictor:
    def __init__(self,config,checkpoint):
        from ..nn.factory import build_network
        self.checkpoint=deepcopy(checkpoint);self.iter=checkpoint['iter']
        self.device=torch.device('cuda' if config.device=='auto' and torch.cuda.is_available() else 'cpu' if config.device=='auto' else config.device)
        self.network=build_network(config)
        if config.method=='hard_dirichlet':
            from ..methods.hard_dirichlet import BoxDirichletNetwork
            self.network=BoxDirichletNetwork(self.network,*config.bounds,config.bc_values['left'])
        expected=self.network.state_dict();weights=checkpoint.get('network',{})
        if checkpoint.get('network_spec',getattr(self.network,'_margam_spec',None))!=getattr(self.network,'_margam_spec',None):raise ValueError('Prediction checkpoint architecture differs.')
        if set(weights)!=set(expected) or any(not isinstance(value,torch.Tensor) or value.shape!=expected[name].shape or value.dtype!=expected[name].dtype or not torch.isfinite(value).all() for name,value in weights.items()):raise ValueError('Prediction checkpoint weights are invalid.')
        self.network.load_state_dict(weights);self.network.to(self.device).eval();self.dtype=next(self.network.parameters()).dtype

    def predict(self,points):
        with torch.no_grad():return self.network(torch.as_tensor(points,dtype=self.dtype,device=self.device))

    def training_state(self):return deepcopy(self.checkpoint)


def prediction_model(config,checkpoint):
    if config.method=='expert' and config.expert_settings['windows']>1:
        from ..templates.poisson import build_pinn
        model=build_pinn(config);model.load_training_state(checkpoint);return model
    return CheckpointPredictor(config,checkpoint)
