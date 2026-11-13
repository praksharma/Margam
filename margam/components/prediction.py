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

    def estimated_parameters(self):return self.network.estimated_parameters() if hasattr(self.network,'estimated_parameters') else {}

    def training_state(self):return deepcopy(self.checkpoint)


def prediction_model(config,checkpoint):
    if config.method=='expert' and config.expert_settings['windows']>1:
        from ..templates.poisson import build_pinn
        model=build_pinn(config);model.load_training_state(checkpoint);return model
    return CheckpointPredictor(config,checkpoint)


# Native factories create pointwise networks. Keep this bound operational rather
# than a model/training parameter; arbitrary research predict() stays unchanged.
NATIVE_PREDICTION_BATCH = 2048


def predict_in_batches(model,points,batch_size=NATIVE_PREDICTION_BATCH):
    """Explicit pointwise inference with bounded device rows and CPU assembly.

    Native output jobs use this for known factory networks. Research callers must
    opt in only for networks whose predictions do not depend on batch membership.
    Output values can differ at floating-point roundoff from a single large GEMM.
    """
    if type(batch_size) is not int or not 1<=batch_size<=100000:
        raise ValueError('Prediction batch size must be an integer between 1 and 100000.')
    dtype=getattr(model,'dtype',torch.get_default_dtype())
    if not isinstance(dtype,torch.dtype):dtype=torch.get_default_dtype()
    rows=torch.as_tensor(points,device='cpu',dtype=dtype).detach()
    if rows.ndim!=2 or not len(rows) or not rows.shape[1] or not torch.isfinite(rows).all():
        raise ValueError('Prediction points must be a finite non-empty (N, D) array.')
    result=None;tail=None;value_dtype=None
    for start in range(0,len(rows),batch_size):
        values=model.predict(rows[start:start+batch_size])
        count=min(batch_size,len(rows)-start)
        if not isinstance(values,torch.Tensor) or values.ndim not in (1,2) or values.shape[0]!=count or values.ndim==2 and not values.shape[1] or not values.is_floating_point() or not torch.isfinite(values).all():
            raise ValueError('Prediction batches must contain finite floating-point values matching their rows.')
        if result is None:
            tail=values.shape[1:];value_dtype=values.dtype
            result=torch.empty((len(rows),*tail),dtype=value_dtype,device='cpu')
        elif values.shape[1:]!=tail or values.dtype!=value_dtype:
            raise ValueError('Prediction batch fields and dtype must remain consistent.')
        result[start:start+count].copy_(values.detach().to('cpu'))
        del values
    return result
