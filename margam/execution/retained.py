"""CPU checkpoint facade for desktop inspection; never holds a live trainer."""
from copy import deepcopy
from types import SimpleNamespace
from .codec import encode_config,encode_request


class NetworkSnapshot:
    def __init__(self,owner):self.owner=owner
    def state_dict(self):return deepcopy(self.owner._checkpoint['network'])
    def __getattr__(self,name):
        if name=='estimated_parameters' and 'estimated_parameters' in self.owner.metadata:
            return lambda:deepcopy(self.owner.metadata[name])
        raise AttributeError(name)


class RetainedModel:
    """Device identifies the numerical run; all retained tensor storage is CPU."""
    storage_device='cpu'
    def __init__(self,config,checkpoint,metadata,device):
        self.config=deepcopy(config);self._checkpoint=checkpoint;self.metadata=metadata;self.device=device
        self.network=NetworkSnapshot(self);self.iter=checkpoint['iter'];self.loss_history=checkpoint['loss_history']
        self.training_history=checkpoint['training_history']
        self.collocation_point_sample=metadata.get('points',checkpoint.get('sampling',{}).get('points'))
        self._constraints=[(SimpleNamespace(name=item['name']),item['points'],item['target']) for item in metadata.get('constraints',[])]
        sampling=checkpoint.get('sampling')
        self.sampling_controller=SimpleNamespace(**sampling) if sampling else None
        if 'rba' in checkpoint:self.attention=checkpoint['rba']['attention']
        if 'inverse' in checkpoint:self.inverse_history=checkpoint['inverse']['history']
    def training_state(self):return deepcopy(self._checkpoint)
    def __getattr__(self,name):
        if name.endswith('_summary') and name in self.metadata:return lambda:deepcopy(self.metadata[name])
        raise AttributeError(name)
    def predict(self,points):
        # Compatibility for private desktop callers; UI evaluation uses its own
        # asynchronous worker. Research code uses actual margam.research trainers.
        import torch
        from .transport import JobTransport
        payload=dict(operation='predict',config=encode_config(self.config),checkpoint=self.training_state(),requests=[],
            dtype=str(next(iter(self._checkpoint['network'].values())).dtype).removeprefix('torch.'),
            points=torch.as_tensor(points,dtype=next(iter(self._checkpoint['network'].values())).dtype).detach().cpu(),metadata=self.metadata)
        return JobTransport().run(payload)['result']['prediction']
