"""Plain PyTorch replacement points on Margam's shared baseline training loop."""
from collections.abc import Mapping
import torch
from ..methods.pinn import PINN


class Trainer(PINN):
    """Train a user network/problem, optionally replacing loss or gradient update.

    loss: callable(prediction, target), or a mapping of term names to callables.
    step: callable(trainer, terms, total) performing the optimizer update only.
    checkpoint_id: explicit versioned identity for your physics/loss/network code.
    Samplers live in PINNProblem/PointConstraint and are sampled once per compile.
    """
    def __init__(self, problem, network, *, weights=None, loss=None, step=None,
                 loss_formulation='mean', checkpoint_id=None):
        if not isinstance(network, torch.nn.Module):
            raise TypeError('network must be a torch.nn.Module.')
        if loss is not None and not callable(loss) and not isinstance(loss, Mapping):
            raise TypeError('loss must be a callable or a mapping of term names to callables.')
        if isinstance(loss, Mapping) and any(not isinstance(k,str) or not callable(v) for k,v in loss.items()):
            raise TypeError('Loss mappings require string names and callable values.')
        if step is not None and not callable(step):raise TypeError('step must be callable.')
        if loss is not None and loss_formulation != 'mean':
            raise ValueError('Custom losses use mean mode; integral squared errors retain the existing built-in loss.')
        if checkpoint_id is not None and (not isinstance(checkpoint_id,str) or not checkpoint_id.strip() or len(checkpoint_id)>4096):
            raise ValueError('checkpoint_id must be nonempty versioned text, up to 4096 characters.')
        super().__init__(problem,network,weights=weights,loss_formulation=loss_formulation)
        self.loss = dict(loss) if isinstance(loss,Mapping) else loss
        self.step = step
        self.checkpoint_id = checkpoint_id

    @property
    def optimizer(self):
        """PyTorch spelling; refers to the same optimizer as legacy optimiser."""
        self._require_compiled()
        return self.optimiser

    def compile(self, *, optimizer=torch.optim.Adam, lr=1e-3, device='auto'):
        """optimizer is a factory(parameters, lr=...), not a prebuilt instance.

        Use functools.partial or a small function for extra optimizer options.
        Closure-based algorithms need a different procedure and are rejected.
        """
        self._compiled=False
        if not callable(optimizer):raise TypeError('optimizer must be a factory(parameters, lr=...).')
        if isinstance(self.loss,dict):
            unknown=set(self.loss)-{'pde',*(c.name for c in self.problem.constraints)}
            if unknown:raise ValueError('Unknown custom loss terms: '+', '.join(sorted(unknown)))
        def factory(parameters, *, lr):
            value=optimizer(parameters,lr=lr)
            if not isinstance(value,torch.optim.Optimizer):
                raise TypeError('optimizer factory must return a torch.optim.Optimizer.')
            return value
        super().compile(lr=lr,device=device,optimiser=factory)
        modules=self.loss.values() if isinstance(self.loss,dict) else (self.loss,)
        for module in modules:
            if isinstance(module,torch.nn.Module):
                module.to(self.device)
                if self.step is None and any(p.requires_grad for p in module.parameters()):
                    self._compiled=False
                    raise ValueError('Trainable loss modules need an explicit custom step managing their parameters.')
        return self

    def compute_losses(self):
        if self.loss is None:return super().compute_losses()
        if self.loss_formulation != 'mean' or self.sampling_controller is not None:
            raise ValueError('Custom losses currently use fixed samples and mean mode.')
        terms={}
        for name,(prediction,target,rows) in self.loss_samples().items():
            prediction,target=self._aligned(prediction,target,rows)
            function=self.loss.get(name,self.metric) if isinstance(self.loss,dict) else self.loss
            terms[name]=function(prediction,target)
        return terms

    def objective(self):
        """Validate scalar differentiable terms before any optimizer update."""
        self._require_compiled()
        from ..components.losses import weighted_loss
        terms=self.compute_losses()
        required={'pde',*(c.name for c in self.problem.constraints)}
        if not isinstance(terms,dict) or not required<=set(terms):
            raise ValueError('Losses must include pde and every named constraint.')
        for name,value in terms.items():
            if not isinstance(name,str) or not name or name=='total' or name.startswith('weighted:'):
                raise ValueError('Loss names must be nonempty and cannot use total or weighted:.')
            if not isinstance(value,torch.Tensor) or value.ndim!=0 or value.device!=self.collocation_point_sample.device or not value.is_floating_point():
                raise ValueError('Each loss must be a scalar floating tensor on the training device.')
            if not torch.isfinite(value).item():raise FloatingPointError('Non-finite research loss; no optimizer update was taken.')
        total=weighted_loss(terms,self.weights)
        if not torch.isfinite(total).item():raise FloatingPointError('Non-finite research objective; no optimizer update was taken.')
        if not total.requires_grad:raise ValueError('The objective must retain its parameter gradients.')
        return terms,total

    def record_contributions(self,record,terms,weights=None):
        return super().record_contributions(record,terms,
            getattr(self,'_record_weights',self.weights) if weights is None else weights)

    def update(self,terms,total):
        self._record_weights=dict(self.weights)
        if self.step is not None:
            self.step(self,terms,total)
            return
        total.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.network.parameters()):
            raise FloatingPointError('Non-finite research gradient; no optimizer update was taken.')
        self.optimizer.step()

    def training_state(self):
        """Tensor/primitive state; Python callbacks and code are never pickled."""
        from .checkpoint import capture
        return capture(self)

    def load_training_state(self,state):
        from .checkpoint import restore
        return restore(self,state)

    def save_checkpoint(self,path):
        """Atomically write a research checkpoint beside the chosen destination."""
        import os
        from pathlib import Path
        import tempfile
        state=self.training_state();path=Path(path);temporary=None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.margam-research-',suffix='.tmp',delete=False) as stream:
                temporary=stream.name;torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,path)
        finally:
            if temporary is not None and os.path.exists(temporary):os.unlink(temporary)

    def load_checkpoint(self,path):
        """Rebuild/compile your Python code first; loading never imports user code."""
        return self.load_training_state(torch.load(path,map_location='cpu',weights_only=True))
