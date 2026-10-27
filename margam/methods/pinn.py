"""Baseline PINN: fixed samples, strong-form residuals and full-batch updates.

All baseline preparation, loss evaluation, optimisation and prediction live
here. Other methods can reuse components without branching this trainer.
"""
import torch
from copy import deepcopy
import hashlib
import json
from ..backend import choose_optimiser, loss_metric
from ..components import weighted_loss


class PINN:
    """Train one pointwise torch module against a PINNProblem.

    Seed before constructing the network and compiling for reproducibility.
    """
    def __init__(self, problem, network, *, weights=None):
        self.problem = problem
        self.network = network
        self.weights = dict(weights or {})
        self._compiled = False

    def compile(self, optimiser_string="adam", lr=1e-3,
                metrics_string="MSE", device="auto", *, optimiser=None):
        """Sample, move data/model, and reset optimisation and histories.

        device="auto" selects CUDA when available, otherwise CPU.
        An optional optimiser factory accepts (parameters, lr=...).
        Closure-based optimisers require a different training procedure.
        """
        self._compiled = False
        if not isinstance(getattr(self.problem,'data_identity',''),str):raise ValueError("Problem data identity must be text.")
        self._data_identity=getattr(self.problem,'data_identity','')
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.network.to(self.device)
        parameters = list(self.network.parameters())
        if not parameters:
            raise ValueError("The network must have trainable parameters.")
        self.dtype = parameters[0].dtype
        self.metric = loss_metric(metrics_string)
        self.collocation_point_sample = self._points(self.problem.collocation_points)
        self.collocation_point_labels = self._target(
            self.problem.residual_target, self.collocation_point_sample)
        self._constraints = []
        names = {"pde"}
        for condition in self.problem.constraints:
            if not condition.name or condition.name in names:
                raise ValueError("Constraint names must be unique and cannot be 'pde'.")
            names.add(condition.name)
            points = self._points(condition.points)
            self._constraints.append((condition, points, self._target(condition.target, points)))
        weighted_loss({name: torch.zeros((), device=self.device) for name in names}, self.weights)
        factory = choose_optimiser(optimiser_string) if optimiser is None else optimiser
        self.optimiser = factory(parameters, lr=lr)
        if isinstance(self.optimiser, torch.optim.LBFGS):
            raise ValueError("LBFGS requires a closure-based recipe.")
        self.iter = 0
        self.training_history = []
        self.loss_history = []
        self._compiled = True
        return self

    def _points(self, source):
        points = source() if callable(source) else source
        points = torch.as_tensor(points, device=self.device, dtype=self.dtype)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
            raise ValueError("Points must be a non-empty (N, D) tensor.")
        return points.detach().clone()

    def _target(self, source, points):
        target = source(points) if callable(source) else source
        return torch.as_tensor(target, device=self.device, dtype=self.dtype).detach().clone()

    def _aligned(self, values, target, rows):
        if values.ndim == 1:
            values = values.unsqueeze(1)
        if values.ndim != 2 or values.shape[0] != rows:
            raise ValueError("Residuals and predictions must have shape (N, C) or (N,).")
        if target.ndim == 0:
            target = target.expand_as(values)
        elif target.ndim == 1:
            target = target.unsqueeze(1)
        if target.shape != values.shape:
            raise ValueError(f"Target shape {tuple(target.shape)} does not match {tuple(values.shape)}.")
        return values, target

    def _term(self, values, target, rows):
        return self.metric(*self._aligned(values,target,rows))

    def loss_samples(self):
        """Shared fresh derivative graphs and shape data for each physical term."""
        self._require_compiled()
        points = self.collocation_point_sample.detach().requires_grad_(True)
        residual = self.problem.residual(points, self.network(points))
        terms = {"pde": (residual, self.collocation_point_labels, len(points))}
        for condition, sample, target in self._constraints:
            points = sample.detach().requires_grad_(condition.operator is not None)
            values = self.network(points)
            if condition.operator is not None:
                values = condition.operator(points, values)
            terms[condition.name] = (values, target, len(points))
        return terms

    def compute_losses(self):
        """Return named, unweighted physical losses."""
        return {name:self._term(*sample) for name,sample in self.loss_samples().items()}

    def train(self, iterations, display_every=1):
        """Run exactly iterations additional updates; return total-loss history.

        Losses are measured before each update. display_every=0 is quiet.
        """
        self._require_compiled()
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
            raise ValueError("iterations must be a non-negative integer.")
        if isinstance(display_every, bool) or not isinstance(display_every, int) or display_every < 0:
            raise ValueError("display_every must be a non-negative integer.")
        self.network.train()
        for _ in range(iterations):
            self.optimiser.zero_grad(set_to_none=True)
            terms = self.compute_losses()
            total = weighted_loss(terms, self.weights)
            if not torch.isfinite(total).item():
                raise FloatingPointError("Non-finite PINN loss; no optimiser step was taken.")
            total.backward()
            self.optimiser.step()
            self.iter += 1
            record = {name: loss.detach().item() for name, loss in terms.items()}
            record["total"] = total.detach().item()
            self.loss_history.append(record)
            self.training_history.append(record["total"])
            self.PDE_loss = terms["pde"].detach()
            self.BC_loss = sum(loss.detach() for name, loss in terms.items() if name != "pde")
            self.total_loss = total.detach()
            if display_every and self.iter % display_every == 0:
                print(f"Iteration: {self.iter}\t Loss: {record['total']:.6g}")
        return self.training_history

    def data_fingerprint(self):
        digest=hashlib.sha256()
        arrays=[self.collocation_point_sample,self.collocation_point_labels]
        for _,points,target in self._constraints:arrays.extend((points,target))
        for value in arrays:
            value=value.detach().cpu().contiguous()
            digest.update(str((tuple(value.shape),str(value.dtype))).encode())
            digest.update(value.numpy().tobytes())
        if self._data_identity:digest.update(self._data_identity.encode())
        return digest.hexdigest()

    def training_state(self):
        """Tensor/primitive checkpoint for this baseline recipe, not a pickled model."""
        self._require_compiled()
        if not isinstance(self.optimiser,torch.optim.Adam):raise ValueError('Checkpoints currently support the baseline Adam recipe only.')
        state={'network':self.network.state_dict(),'optimiser':self.optimiser.state_dict(),
               'data_fingerprint':self.data_fingerprint(),'iter':self.iter,'loss_history':self.loss_history,'training_history':self.training_history}
        if hasattr(self.network,'_margam_spec'):state['network_spec']=self.network._margam_spec
        return deepcopy(state)

    def load_training_state(self,state):
        self._require_compiled()
        fields={'network','optimiser','data_fingerprint','iter','loss_history','training_history'}
        if not isinstance(state,dict) or not fields<=set(state)<=fields|{'network_spec'}:
            raise ValueError('Invalid baseline checkpoint.')
        if state['data_fingerprint']!=self.data_fingerprint():raise ValueError('Checkpoint point data/targets differ from the current files or sampling recipe.')
        expected_spec=getattr(self.network,'_margam_spec',None)
        if 'network_spec' in state:
            try:matches=json.dumps(state['network_spec'],sort_keys=True,allow_nan=False)==json.dumps(expected_spec,sort_keys=True,allow_nan=False)
            except (TypeError,ValueError):matches=False
            if not matches:raise ValueError('Checkpoint network architecture/settings differ from this recipe.')
        elif expected_spec is not None and (expected_spec['type']!='fcnn' or expected_spec['options']):
            raise ValueError('This architecture requires a network settings checkpoint.')
        steps=state['iter']
        if type(steps) is not int or not 0<=steps<=100000 or len(state['loss_history'])!=steps or len(state['training_history'])!=steps:
            raise ValueError('Checkpoint history does not match its step count.')
        expected=self.network.state_dict()
        if set(state['network'])!=set(expected) or any(not isinstance(value,torch.Tensor) or value.shape!=expected[key].shape or value.dtype!=expected[key].dtype or not torch.isfinite(value).all() for key,value in state['network'].items()):
            raise ValueError('Checkpoint network does not match the configured architecture/dtype.')
        groups=state['optimiser'].get('param_groups',[])
        expected_groups=self.optimiser.state_dict()['param_groups']
        if len(groups)!=len(expected_groups) or any(set(group)!=set(expected_group) or any(group[key]!=expected_group[key] for key in group if key!='params') for group,expected_group in zip(groups,expected_groups)):
            raise ValueError('Checkpoint optimiser settings do not match the recipe.')
        self.network.load_state_dict(state['network'])
        self.optimiser.load_state_dict(state['optimiser'])
        for parameter,values in self.optimiser.state.items():
            for name,value in values.items():
                if isinstance(value,torch.Tensor) and (not torch.isfinite(value).all() or (name!='step' and value.shape!=parameter.shape)):
                    raise ValueError('Invalid optimiser checkpoint tensor.')
        self.iter=steps
        self.loss_history=deepcopy(state['loss_history']);self.training_history=list(state['training_history'])
        return self

    def predict(self, points):
        """Evaluate values without gradients, restoring the network's mode."""
        self._require_compiled()
        was_training = self.network.training
        try:
            self.network.eval()
            with torch.no_grad():
                return self.network(self._points(points))
        finally:
            self.network.train(was_training)

    def _require_compiled(self):
        if not self._compiled:
            raise RuntimeError("Call compile() before training, predicting or computing losses.")
