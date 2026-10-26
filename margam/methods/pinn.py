"""Baseline PINN: fixed samples, strong-form residuals and full-batch updates.

All baseline preparation, loss evaluation, optimisation and prediction live
here. Other methods can reuse components without branching this trainer.
"""
import torch
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

    def _term(self, values, target, rows):
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
        return self.metric(values, target)

    def compute_losses(self):
        """Build fresh derivative graphs; return named, unweighted terms."""
        self._require_compiled()
        points = self.collocation_point_sample.detach().requires_grad_(True)
        residual = self.problem.residual(points, self.network(points))
        terms = {"pde": self._term(residual, self.collocation_point_labels, len(points))}
        for condition, sample, target in self._constraints:
            points = sample.detach().requires_grad_(condition.operator is not None)
            values = self.network(points)
            if condition.operator is not None:
                values = condition.operator(points, values)
            terms[condition.name] = self._term(values, target, len(points))
        return terms

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
