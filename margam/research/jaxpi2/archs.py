"""PyTorch conversion of JAXPI2 architectures (PirateNet, ModifiedMlp, Mlp, embeddings).

Matches sifanexisted/jaxpi2 (jaxpi/archs.py) and arXiv:2507.08972v2 Section 'Network
Architecture', with optional Random Weight Factorization (RWF) from Eqs. (11)-(12).
"""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F


def stable_gelu(x: torch.Tensor) -> torch.Tensor:
    """Tanh-approximation GELU formula with stable higher-order autograd paths.

    PyTorch's fused GELU double backward uses cosh; differentiating it for
    strong-form PINN parameter gradients can overflow at finite activations.
    Composing tanh directly keeps higher derivatives bounded.
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def stable_swish(x: torch.Tensor) -> torch.Tensor:
    """Swish / SiLU activation x * sigmoid(x) with bounded higher-order autograd.

    Composing x * torch.sigmoid(x) expresses all higher derivatives as polynomials
    in s = sigmoid(x) in (0, 1), avoiding exp/cosh overflow in 3rd-order PINN grads.
    """
    return x * torch.sigmoid(x)


def get_activation(name: str):
    key = name.lower() if isinstance(name, str) else ""
    if key == "relu":
        return F.relu
    if key == "gelu":
        return stable_gelu
    if key in ("swish", "silu"):
        return stable_swish
    if key == "sigmoid":
        return torch.sigmoid
    if key == "tanh":
        return torch.tanh
    if key == "sin":
        return torch.sin
    raise NotImplementedError(f"Activation {name} not supported yet!")


class DirectFactorizedLinear(nn.Module):
    """Random Weight Factorization W = diag(g) V (Torch row convention)."""

    def __init__(self, inputs: int, outputs: int, mean: float = 1.0, std: float = 0.1):
        super().__init__()
        sigma = math.sqrt(2.0 / (inputs + outputs)) / 0.87962566103423978
        weight = nn.init.trunc_normal_(
            torch.empty(outputs, inputs), std=sigma, a=-2.0 * sigma, b=2.0 * sigma
        )
        g = torch.randn(outputs).mul_(std).add_(mean).exp_()
        self.g = nn.Parameter(g)
        self.v = nn.Parameter(weight / g[:, None])
        self.bias = nn.Parameter(torch.zeros(outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.g[:, None] * self.v, self.bias)


class PeriodEmbs(nn.Module):
    """Exact periodic cos/sin embeddings for selected coordinate axes (Dong & Ni 2021)."""

    def __init__(
        self,
        period: Sequence[float],
        axis: Sequence[int],
        trainable: Sequence[bool],
    ):
        super().__init__()
        if not (len(period) == len(axis) == len(trainable)) or len(axis) == 0:
            raise ValueError("PeriodEmbs requires matching non-empty period, axis, and trainable sequences.")
        self.period = tuple(float(p) for p in period)
        self.axis = tuple(int(a) for a in axis)
        self.trainable = tuple(bool(t) for t in trainable)
        for idx, is_trainable in enumerate(self.trainable):
            val = torch.tensor(self.period[idx])
            if is_trainable:
                self.register_parameter(f"period_{idx}", nn.Parameter(val))
            else:
                self.register_buffer(f"period_{idx}", val)

    def output_dim(self, input_dim: int) -> int:
        if any(a < 0 or a >= input_dim for a in self.axis):
            raise ValueError(f"Periodic axis {self.axis} out of range for input_dim={input_dim}.")
        return input_dim + len(self.axis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = []
        for i in range(x.shape[-1]):
            xi = x[..., i : i + 1]
            if i in self.axis:
                idx = self.axis.index(i)
                p = getattr(self, f"period_{idx}")
                parts.extend((torch.cos(p * xi), torch.sin(p * xi)))
            else:
                parts.append(xi)
        return torch.cat(parts, dim=-1)

    def reference_parameter_map(self, scope: str = "PeriodEmbs_0"):
        result = []
        for idx, is_trainable in enumerate(self.trainable):
            if is_trainable:
                result.append((getattr(self, f"period_{idx}"), (scope, f"period_{idx}"), False))
        return result


class FourierEmbs(nn.Module):
    """Trainable Random Fourier Feature embedding [cos(x B), sin(x B)]."""

    def __init__(self, in_features: int, embed_scale: float = 1.0, embed_dim: int = 256):
        super().__init__()
        assert embed_dim > 0 and embed_dim % 2 == 0, (
            f"fourier_emb.embed_dim must be even, got {embed_dim}"
        )
        if not math.isfinite(embed_scale) or embed_scale <= 0:
            raise ValueError("embed_scale must be finite and positive.")
        self.in_features = int(in_features)
        self.embed_scale = float(embed_scale)
        self.embed_dim = int(embed_dim)
        self.kernel = nn.Parameter(
            torch.randn(self.in_features, self.embed_dim // 2) * self.embed_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phase = x @ self.kernel
        return torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)

    def reference_parameter_map(self, scope: str = "FourierEmbs_0"):
        return [(self.kernel, (scope, "kernel"), False)]


class LecunLinear(nn.Module):
    """Standard Dense layer matching Flax nn.Dense (truncated LeCun normal kernel, zero bias)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        sigma = math.sqrt(1.0 / in_features) / 0.87962566103423978
        weight = nn.init.trunc_normal_(
            torch.empty(out_features, in_features), std=sigma, a=-2.0 * sigma, b=2.0 * sigma
        )
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


def make_dense(in_features: int, out_features: int, reparam: Optional[Mapping] = None) -> nn.Module:
    if reparam is None:
        return LecunLinear(in_features, out_features)
    kind = reparam.get("type", "weight_fact")
    if kind != "weight_fact":
        raise ValueError(f"Unsupported reparam type: {kind}")
    mean = float(reparam.get("mean", 1.0))
    stddev = float(reparam.get("stddev", 0.1))
    return DirectFactorizedLinear(in_features, out_features, mean, stddev)


def dense_parameter_map(layer: nn.Module, path_prefix: Tuple[str, ...]):
    if isinstance(layer, DirectFactorizedLinear):
        return [
            (layer.g, (*path_prefix, "kernel", 0), False),
            (layer.v, (*path_prefix, "kernel", 1), True),
            (layer.bias, (*path_prefix, "bias"), False),
        ]
    return [
        (layer.weight, (*path_prefix, "kernel"), True),
        (layer.bias, (*path_prefix, "bias"), False),
    ]


@dataclass(frozen=True)
class ArchConfig:
    arch_name: str = "PirateNet"
    input_dim: int = 4
    out_dim: int = 4
    num_layers: int = 2
    hidden_dim: int = 256
    activation: str = "swish"
    nonlinearity: Union[float, Tuple[float, ...]] = 0.0
    periodicity: Optional[dict] = None
    fourier_emb: Optional[dict] = None
    reparam: Optional[dict] = None

    @property
    def output_dim(self) -> int:
        return self.out_dim


def _build_embeddings(config: ArchConfig):
    cur_dim = int(config.input_dim)
    period_emb = None
    if config.periodicity is not None:
        period_emb = PeriodEmbs(
            period=config.periodicity["period"],
            axis=config.periodicity["axis"],
            trainable=config.periodicity["trainable"],
        )
        cur_dim = period_emb.output_dim(cur_dim)
    fourier_emb = None
    if config.fourier_emb is not None:
        fourier_emb = FourierEmbs(
            in_features=cur_dim,
            embed_scale=config.fourier_emb["embed_scale"],
            embed_dim=config.fourier_emb["embed_dim"],
        )
        cur_dim = fourier_emb.embed_dim
    return period_emb, fourier_emb, cur_dim


class Mlp(nn.Module):
    def __init__(self, config: ArchConfig):
        super().__init__()
        if config.num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        self.config = config
        self.activation_fn = get_activation(config.activation)
        self.period_emb, self.fourier_emb, in_dim = _build_embeddings(config)
        self.hidden = nn.ModuleList(
            make_dense(in_dim if i == 0 else config.hidden_dim, config.hidden_dim, config.reparam)
            for i in range(config.num_layers)
        )
        self.output = make_dense(config.hidden_dim, config.out_dim, config.reparam)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.period_emb is not None:
            x = self.period_emb(x)
        if self.fourier_emb is not None:
            x = self.fourier_emb(x)
        for layer in self.hidden:
            x = self.activation_fn(layer(x))
        return self.output(x)

    def reference_parameter_map(self):
        result = []
        if self.period_emb is not None:
            result.extend(self.period_emb.reference_parameter_map())
        if self.fourier_emb is not None:
            result.extend(self.fourier_emb.reference_parameter_map())
        for i, layer in enumerate([*self.hidden, self.output]):
            result.extend(dense_parameter_map(layer, (f"Dense_{i}",)))
        return result


class ModifiedMlp(nn.Module):
    def __init__(self, config: ArchConfig):
        super().__init__()
        if config.num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        self.config = config
        self.activation_fn = get_activation(config.activation)
        self.period_emb, self.fourier_emb, in_dim = _build_embeddings(config)
        self.encoder_u = make_dense(in_dim, config.hidden_dim, config.reparam)
        self.encoder_v = make_dense(in_dim, config.hidden_dim, config.reparam)
        self.hidden = nn.ModuleList(
            make_dense(in_dim if i == 0 else config.hidden_dim, config.hidden_dim, config.reparam)
            for i in range(config.num_layers)
        )
        self.output = make_dense(config.hidden_dim, config.out_dim, config.reparam)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.period_emb is not None:
            x = self.period_emb(x)
        if self.fourier_emb is not None:
            x = self.fourier_emb(x)
        u = self.activation_fn(self.encoder_u(x))
        v = self.activation_fn(self.encoder_v(x))
        for layer in self.hidden:
            x = self.activation_fn(layer(x))
            x = x * u + (1.0 - x) * v
        return self.output(x)

    def reference_parameter_map(self):
        result = []
        if self.period_emb is not None:
            result.extend(self.period_emb.reference_parameter_map())
        if self.fourier_emb is not None:
            result.extend(self.fourier_emb.reference_parameter_map())
        layers = [self.encoder_u, self.encoder_v, *self.hidden, self.output]
        for i, layer in enumerate(layers):
            result.extend(dense_parameter_map(layer, (f"Dense_{i}",)))
        return result


class PirateBlock(nn.Module):
    """Three-layer residual adaptive block with dual gating and trainable skip alpha."""

    def __init__(
        self,
        hidden_dim: int,
        activation: str,
        nonlinearity: float = 0.0,
        reparam: Optional[Mapping] = None,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.activation_fn = get_activation(activation)
        self.dense_0 = make_dense(self.hidden_dim, self.hidden_dim, reparam)
        self.dense_1 = make_dense(self.hidden_dim, self.hidden_dim, reparam)
        self.dense_2 = make_dense(self.hidden_dim, self.hidden_dim, reparam)
        self.alpha = nn.Parameter(torch.full((1,), float(nonlinearity)))

    def forward(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.activation_fn(self.dense_0(x))
        x = x * u + (1.0 - x) * v
        x = self.activation_fn(self.dense_1(x))
        x = x * u + (1.0 - x) * v
        x = self.activation_fn(self.dense_2(x))
        return self.alpha * x + (1.0 - self.alpha) * identity

    def reference_parameter_map(self, block_scope: str):
        result = []
        for idx, layer in enumerate((self.dense_0, self.dense_1, self.dense_2)):
            result.extend(dense_parameter_map(layer, (block_scope, f"Dense_{idx}")))
        result.append((self.alpha, (block_scope, "alpha"), False))
        return result


class PirateNet(nn.Module):
    """Physics-informed residual adaptive network (PirateNet)."""

    def __init__(self, config: ArchConfig):
        super().__init__()
        if config.num_layers < 1:
            raise ValueError("num_layers (residual blocks) must be >= 1.")
        self.config = config
        self.activation_fn = get_activation(config.activation)
        if isinstance(config.nonlinearity, (int, float)):
            nonlinearities = [float(config.nonlinearity)] * config.num_layers
        else:
            assert len(config.nonlinearity) == config.num_layers
            nonlinearities = [float(v) for v in config.nonlinearity]
        self.nonlinearities = tuple(nonlinearities)
        self.period_emb, self.fourier_emb, in_dim = _build_embeddings(config)
        assert in_dim == config.hidden_dim, (
            f"PirateNet residual blocks require the (embedded) input to have "
            f"hidden_dim={config.hidden_dim} features, got {in_dim}. "
            "Use fourier_emb with embed_dim == hidden_dim."
        )
        self.encoder_u = make_dense(config.hidden_dim, config.hidden_dim, config.reparam)
        self.encoder_v = make_dense(config.hidden_dim, config.hidden_dim, config.reparam)
        self.blocks = nn.ModuleList(
            PirateBlock(
                hidden_dim=config.hidden_dim,
                activation=config.activation,
                nonlinearity=self.nonlinearities[i],
                reparam=config.reparam,
            )
            for i in range(config.num_layers)
        )
        self.output = make_dense(config.hidden_dim, config.out_dim, config.reparam)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.period_emb is not None:
            x = self.period_emb(x)
        if self.fourier_emb is not None:
            x = self.fourier_emb(x)
        u = self.activation_fn(self.encoder_u(x))
        v = self.activation_fn(self.encoder_v(x))
        for block in self.blocks:
            x = block(x, u, v)
        return self.output(x)

    def alphas(self) -> Tuple[float, ...]:
        return tuple(float(block.alpha.detach().cpu().item()) for block in self.blocks)

    def reference_parameter_map(self):
        result = []
        if self.period_emb is not None:
            result.extend(self.period_emb.reference_parameter_map())
        if self.fourier_emb is not None:
            result.extend(self.fourier_emb.reference_parameter_map())
        result.extend(dense_parameter_map(self.encoder_u, ("Dense_0",)))
        result.extend(dense_parameter_map(self.encoder_v, ("Dense_1",)))
        for idx, block in enumerate(self.blocks):
            result.extend(block.reference_parameter_map(f"PirateBlock_{idx}"))
        result.extend(dense_parameter_map(self.output, ("Dense_2",)))
        return result


def create_arch(config: Union[ArchConfig, Mapping]) -> nn.Module:
    if not isinstance(config, ArchConfig):
        data = dict(config)
        if "nonlinearity" in data and isinstance(data["nonlinearity"], list):
            data["nonlinearity"] = tuple(data["nonlinearity"])
        config = ArchConfig(**data)
    name = config.arch_name.lower()
    if name == "mlp":
        return Mlp(config)
    if name == "modifiedmlp":
        return ModifiedMlp(config)
    if name == "piratenet":
        return PirateNet(config)
    raise NotImplementedError(f"Arch {config.arch_name} not supported yet!")


class TimeScaledField(nn.Module):
    """Coordinate wrapper dividing t by t_max (and optional spatial lengths) inside autograd."""

    def __init__(
        self,
        core: nn.Module,
        t_max: float = 1.0,
        lengths: Optional[Sequence[float]] = None,
        prior: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        input_dim = core.config.input_dim
        out_dim = core.config.output_dim
        if lengths is None:
            lengths = (1.0,) * (input_dim - 1)
        if len(lengths) != input_dim - 1:
            raise ValueError("lengths must have input_dim - 1 entries.")
        divisors = (float(t_max), *(float(v) for v in lengths))
        if not all(math.isfinite(v) and v > 0 for v in divisors):
            raise ValueError("Coordinate divisors must be finite and positive.")
        if prior is None:
            prior_tuple = (0.0,) * out_dim
        else:
            if len(prior) != out_dim:
                raise ValueError("prior must match out_dim.")
            prior_tuple = tuple(float(v) for v in prior)
        self.core = core
        self.t_max = float(t_max)
        self.register_buffer("divisors", torch.tensor(divisors))
        self.register_buffer("prior", torch.tensor(prior_tuple))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.core(q / self.divisors) + self.prior


class SphereField(nn.Module):
    """Dimensionless q=(t,x,y,z) -> (u,v,w,p) with free-stream prior (1,0,0,0)."""

    def __init__(self, core: nn.Module, duration: float = 1.0, lengths: Sequence[float] = (32.0, 16.0, 16.0)):
        super().__init__()
        if core.config.input_dim != 4 or core.config.output_dim != 4:
            raise ValueError("SphereField requires four coordinates and four output components.")
        if not all(math.isfinite(v) and v > 0 for v in (duration, *lengths)):
            raise ValueError("Coordinate divisors must be finite and positive.")
        self.core = core
        self.register_buffer("divisors", torch.tensor((float(duration), *(float(v) for v in lengths))))
        self.register_buffer("prior", torch.tensor((1.0, 0.0, 0.0, 0.0)))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.core(q / self.divisors) + self.prior
