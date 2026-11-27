"""Independent Finite Basis PINN (FBPINN) with multilevel and progressive modes.

Based on:
- Moseley, Markham & Nissen-Meyer, Adv. Comput. Math. 49, 62 (2023), arXiv:2107.07871
- Dolean, Heinlein, Mishra & Moseley, CMAME 429, 117116 (2024), arXiv:2306.05486
"""
from copy import deepcopy
from itertools import product
import math
import torch
import torch.nn as nn

from .pinn import PINN
from ..components.fbpinn_config import (
    WINDOW_TYPES,
    active_subdomain_counts,
    fbpinn_defaults,
    resolve_schedule_axis,
    spatial_time_axes,
    validate_fbpinn_settings,
)
from ..components.losses import loss_record_keys, weighted_loss

STATE_INACTIVE = 0
STATE_ACTIVE = 1
STATE_FIXED = 2


def subdomain_boxes(spatial_bounds, counts, overlap, param_bounds=((), ())):
    """Build cell-centered overlapping boxes and clipped local normalization bounds."""
    lower, upper = spatial_bounds
    p_lo, p_hi = param_bounds
    d_st = len(lower)
    if len(upper) != d_st or len(counts) != d_st:
        raise ValueError('Subdomain counts must match the spatial/time domain dimension.')
    axis_specs = []
    for k in range(d_st):
        lo, hi = float(lower[k]), float(upper[k])
        n = int(counts[k])
        width = (hi - lo) / n
        half = 0.5 * float(overlap) * width
        entries = []
        for idx in range(n):
            center = lo + (idx + 0.5) * width
            supp_lo = center - half
            supp_hi = center + half
            phys_lo = max(lo, supp_lo)
            phys_hi = min(hi, supp_hi)
            entries.append((center, half, supp_lo, supp_hi, phys_lo, phys_hi))
        axis_specs.append(entries)
    boxes = []
    for multi in product(*(range(int(n)) for n in counts)):
        centers = [axis_specs[k][multi[k]][0] for k in range(d_st)]
        half_widths = [axis_specs[k][multi[k]][1] for k in range(d_st)]
        supp_lo = [axis_specs[k][multi[k]][2] for k in range(d_st)]
        supp_hi = [axis_specs[k][multi[k]][3] for k in range(d_st)]
        net_lo = [axis_specs[k][multi[k]][4] for k in range(d_st)] + [float(v) for v in p_lo]
        net_hi = [axis_specs[k][multi[k]][5] for k in range(d_st)] + [float(v) for v in p_hi]
        boxes.append(dict(
            multi_index=multi,
            center=centers,
            half_width=half_widths,
            support_lo=supp_lo,
            support_hi=supp_hi,
            network_bounds=(net_lo, net_hi),
        ))
    return boxes


class FiniteBasisNetwork(nn.Module):
    """Overlapping partition-of-unity domain decomposition network.

    Combines locally normalized subdomain networks through smooth C^2, C^inf, or
    C^1 partition-of-unity windows, with an optional Level-0 global coarse network.
    """

    def __init__(
        self,
        subdomains,
        boxes,
        *,
        counts,
        overlap=1.5,
        window='cosine_sq',
        window_sigma=0.05,
        output_dim=1,
        coarse_network=None,
    ):
        super().__init__()
        if not subdomains or len(subdomains) != len(boxes):
            raise ValueError('FiniteBasisNetwork requires one sub-network per subdomain box.')
        if window not in WINDOW_TYPES:
            raise ValueError('Unsupported FBPINN window type.')
        self.subdomains = nn.ModuleList(subdomains)
        self.coarse_network = coarse_network
        self.counts = tuple(int(c) for c in counts)
        self.num_subdomains = len(subdomains)
        self.d_st = len(boxes[0]['center'])
        self.output_dim = int(output_dim)
        self.overlap = float(overlap)
        self.window = window
        self.window_sigma = float(window_sigma)
        self.multilevel = coarse_network is not None
        self.multi_indices = tuple(tuple(box['multi_index']) for box in boxes)

        dtype = next(subdomains[0].parameters()).dtype
        self.register_buffer('centers', torch.tensor([b['center'] for b in boxes], dtype=dtype))
        self.register_buffer('half_widths', torch.tensor([b['half_width'] for b in boxes], dtype=dtype))
        self.register_buffer('support_lo', torch.tensor([b['support_lo'] for b in boxes], dtype=dtype))
        self.register_buffer('support_hi', torch.tensor([b['support_hi'] for b in boxes], dtype=dtype))
        self.register_buffer('subdomain_state', torch.ones(len(subdomains), dtype=torch.int64))
        self._states_cache = (STATE_ACTIVE,) * len(subdomains)
        self._fast_fcnn = self._detect_fast_fcnn(subdomains)

    @staticmethod
    def _detect_fast_fcnn(subdomains):
        from ..nn.FCNN import FullyConnected
        first = subdomains[0]
        if not isinstance(first, FullyConnected) or first.input_lower is None or first.input_upper is None:
            return False
        if not all(type(layer) is nn.Linear for layer in first.linears):
            return False
        ref_sizes = tuple(first.layer_size)
        ref_act = first.activation
        same_act = (
            (lambda a: type(a) is type(ref_act) and not list(a.parameters()))
            if isinstance(ref_act, nn.Module)
            else (lambda a: a is ref_act)
        )
        for net in subdomains[1:]:
            if (
                not isinstance(net, FullyConnected)
                or net.input_lower is None
                or net.input_upper is None
                or tuple(net.layer_size) != ref_sizes
                or not same_act(net.activation)
                or not all(type(layer) is nn.Linear for layer in net.linears)
            ):
                return False
        return True

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self._states_cache = tuple(int(v) for v in self.subdomain_state.detach().cpu().tolist())

    def set_subdomain_states(self, states):
        if len(states) != self.num_subdomains or any(int(v) not in (STATE_INACTIVE, STATE_ACTIVE, STATE_FIXED) for v in states):
            raise ValueError('Invalid FBPINN subdomain states.')
        self._states_cache = tuple(int(v) for v in states)
        with torch.no_grad():
            self.subdomain_state.copy_(torch.tensor(self._states_cache, dtype=torch.int64, device=self.subdomain_state.device))

    def unnormalized_windows(self, x_st):
        s = (x_st.unsqueeze(1) - self.centers.unsqueeze(0)) / self.half_widths.unsqueeze(0)
        if self.window == 'smooth_sigmoid':
            scale = 2.0 * self.window_sigma
            phi = torch.sigmoid((s + 1.0) / scale) * torch.sigmoid((1.0 - s) / scale)
            return phi.prod(dim=-1)
        inside = s.abs() < 1.0
        s_safe = s.clamp(-1.0, 1.0)
        base = 0.5 * (1.0 + torch.cos(math.pi * s_safe))
        if self.window == 'cosine_sq':
            base = base.square()
        phi = torch.where(inside, base, torch.zeros_like(s))
        return phi.prod(dim=-1)

    def normalized_windows(self, x_st):
        raw = self.unnormalized_windows(x_st)
        included = (self.subdomain_state != STATE_INACTIVE).to(raw.dtype).unsqueeze(0)
        active_raw = raw * included
        denom = active_raw.sum(dim=1, keepdim=True)
        has_support = denom > 0.0
        denom_safe = torch.where(has_support, denom, torch.ones_like(denom))
        norm = torch.where(has_support, active_raw / denom_safe, torch.zeros_like(active_raw))
        return raw, norm

    def _forward_batched_fcnn(self, x, w_hat, inc_indices):
        nets = [self.subdomains[j] for j in inc_indices]
        lo = torch.stack([net.input_lower for net in nets], dim=0)
        hi = torch.stack([net.input_upper for net in nets], dim=0)
        h = (x.unsqueeze(1) - lo.unsqueeze(0)) / (hi - lo).unsqueeze(0)
        n_layers = len(nets[0].linears)
        act = nets[0].activation
        for layer_idx in range(n_layers):
            weight = torch.stack([net.linears[layer_idx].weight for net in nets], dim=0)
            bias = torch.stack([net.linears[layer_idx].bias for net in nets], dim=0)
            h = torch.einsum('nki,koi->nko', h, weight) + bias.unsqueeze(0)
            if layer_idx < n_layers - 1:
                h = act(h)
        w_sel = w_hat if len(inc_indices) == self.num_subdomains else w_hat[:, inc_indices]
        return (h * w_sel.unsqueeze(-1)).sum(dim=1)

    def forward(self, x):
        states = self._states_cache
        if self.num_subdomains == 1 and self.coarse_network is None:
            if states[0] == STATE_INACTIVE:
                return x.new_zeros((x.shape[0], self.output_dim))
            return self.subdomains[0](x)

        x_st = x[:, :self.d_st]
        w_raw, w_hat = self.normalized_windows(x_st)
        inc_indices = [j for j, st in enumerate(states) if st != STATE_INACTIVE]
        if not inc_indices:
            out = x.new_zeros((x.shape[0], self.output_dim))
        elif self._fast_fcnn:
            out = self._forward_batched_fcnn(x, w_hat, inc_indices)
        else:
            compact_cpu = (self.window in ('cosine_sq', 'cosine')) and (not x.is_cuda)
            out = x.new_zeros((x.shape[0], self.output_dim))
            for j in inc_indices:
                subnet = self.subdomains[j]
                if compact_cpu:
                    idx = torch.nonzero(w_raw[:, j] > 0.0, as_tuple=False).squeeze(1)
                    if idx.numel() == 0:
                        continue
                    x_sub = x.index_select(0, idx)
                    w_sub = w_hat.index_select(0, idx)[:, j:j + 1]
                    out = out.index_add(0, idx, subnet(x_sub) * w_sub)
                else:
                    out = out + subnet(x) * w_hat[:, j:j + 1]

        if self.coarse_network is not None:
            out = 0.5 * (self.coarse_network(x) + out)
        return out


def build_fbpinn_network(config, single_builder):
    """Build a FiniteBasisNetwork using the shared single-network architecture factory."""
    settings = config.fbpinn_settings
    validate_fbpinn_settings(settings)
    st_bounds = getattr(config, 'spatial_time_bounds', config.bounds)
    d_st = len(st_bounds[0])
    p_bounds = (config.bounds[0][d_st:], config.bounds[1][d_st:])
    counts = active_subdomain_counts(config)
    boxes = subdomain_boxes(st_bounds, counts, settings['overlap'], p_bounds)
    base_seed = config.seed if config.network_seed is None else config.network_seed
    subdomains = []
    base_spec = None
    for idx, box in enumerate(boxes):
        net, spec = single_builder(box['network_bounds'], (base_seed + idx) % (2 ** 31))
        if base_spec is None:
            base_spec = spec
        subdomains.append(net)
    coarse = None
    if settings['multilevel']:
        coarse, _ = single_builder(config.bounds, (base_seed + len(boxes)) % (2 ** 31))
    network = FiniteBasisNetwork(
        subdomains,
        boxes,
        counts=counts,
        overlap=settings['overlap'],
        window=settings['window'],
        window_sigma=settings['window_sigma'],
        output_dim=len(config.field_names),
        coarse_network=coarse,
    )
    network._margam_spec = deepcopy(dict(
        method='fbpinn',
        base_type=base_spec['type'],
        base_options=base_spec['options'],
        layers=list(base_spec['layers']),
        inputs=base_spec['inputs'],
        outputs=base_spec['outputs'],
        bounds=[list(row) for row in config.bounds],
        spatial_time_dim=d_st,
        subdomains=list(counts),
        overlap=float(settings['overlap']),
        window=settings['window'],
        window_sigma=float(settings['window_sigma']),
        multilevel=bool(settings['multilevel']),
        seed=base_seed,
    ))
    return network


class FiniteBasisPINN(PINN):
    """Finite Basis PINN trainer supporting simultaneous, multilevel, and progressive modes."""

    def __init__(self, problem, network, *, settings=None, axes=None, weights=None, configuration=None):
        super().__init__(problem, network, weights=weights, loss_formulation='mean')
        if not isinstance(network, FiniteBasisNetwork):
            raise ValueError('FiniteBasisPINN requires a FiniteBasisNetwork.')
        self.settings = deepcopy(fbpinn_defaults() if settings is None else settings)
        validate_fbpinn_settings(self.settings)
        self.axes = tuple(axes) if axes is not None else tuple(('x', 'y', 'z', 't')[:network.d_st])
        if len(self.axes) != network.d_st:
            raise ValueError('FBPINN axis names must match the decomposed domain dimension.')
        self.configuration = deepcopy(configuration or {})
        for key in ('iterations', 'device'):
            self.configuration.pop(key, None)
        self.schedule_axis_index = None
        self.schedule_axis_name = None
        self.total_slices = 1
        self._subdomain_slices = [0] * network.num_subdomains

    def _resolve_schedule(self):
        if self.settings['schedule'] != 'progressive':
            self.schedule_axis_index = None
            self.schedule_axis_name = None
            self.total_slices = 1
            self._subdomain_slices = [0] * self.network.num_subdomains
            return
        if self.network.multilevel:
            raise ValueError('Multilevel coarse-to-fine FBPINN requires simultaneous training.')
        choice = self.settings['schedule_axis']
        counts = self.network.counts
        if choice == 'auto':
            if 't' in self.axes and counts[self.axes.index('t')] >= 2:
                idx = self.axes.index('t')
            else:
                idx = next((i for i, count in enumerate(counts) if count >= 2), None)
            if idx is None:
                raise ValueError('Progressive FBPINN marching requires at least 2 subdomains along an active axis.')
            name = self.axes[idx]
        else:
            if choice not in self.axes:
                raise ValueError(f'FBPINN schedule axis {choice!r} is not present in this domain.')
            idx = self.axes.index(choice)
            if counts[idx] < 2:
                raise ValueError(f'Progressive FBPINN marching along {choice} requires at least 2 subdomains on that axis.')
            name = choice
        self.schedule_axis_index = idx
        self.schedule_axis_name = name
        self.total_slices = counts[idx]
        self._subdomain_slices = [multi[idx] for multi in self.network.multi_indices]

    def _point_membership(self, points):
        """Return (N, M) bool tensor of points inside each subdomain's support box."""
        x_st = points[:, :self.network.d_st]
        eps = 1e-12 if self.dtype == torch.float64 else 1e-6
        lo = self.network.support_lo.unsqueeze(0) - eps
        hi = self.network.support_hi.unsqueeze(0) + eps
        return ((x_st.unsqueeze(1) >= lo) & (x_st.unsqueeze(1) <= hi)).all(dim=-1)

    def _sync_requires_grad(self):
        states = self.network.subdomain_state.tolist()
        for j, subnet in enumerate(self.network.subdomains):
            trainable = states[j] == STATE_ACTIVE
            for param in subnet.parameters():
                param.requires_grad_(trainable)
        if self.network.coarse_network is not None:
            for param in self.network.coarse_network.parameters():
                param.requires_grad_(True)

    def _apply_schedule(self, step):
        if self.settings['schedule'] == 'simultaneous':
            self.network.set_subdomain_states([STATE_ACTIVE] * self.network.num_subdomains)
            self._sync_requires_grad()
            return None
        frontier = min(self.total_slices - 1, step // self.settings['active_steps'])
        freeze = self.settings['freeze_completed']
        states = []
        for g in self._subdomain_slices:
            if g > frontier:
                states.append(STATE_INACTIVE)
            elif g == frontier:
                states.append(STATE_ACTIVE)
            else:
                states.append(STATE_FIXED if freeze else STATE_ACTIVE)
        self.network.set_subdomain_states(states)
        self._sync_requires_grad()
        return frontier

    def compile(self, optimiser_string='adam', lr=1e-3, metrics_string='MSE', device='auto', *, optimiser=None):
        if optimiser_string != 'adam' or metrics_string != 'MSE' or optimiser is not None or self.loss_formulation != 'mean':
            raise ValueError('FBPINN currently uses Adam and mean squared losses.')
        self._fbpinn_fingerprint = None
        self._resolve_schedule()
        # Ensure all parameters are trainable when Adam registers them.
        self.network.set_subdomain_states([STATE_ACTIVE] * self.network.num_subdomains)
        self._sync_requires_grad()
        super().compile(optimiser_string, lr, metrics_string, device)
        self._fixed_fingerprint = super().data_fingerprint()

        # Device-independent CPU fingerprint for portable CPU/CUDA checkpoints.
        saved = (self.collocation_point_sample, self.collocation_point_labels, self._constraints)
        def cpu_target(source, pts):
            return torch.as_tensor(source(pts) if callable(source) else source, dtype=self.dtype, device='cpu').detach()
        try:
            self.collocation_point_sample = saved[0].detach().cpu()
            self.collocation_point_labels = cpu_target(self.problem.residual_target, self.collocation_point_sample)
            self._constraints = [(c, pts.detach().cpu(), cpu_target(c.target, pts.detach().cpu())) for c, pts, _ in saved[2]]
            self._fbpinn_fingerprint = super().data_fingerprint()
        finally:
            self.collocation_point_sample, self.collocation_point_labels, self._constraints = saved

        self._pde_membership = self._point_membership(self.collocation_point_sample)
        self._bc_memberships = [self._point_membership(pts) for _, pts, _ in self._constraints]

        self._slice_pde_masks = []
        self._slice_bc_masks = []
        if self.settings['schedule'] == 'progressive':
            for g in range(self.total_slices):
                sub_ids = [j for j, sl in enumerate(self._subdomain_slices) if sl == g]
                pde_mask = self._pde_membership[:, sub_ids].any(dim=1)
                if not pde_mask.any().item():
                    raise ValueError(
                        f'Progressive FBPINN slice {g + 1}/{self.total_slices} along {self.schedule_axis_name} '
                        'contains zero PDE collocation points; increase interior points or reduce subdomains.'
                    )
                self._slice_pde_masks.append(pde_mask)
                self._slice_bc_masks.append([m[:, sub_ids].any(dim=1) for m in self._bc_memberships])

        self._apply_schedule(0)
        self.signature = dict(
            algorithm='moseley2023-fbpinn-v1',
            settings=deepcopy(self.settings),
            axes=list(self.axes),
            counts=list(self.network.counts),
            weights=deepcopy(self._configured_weights),
            configuration=deepcopy(self.configuration),
        )
        return self

    def data_fingerprint(self):
        return self._fbpinn_fingerprint if self._fbpinn_fingerprint is not None else super().data_fingerprint()

    def _slice_target(self, target, mask, total_rows):
        if target.ndim >= 1 and target.shape[0] == total_rows:
            return target[mask]
        return target

    def _progressive_losses(self, frontier):
        if self.settings['freeze_completed']:
            pde_mask = self._slice_pde_masks[frontier]
            bc_masks = self._slice_bc_masks[frontier]
        else:
            pde_mask = self._slice_pde_masks[0].clone()
            for g in range(1, frontier + 1):
                pde_mask |= self._slice_pde_masks[g]
            bc_masks = []
            for idx in range(len(self._constraints)):
                m = self._slice_bc_masks[0][idx].clone()
                for g in range(1, frontier + 1):
                    m |= self._slice_bc_masks[g][idx]
                bc_masks.append(m)

        active_points = self.collocation_point_sample[pde_mask].detach().requires_grad_(True)
        active_labels = self._slice_target(self.collocation_point_labels, pde_mask, len(self.collocation_point_sample))
        residual = self.problem.residual(active_points, self.network(active_points))
        terms = {'pde': self._term(residual, active_labels, len(active_points))}

        for idx, (condition, sample, target) in enumerate(self._constraints):
            mask = bc_masks[idx]
            if not mask.any().item():
                terms[condition.name] = self.collocation_point_sample.new_zeros(())
                continue
            pts = sample[mask].detach().requires_grad_(condition.operator is not None)
            values = self.network(pts)
            if condition.operator is not None:
                values = condition.operator(pts, values)
            tgt = self._slice_target(target, mask, len(sample))
            terms[condition.name] = self._term(values, tgt, len(pts))
        return terms

    def train(self, iterations, display_every=1):
        self._require_compiled()
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
            raise ValueError('iterations must be a non-negative integer.')
        if isinstance(display_every, bool) or not isinstance(display_every, int) or display_every < 0:
            raise ValueError('display_every must be a non-negative integer.')
        if self.loss_formulation != 'mean' or self.sampling_controller is not None or super().data_fingerprint() != self._fixed_fingerprint:
            raise ValueError('FBPINN currently requires fixed points and mean MSE; resampling is not supported.')
        self.network.train()
        for _ in range(iterations):
            frontier = self._apply_schedule(self.iter)
            self.optimiser.zero_grad(set_to_none=True)
            terms = self.compute_losses() if frontier is None else self._progressive_losses(frontier)
            total = weighted_loss(terms, self.weights)
            if any(not torch.isfinite(value).item() for value in [*terms.values(), total]):
                raise FloatingPointError('Non-finite FBPINN loss; no optimiser step was taken.')
            total.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.network.parameters()):
                self.optimiser.zero_grad(set_to_none=True)
                raise FloatingPointError('Non-finite FBPINN gradient; no optimiser step was taken.')
            self.optimiser.step()
            self.iter += 1
            record = {name: loss.detach().item() for name, loss in terms.items()}
            record['total'] = total.detach().item()
            self.record_contributions(record, terms)
            self.loss_history.append(record)
            self.training_history.append(record['total'])
            self.PDE_loss = terms['pde'].detach()
            self.BC_loss = sum(loss.detach() for name, loss in terms.items() if name != 'pde')
            self.total_loss = total.detach()
            if display_every and self.iter % display_every == 0:
                print(f'Iteration: {self.iter} FBPINN loss: {record["total"]:.6g}')
        return self.training_history

    def fbpinn_summary(self):
        self._require_compiled()
        states = self.network.subdomain_state.tolist()
        labels = {STATE_INACTIVE: 'inactive', STATE_ACTIVE: 'active', STATE_FIXED: 'fixed'}
        was_training = self.network.training
        per_subdomain = []
        try:
            self.network.eval()
            included_ids = [j for j, s in enumerate(states) if s != STATE_INACTIVE]
            covered_mask = self._pde_membership[:, included_ids].any(dim=1) if included_ids else torch.zeros(len(self.collocation_point_sample), dtype=torch.bool, device=self.device)
            point_sq = torch.zeros(len(self.collocation_point_sample), dtype=self.dtype, device=self.device)
            if covered_mask.any().item():
                with torch.enable_grad():
                    points = self.collocation_point_sample[covered_mask].detach().requires_grad_(True)
                    residual = self.problem.residual(points, self.network(points))
                    covered_labels = self._slice_target(self.collocation_point_labels, covered_mask, len(self.collocation_point_sample))
                    values, target = self._aligned(residual, covered_labels, len(points))
                    point_sq[covered_mask] = (values - target).detach().square().mean(dim=1)
            for j in range(self.network.num_subdomains):
                mask = self._pde_membership[:, j]
                count = int(mask.sum().item())
                rmse = math.sqrt(float(point_sq[mask].mean().item())) if count > 0 and states[j] != STATE_INACTIVE else None
                per_subdomain.append(dict(
                    index=j,
                    multi_index=list(self.network.multi_indices[j]),
                    state=labels[states[j]],
                    points=count,
                    pde_rmse=rmse,
                ))
        finally:
            self.network.train(was_training)
        return dict(
            step=self.iter,
            subdomains=list(self.network.counts),
            axes=list(self.axes),
            total_subdomains=self.network.num_subdomains,
            total_networks=self.network.num_subdomains + (1 if self.network.multilevel else 0),
            trainable_parameters=sum(p.numel() for p in self.network.parameters()),
            window=self.network.window,
            overlap=float(self.network.overlap),
            multilevel=bool(self.network.multilevel),
            schedule=self.settings['schedule'],
            schedule_axis=self.schedule_axis_name,
            states=dict(
                active=sum(s == STATE_ACTIVE for s in states),
                fixed=sum(s == STATE_FIXED for s in states),
                inactive=sum(s == STATE_INACTIVE for s in states),
            ),
            subdomain_residuals=per_subdomain,
        )

    def training_state(self):
        state = super().training_state()
        state['fbpinn'] = deepcopy(dict(
            version=1,
            signature=self.signature,
            subdomain_state=self.network.subdomain_state.detach().cpu().tolist(),
        ))
        return state

    def load_training_state(self, state):
        self._require_compiled()
        if not isinstance(state, dict):
            raise ValueError('Invalid FBPINN checkpoint.')
        payload = state.get('fbpinn')
        if not isinstance(payload, dict) or set(payload) != {'version', 'signature', 'subdomain_state'}:
            raise ValueError('Invalid FBPINN checkpoint metadata.')
        if type(payload['version']) is not int or payload['version'] != 1:
            raise ValueError('Unsupported FBPINN checkpoint version.')
        if payload['signature'] != self.signature:
            raise ValueError('FBPINN checkpoint configuration differs from the current recipe.')
        sub_state = payload['subdomain_state']
        if (
            not isinstance(sub_state, list)
            or len(sub_state) != self.network.num_subdomains
            or any(type(v) is not int or v not in (STATE_INACTIVE, STATE_ACTIVE, STATE_FIXED) for v in sub_state)
        ):
            raise ValueError('Invalid FBPINN subdomain state in checkpoint.')
        net_weights = state.get('network')
        if not isinstance(net_weights, dict) or 'subdomain_state' not in net_weights:
            raise ValueError('FBPINN checkpoint network state is missing subdomain states.')
        buf = net_weights['subdomain_state']
        if not isinstance(buf, torch.Tensor) or buf.dtype != torch.int64 or buf.tolist() != sub_state:
            raise ValueError('FBPINN checkpoint subdomain state buffer disagrees with metadata.')

        names = {'pde', 'total', *(c.name for c, _, _ in self._constraints)}
        constraint_names = [c.name for c, _, _ in self._constraints]
        history = state.get('loss_history')
        totals = state.get('training_history')
        steps = state.get('iter')
        if (
            type(steps) is not int
            or not 0 <= steps <= 100000
            or not isinstance(history, list)
            or len(history) != steps
            or not isinstance(totals, list)
            or len(totals) != steps
            or any(
                not isinstance(row, dict)
                or not loss_record_keys(row, names, constraint_names)
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in row.values())
                for row in history
            )
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in totals)
        ):
            raise ValueError('Invalid FBPINN checkpoint history.')

        # Snapshot full trainer state for atomic rollback on any failure.
        snapshot_network = deepcopy(self.network.state_dict())
        snapshot_optimiser = deepcopy(self.optimiser.state_dict())
        snapshot_iter = self.iter
        snapshot_loss = deepcopy(self.loss_history)
        snapshot_training = deepcopy(self.training_history)
        snapshot_pde = getattr(self, 'PDE_loss', None)
        snapshot_bc = getattr(self, 'BC_loss', None)
        snapshot_total = getattr(self, 'total_loss', None)
        snapshot_grads = [p.requires_grad for p in self.network.parameters()]
        try:
            super().load_training_state({key: value for key, value in state.items() if key != 'fbpinn'})
            self.network.set_subdomain_states(sub_state)
            self._sync_requires_grad()
        except Exception:
            self.network.load_state_dict(snapshot_network)
            self.optimiser.load_state_dict(snapshot_optimiser)
            self.iter = snapshot_iter
            self.loss_history = snapshot_loss
            self.training_history = snapshot_training
            if snapshot_pde is not None:
                self.PDE_loss = snapshot_pde
            if snapshot_bc is not None:
                self.BC_loss = snapshot_bc
            if snapshot_total is not None:
                self.total_loss = snapshot_total
            for param, req in zip(self.network.parameters(), snapshot_grads):
                param.requires_grad_(req)
            raise
        return self
