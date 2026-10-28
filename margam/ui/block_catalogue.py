"""The currently insertable UI blocks; no plugin or solver registration system."""

# Category, then (display title, existing template kind).
BLOCK_CATEGORIES = {
    'Geometry': (('Rectangle', 'domain'),),
    'Sampling': (('Interior sampler', 'interior'), ('Boundary points', 'boundary'), ('Residual refinement (RAR)','rar'), ('Residual importance sampling','importance')),
    'Neural networks': (('Neural network', 'network'),),
    'Loss / Physics': (('PDE residual', 'pde'), ('Predefined PDE','predefined_pde'), ('Boundary condition', 'bc')),
    'PINN recipes': (('Baseline PINN','recipe'), ('Self-adaptive PINN','sa_recipe'), ('Learning-rate annealing PINN','lra_recipe')),
    'Training': (('PINN method', 'method'), ('Self-adaptive trainer','sa_method'), ('LRA trainer','lra_method')),
    'Results': (('Solution Plot', 'solution'), ('Training Monitor', 'monitor'),
                ('Reference', 'reference'), ('Comparison', 'comparison'), ('CSV Export','export'), ('ParaView Export','paraview')),
}
KINDS = {title: kind for entries in BLOCK_CATEGORIES.values() for title, kind in entries}
BLOCK_GROUPS = {group: tuple(title for title, _ in entries)
                for group, entries in BLOCK_CATEGORIES.items()}
BLOCK_MIME = 'application/x-margam-block'


def dropped_block(mime):
    """Accept a known title only, never files, arbitrary text or serialised code."""
    if not mime.hasFormat(BLOCK_MIME):
        return None
    data = mime.data(BLOCK_MIME)
    if len(data) > 100:
        return None
    try:
        title = bytes(data).decode('utf-8')
    except UnicodeDecodeError:
        return None
    return title if title in KINDS else None


# Concrete capabilities for the current editor, shared by library, inspector and Help.
CAPABILITIES = {
    'lra_recipe': ('LRA recipe · compact or expanded', 'Existing pointwise network plus gradient-balanced Adam trainer. One coefficient per named constraint, max-PDE/mean-constraint gradient ratio and EMA; no paper architecture or base-rate decay.'),
    'lra_method': ('Learning-rate annealing loss balancing', 'Wang et al. supplied v1 Algorithm 1: raw constraint-gradient denominator, EMA alpha is the new-estimate fraction. PDE weight fixed; each constraint gets a scalar coefficient. Not per-point attention or an Adam-rate scheduler.'),
    'sa_recipe': ('Self-adaptive recipe · compact or expanded', 'Per-point squared masks; Adam descent for the network and ascent for masks. Fixed samples; PDE adaptation by default; optional BC adaptation. No GP resampling or L-BFGS phase.'),
    'sa_method': ('Pointwise adaptive weights and Adam min/max', 'Independent positive initial mask weights and ascent learning rate; optional BC masks. Squared mask, finite cap. Raw term losses plus adaptive total; increasing total is not an accuracy metric.'),
    'recipe': ('Baseline recipe · compact or expanded', 'Contains a replaceable tanh network and full-batch Adam trainer. Exposes steps, learning rate and device. Connect PDE and BC inputs; double-click to expand. Collapse a baseline trainer and its private network to retain edits.'),
    'domain': ('Bounds and shape', 'Interval, axis-aligned rectangle, or a 1D space–time slab. Advanced CAD belongs to GeoSampler.'),
    'rar': ('Initial sampler, candidate pool, schedule, additions and cap', 'Adds the largest-residual candidate points after warm-up. Built-in 1D/2D domains; baseline, hard-BC and LRA trainers. BC points stay fixed. Imported interiors and SA point masks are not supported.'),
    'importance': ('Initial sampler/budget, candidates, schedule and probability mixture', 'Fixed initial point count; residual-based draws with replacement and inverse-probability MSE weights. Built-in 1D/2D domains with baseline, hard-BC or LRA; no imported interiors or SA masks.'),
    'interior': ('Method, budget, seed', 'Fixed meshless PDE points. Grid may use fewer than the budget; other implemented methods return the exact count.'),
    'boundary': ('Named side, method, count, seed', 'Independent rectangle-side recipes; interval endpoints are fixed. Space–time has left/right traces and an initial set, with no final-time constraint. BC assignment is separate.'),
    'predefined_pde': ('PDE family, coefficient, forcing/diffusivity', 'Tested equation presets only. Alternative to PDE residual; BCs and references remain separate.'),
    'pde': ('Physics, residual expressions, viscosity/coefficient', 'Scalar spatial/1D space–time residual or steady 2D incompressible u/v/p. No transient flow system or extra pointwise inputs.'),
    'bc': ('Type, constant/expression target, custom operator', 'Dirichlet, outward derivative or Custom (e.g. Robin). A Dirichlet anchor is required.'),
    'network': ('Architecture, hidden layers, seed and per-architecture options', 'Pointwise fully connected, Fourier MLP/features, highway/modified/multiscale Fourier, DGM, SIREN and multiplicative Fourier/Gabor networks. Shared by baseline and SA. No neural operators.'),
    'method': ('Variant, loss formulation, updates, learning rate, loss weights, device', 'Baseline or equal-constant hard Dirichlet PINN; mean MSE or domain/boundary integral loss; full-batch Adam. Optional residual sampling blocks; no LBFGS.'),
    'solution': ('Name, evaluation points per axis', 'Explicit scalar prediction or u/v/p table request; no reference assumed.'),
    'monitor': ('Name, retained updates', 'Explicit bounded loss recording; closing its view does not stop recording.'),
    'reference': ('Preset, coordinate expression or mapped values', 'Explicit scalar source. Imported data requires matching coordinates; preset sources reject custom physics.'),
    'paraview': ('Name, regular-grid evaluation resolution', 'VTI predictions on regular line/rectangle grids: scalar fields or u/v/p plus a velocity vector. Export a snapshot on request. Evaluation resolution is independent of training samples.'),
    'export': ('Name, evaluation points per axis', 'Explicit scalar/u-v-p data request. Export CSV only after training and on user action; no plot window.'),
    'comparison': ('Name, evaluation points per axis', 'Requires a PINN run and an explicit reference matching coordinates and scalar/u-v-p fields.'),
}


def describe_block(kind):
    from .ports import INPUTS,OUTPUTS
    parameters,limits=CAPABILITIES[kind]
    inputs=', '.join(spec.label for spec in INPUTS[kind]) or 'none'
    output=OUTPUTS[kind]
    return ('Implemented · '+parameters+'\nInputs: '+inputs+'\nOutput: '+(output.label if output else 'none')+'\n'+limits)
