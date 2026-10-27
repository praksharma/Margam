"""The currently insertable UI blocks; no plugin or solver registration system."""

# Category, then (display title, existing template kind).
BLOCK_CATEGORIES = {
    'Geometry': (('Rectangle', 'domain'),),
    'Sampling': (('Interior sampler', 'interior'), ('Boundary points', 'boundary')),
    'Neural networks': (('Neural network', 'network'),),
    'Loss / Physics': (('PDE residual', 'pde'), ('Predefined PDE','predefined_pde'), ('Boundary condition', 'bc')),
    'Optimisation / PINN': (('PINN method', 'method'), ('Baseline PINN','recipe'), ('Self-adaptive PINN','sa_recipe'), ('Self-adaptive trainer','sa_method')),
    'Results': (('Solution Plot', 'solution'), ('Training Monitor', 'monitor'),
                ('Reference', 'reference'), ('Comparison', 'comparison'), ('CSV Export','export')),
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
    'sa_recipe': ('Self-adaptive recipe · compact or expanded', 'Per-point squared masks; Adam descent for the network and ascent for masks. Fixed samples; PDE adaptation by default; optional BC adaptation. No GP resampling or L-BFGS phase.'),
    'sa_method': ('Pointwise adaptive weights and Adam min/max', 'Independent positive initial mask weights and ascent learning rate; optional BC masks. Squared mask, finite cap. Raw term losses plus adaptive total; increasing total is not an accuracy metric.'),
    'recipe': ('Baseline recipe · compact or expanded', 'Contains a replaceable tanh network and full-batch Adam trainer. Exposes steps, learning rate and device. Connect PDE and BC inputs; double-click to expand. Collapse a baseline trainer and its private network to retain edits.'),
    'domain': ('Bounds and shape', 'Interval or axis-aligned rectangle. Advanced CAD belongs to GeoSampler.'),
    'interior': ('Method, budget, seed', 'Fixed meshless PDE points. Grid may use fewer than the budget; other implemented methods return the exact count.'),
    'boundary': ('Named side, method, count, seed', 'Independent rectangle-side recipes; interval endpoints are fixed. BC assignment is separate.'),
    'predefined_pde': ('PDE family, coefficient, forcing/diffusivity', 'Tested equation presets only. Alternative to PDE residual; BCs and references remain separate.'),
    'pde': ('Physics, residual expressions, viscosity/coefficient', 'Scalar residual or steady 2D incompressible u/v/p. No transient system or extra pointwise inputs.'),
    'bc': ('Type, constant/expression target, custom operator', 'Dirichlet, outward derivative or Custom (e.g. Robin). A Dirichlet anchor is required.'),
    'network': ('Hidden-layer widths, initialisation seed', 'Fully connected tanh / Xavier normal, fixed domain normalisation. No architecture selector.'),
    'method': ('Variant, updates, learning rate, loss weights, device', 'Baseline or equal-constant hard Dirichlet PINN; fixed-sample full-batch Adam. No LBFGS or adaptive sampling.'),
    'solution': ('Name, evaluation points per axis', 'Explicit scalar prediction or u/v/p table request; no reference assumed.'),
    'monitor': ('Name, retained updates', 'Explicit bounded loss recording; closing its view does not stop recording.'),
    'reference': ('Preset, coordinate expression or mapped values', 'Explicit scalar source. Imported data requires matching coordinates; preset sources reject custom physics.'),
    'export': ('Name, evaluation points per axis', 'Explicit scalar/u-v-p data request. Export CSV only after training and on user action; no plot window.'),
    'comparison': ('Name, evaluation points per axis', 'Requires a PINN run and an explicit reference matching coordinates and scalar/u-v-p fields.'),
}


def describe_block(kind):
    from .ports import INPUTS,OUTPUTS
    parameters,limits=CAPABILITIES[kind]
    inputs=', '.join(spec.label for spec in INPUTS[kind]) or 'none'
    output=OUTPUTS[kind]
    return ('Implemented · '+parameters+'\nInputs: '+inputs+'\nOutput: '+(output.label if output else 'none')+'\n'+limits)
