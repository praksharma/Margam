"""The currently insertable UI blocks; no plugin or solver registration system."""

# Category, then (display title, existing template kind).
BLOCK_CATEGORIES = {
    'Geometry': (('Rectangle', 'domain'),),
    'Sampling': (('Interior sampler', 'interior'), ('Boundary points', 'boundary')),
    'Neural networks': (('Neural network', 'network'),),
    'Loss / Physics': (('PDE residual', 'pde'), ('Boundary condition', 'bc')),
    'Optimisation / PINN': (('PINN method', 'method'),),
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
    'domain': ('Bounds and shape', 'Interval or axis-aligned rectangle. Advanced CAD belongs to GeoSampler.'),
    'interior': ('Method, budget, seed', 'Fixed meshless PDE points. Grid may use fewer than the budget; other implemented methods return the exact count.'),
    'boundary': ('Named side, method, count, seed', 'Independent rectangle-side recipes; interval endpoints are fixed. BC assignment is separate.'),
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
