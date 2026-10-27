"""The currently insertable UI blocks; no plugin or solver registration system."""

# Category, then (display title, existing template kind).
BLOCK_CATEGORIES = {
    'Geometry': (('Rectangle', 'domain'),),
    'Sampling': (('Interior sampler', 'interior'), ('Boundary points', 'boundary')),
    'Neural networks': (('Neural network', 'network'),),
    'Loss / Physics': (('PDE residual', 'pde'), ('Boundary condition', 'bc')),
    'Optimisation / PINN': (('PINN method', 'method'),),
    'Results': (('Solution Plot', 'solution'), ('Training Monitor', 'monitor'),
                ('Reference', 'reference'), ('Comparison', 'comparison')),
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
