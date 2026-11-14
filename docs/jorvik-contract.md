# Jorvik data boundary

For the current `.jorvik` document and NetCDF workflow, see [Jorvik geometry](jorvik.md).
The separate helper below supports the legacy schema-v2 NPZ format; `_gs_metadata`
is a compatibility field and has not been renamed.

Jorvik owns CAD loading, face/edge selection, geometry-aware sampling and
normal construction. Margam consumes explicitly prepared point sets; it does not
reimplement those tools or add a 3D viewport.

The implemented bridge accepts Jorvik **schema-v2 NPZ** exports for datasets
on a common axis-aligned plane. It reads `_gs_metadata`, labelled array references,
optional entity tags and normal-array references without importing Jorvik,
Gmsh, SciPy or xarray. The user explicitly selects source axes and labels. A dropped
coordinate must be constant across selected sets; normal components along that
coordinate must be zero. Non-planar data is rejected rather than projected silently.

```bash
python -m examples.prepare_jorvik_2d \
  --input /absolute/path/samples.npz \
  --output /absolute/path/new-prepared-directory \
  --axes x,y --labels interior wall
```

The destination must be new. It contains `dataset-000.npz`, etc., plus `mapping.json`
with original labels, source hash, source axes, plane coordinate and unchanged units
metadata. Map each prepared dataset explicitly in Margam: `points` for coordinates,
`normals` only when present, and no target values until physics is assigned. The
built-in scalar/flow builder uses named side slots; Prepared 2D scalar models
accept user-named sets. Neither naming scheme infers CAD topology. Arbitrary Dirichlet point sets do not need normals. Normal-dependent
operators on off-box sets require explicit unit normals with outward orientation
supplied by the data provider. No units or normal sign is inferred.

For imported non-rectangular clouds the rectangle is only a normalisation/evaluation
bounding box. The optional sample preview says so. Its regular output grid includes
all bounding-box points; the application does not infer a fluid/solid mask. Choose
an appropriate explicit evaluation workflow before interpreting values outside a
complex domain. The CSV export is not a CAD or ParaView integration.

For this legacy NPZ helper, full 3D coordinate/field support, non-axis-aligned
reductions and automatic graph wiring remain deferred. The separate Jorvik
geometry block supports saved-document editing and explicit NetCDF updates.
The implemented planar bridge is tested against the upstream schema contract;
no claim of validating every CAD model or upstream release is made.


## Use prepared labels in a non-rectangular scalar model

The [Interior data and Boundary data blocks](training-data.md) now accept the
prepared coordinate/normal arrays. Choose Prepared 2D points, map the interior
file, add each selected boundary label, then assign scalar conditions downstream.
Labels have no inferred normal direction; normal-dependent conditions require
explicit outward unit arrays. The existing preparation helper remains separate
from graph editing; automatic mapping.json-to-graph wiring is not implemented.
