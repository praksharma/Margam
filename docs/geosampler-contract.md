# GeoSampler boundary

GeoSampler owns CAD loading, face/edge selection, geometry-aware sampling and
normal construction. Margam consumes explicitly prepared point sets; it does not
reimplement those tools or add a 3D viewport.

The implemented bridge accepts GeoSampler **schema-v2 NPZ** exports for datasets
on a common axis-aligned plane. It reads `_gs_metadata`, labelled array references,
optional entity tags and normal-array references without importing GeoSampler,
Gmsh, SciPy or xarray. The user explicitly selects source axes and labels. A dropped
coordinate must be constant across selected sets; normal components along that
coordinate must be zero. Non-planar data is rejected rather than projected silently.

```bash
python -m examples.prepare_geosampler_2d \
  --input /absolute/path/samples.npz \
  --output /absolute/path/new-prepared-directory \
  --axes x,y --labels interior wall
```

The destination must be new. It contains `dataset-000.npz`, etc., plus `mapping.json`
with original labels, source hash, source axes, plane coordinate and unchanged units
metadata. Map each prepared dataset explicitly in Margam: `points` for coordinates,
`normals` only when present, and no target values until physics is assigned. The
current scalar/flow builder uses named boundary slots; their labels do not infer
CAD topology. Arbitrary Dirichlet point sets do not need normals. Normal-dependent
operators on off-box sets require explicit unit normals with outward orientation
supplied by the data provider. No units or normal sign is inferred.

For imported non-rectangular clouds the rectangle is only a normalisation/evaluation
bounding box. The optional sample preview says so. Its regular output grid includes
all bounding-box points; the application does not infer a fluid/solid mask. Choose
an appropriate explicit evaluation workflow before interpreting values outside a
complex domain. The CSV export is not a CAD or ParaView integration.

Full 3D coordinate/field support, non-axis-aligned reductions, NetCDF manifests,
automatic region-to-BC linking and live CAD selection remain deferred extensions.
The implemented planar bridge is tested against the upstream schema contract;
no claim of validating every CAD model or upstream release is made.
