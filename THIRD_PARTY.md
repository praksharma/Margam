# Source attribution

`margam/ui/theme.py` is adapted from `src/geosampler/theme.py` in GeoSampler,
revision `c532cecec8777ee239e4c08bb41d7d922a82bc5c`.
Source: https://github.com/praksharma/GeoSampler
Licence: AGPL-3.0-only; upstream licence text retained in `assets/LICENSE-GEOSAMPLER`.

GeoSampler attributes the application shell and theme palette to PyniteGUI:
https://github.com/praksharma/PyniteGUI
PyniteGUI source revision recorded upstream:
`c421ec091b467da7e3955925622988bba36153a7`, author Prakhar.

Adaptations rename the application theme property for Margam. The prototype
uses these colours in its native Qt canvas, ports, connections and 2D preview.
No Three.js, CAD fixture, or WebEngine code is copied.

The Hammersley and equal-volume recursive stratified algorithms in
`margam/components/sampling.py` are adapted from GeoSampler
`src/geosampler/sampling.py`, revision c532cecec8777ee239e4c08bb41d7d922a82bc5c,
AGPL-3.0-only. They are restricted to intervals/rectangles and do not import
GeoSampler, SciPy, Gmsh or any CAD/UI code. Existing Margam grid/random sequences
remain unchanged.

`margam/ui/desktop_state.py` follows GeoSampler's QSettings/saveState layout and
QLockFile-per-recovery-file patterns (`app.py`, `recovery.py`, same revision above).
The implementation is specific to Margam JSON model drafts; no CAD caches, worker
protocols or GeoSampler project payloads are copied.

The planar import adapter consumes the GeoSampler schema-v2 NPZ manifest contract
from `src/geosampler/export.py` at the same recorded revision. It is a Margam data
adapter; no CAD import, Gmsh worker or 3D viewport implementation is included.

## PhysicsNeMo pointwise networks

The pointwise architecture adaptations in `margam/nn/{pointwise,fourier,highway,
modified_fourier,multiscale_fourier,dgm,siren,multiplicative_filter}.py` draw from
NVIDIA PhysicsNeMo/PhysicsNeMo Sym, Apache-2.0. Copyright notices are retained in
those files; the full Apache-2.0 text is retained in `assets/LICENSE-PHYSICSNEMO`.
No upstream NOTICE file was present in the inspected local repository root.

Inspected local source: `physicsnemo/nn/module/{fourier_layers,siren_layers,
dgm_layers,fully_connected_layers}.py`, revision
`96b7cd1c75914c1d2a438937a4fd340c08bebe59` of
https://github.com/NVIDIA/physicsnemo .

Complete model equations were also checked against
`physicsnemo/sym/models/{fourier_net,highway_fourier_net,modified_fourier_net,
multiscale_fourier_net,dgm,siren,multiplicative_filter_net}.py`, revision
`0c5f18c31d03d46ec0cc3ef4ccbf761a956dd38c` of
https://github.com/NVIDIA/physicsnemo-sym .

Margam modifications: plain tensor pointwise I/O and fixed [0,1] domain scaling;
shared network construction for scalar/u-v-p PINNs; Torch-only seeded frequency
and Gamma sampling; bounded concrete UI settings; no key/dictionary Arch wrappers,
weight normalisation, adaptive activation parameters, or PhysicsNeMo runtime.
Smooth tanh/SiLU/sine activations are selectable where supported. The original
Margam fully connected implementation and defaults are unchanged.

Gabor uses a softplus positive-width parameter and a direct sum of squared distances,
so trained widths stay positive and second derivatives remain finite at a filter
centre. These are explicit changes from the inspected upstream Gabor layer. The
shared pointwise and linear helpers implement only the subset used by these networks.
No FNO, AFNO, spectral operator, graph, convolutional or attention architecture was
imported. See `docs/neural-networks.md` for per-family contracts and differences.
