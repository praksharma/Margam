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
