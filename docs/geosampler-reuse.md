# GeoSampler reuse assessment

Inspected 26 October 2026, local checkout `/home/hell/Desktop/GeoSampler`,
revision `c532cecec8777ee239e4c08bb41d7d922a82bc5c`.
This is a source review, not a new validation of GeoSampler's test suite.
Initial source review was followed by adapting `theme.py` into `margam/ui/theme.py`.
GeoSampler itself remains unchanged. See `../THIRD_PARTY.md` and `../LICENSE`.
Both light and dark themes passed offscreen rendering checks in the prototype.

## Best candidates

| Capability | Source under `src/geosampler/` | Margam use and adaptation |
| --- | --- | --- |
| Light/dark palette | `theme.py` | Small Qt-only module. Rename the app property, apply colours to our custom blocks, wires, and 2D preview as well as widgets. |
| Dock layout and settings | `app.py` | Reuse the pattern for View menu, reset layout, theme, window geometry and dock persistence. Use independent Margam settings and stable dock object names. Do not copy the CAD-coupled main window wholesale. |
| Sampling catalogue | `sampling_methods.py` | One inventory feeds both UI and numerical code, matching our documentation/discovery goal. Extend metadata with settings, limitations, and examples. |
| Unit-domain sampling | `sampling.py`: `unit_points`, `radical_inverse` | Extract the numerical functions without Gmsh integration. Map unit coordinates into intervals/rectangles; map 1D samples onto named sides and supply analytical outward normals. |
| Sampling controls | `sampling_panel.py` | Reuse interaction patterns: method, seed, count, explicit Generate, Cancel, status, scrollable controls. Replace window/CAD dependencies with Margam component interfaces. |
| Named regions | `regions.py` | Preserve stable identity, editable label, colour, and snapshots of provenance. Replace CAD tags/hashes with rectangle side identities or imported point-set identities. Keep BC physics separate. |
| Export contract | `export.py`, `normals.py` | Future GeoSampler adapter can read labelled arrays, manifest, entity tags, normal arrays and availability metadata. Do not infer physics from labels. |
| Background work | `sampling_panel.py`, `project_worker.py` | Reuse process/thread ownership and completion/error handling patterns. Training needs its own progress, cancellation and result protocol. |
| Save/recovery | `project.py`, `project_ui.py`, `recovery.py`, `shutdown.py` | Borrow versioned manifest + numeric arrays, atomic writes, validate-before-replace, unsaved prompts and recovery lifecycle. Define a Margam schema for graphs, methods and physics instead of adopting the CAD archive. |
| Validation patterns | `tests/` | Reference reproducibility, region integrity, failed-load preservation, persistence, UI lifecycle and export round-trip cases. Adapt to Margam behaviour. |

## Sampling details that matter

The numerical engine implements random, regular grid, LHS, Sobol, Halton,
Hammersley and stratified sampling. It is independent of Qt; the unit sampling
functions can run without CAD. NumPy is required and QMC methods load SciPy.

Grid sampling returns complete Cartesian layers and may produce fewer points
than the requested budget. The UI should display the actual generated count.
Sobol generation takes a power-of-two design and truncates it for arbitrary
requested counts; do not imply that every truncated design retains full balance
properties. Sampling is seeded. Current numerical validation permits dimensions
1 through 4 and caps the count at 500,000.

For rectangles, use the numerical designs directly rather than constructing CAD
or a tessellation. Keep interior and boundary sampling distinct; handle endpoint
and corner policy explicitly. Selection and BC assignment remain Margam work.

## Existing GeoSampler export structure

The inspected local implementation uses schema version 2. Each labelled point
set is an independent N×3 coordinate variable, with a separate point dimension.
Labels that cannot serve as safe variable names have internal keys and a label
map. Metadata records sampling method, seed, source identity and entity details.
Applicable normals have their own arrays and explicit availability information.
NPZ carries a `_gs_metadata` manifest; NetCDF carries corresponding dataset
metadata. Preserve these semantics in an importer, not just the visible arrays.

Margam's 1D/2D inputs should not inherit a mandatory third coordinate. An adapter
must explicitly choose the physical coordinates or plane representation. Do not
silently discard z for arbitrary imported geometries. Units in GeoSampler are
currently described as source geometry units, unspecified; import must not assume
metres. Outward normals are conditional and orientation-validated, not universally
available for every selected entity.

## Keep outside the initial Margam application

- Three.js, WebEngine, QWebChannel, camera controls, picking, orientation gizmo.
- NVIDIA/WebGL graphics workarounds specific to that rendering stack.
- Gmsh CAD import, containment and tessellation for basic rectangle sampling.
- CAD project archives and transformations as a substitute for a PINN model format.

GeoSampler remains the future complex-geometry route. A 2D native Qt canvas and
preview suffice for the current Margam scope.

## Suggested order

1. Adapt theme and dock-layout persistence into the prototype.
2. Extract unit-domain samplers and catalogue with source attribution; validate
   reproducibility, bounds, actual counts, rectangle mapping and side normals.
3. Connect rectangle controls, point preview and named boundary sets.
4. Add Margam-specific graph persistence and responsive training execution.
5. Later, implement the versioned GeoSampler export adapter and evaluate a closer
   integration only if the workflow calls for it.

## Provenance and compatibility

GeoSampler declares AGPL-3.0-only. Its `THIRD_PARTY.md` attributes application-shell
and theme ancestry to PyniteGUI. Retain applicable licence text and provenance
when copying or adapting source. Three.js and example CAD fixtures have separate
notices; they are not needed for the recommended first reuse.

The clone pins PySide6 6.11.2 and documents a Qt 6.12 NVIDIA/WebEngine issue.
Margam currently uses PySide6 6.12.0 with a native Graphics View canvas, not that
WebEngine stack; the note alone does not establish a Margam issue or justify a
downgrade. NumPy/SciPy/xarray belong in explicitly declared dependencies when
numerical sampling or imports are added. Do not install the entire GeoSampler
application merely to borrow its desktop theme or rectangle samplers.

GeoSampler's guide calls the release a development preview, despite its broader
implemented feature set. Its documented validation is useful evidence, but does
not replace testing any code adapted into Margam on the target environment.
