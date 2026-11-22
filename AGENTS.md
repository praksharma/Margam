# Margam repository instructions

## Working location and environment

- Work in the Margam repository root. Do not put project code or documentation
  in a separate chat workspace, even if that is the chat's initial directory.
- Reuse the single repository-root `.venv`. Check its Python version and installed
  dependencies before changing it. Current development uses Python 3.14.
- Do not create virtual environments inside feature folders or invent additional
  environment/backup directories. A Python upgrade should be a deliberate,
  user-authorized change to the root environment.
- Use `uv pip ... --python .venv/bin/python` for dependency management. Do not assume
  the environment has its own pip module. Follow README.md for running commands.

## Repository organisation

- Source: `margam/`; numerical methods: `margam/methods/`; shared components:
  `margam/components/`; configured examples: `margam/templates/`; desktop: `margam/ui/`.
- Keep one file per PINN recipe under `margam/recipes/`. That file defines its
  components, connections, defaults, ports and compact settings. Compact and
  expanded insertion must share that definition; physics examples stay separate.
- Keep one file per PINN method. Reuse shared sampling, optimisation, networks and
  loss components, while allowing a method to own its distinctive behaviour.
- Tests belong in `tests/`, runnable examples in `examples/`, documentation and
  plans in `docs/`, static resources in `assets/`, screenshots in
  `assets/screenshots/`. Keep the root limited to standard project metadata,
  instructions and setup files. Do not leave scratch code or images there.
- Keep implementation status, component capabilities and limitations documented.
  Update README.md running commands and docs/todo.md as work changes.

## Scope and interaction design

- Build concrete supported workflows first. Do not invent generic registries,
  extension frameworks, universal plot actions or broad abstractions without a
  demonstrated need. Share actual common behaviour rather than hypothetical cases.
- Follow the documented Simulink interaction reference in docs/simulink-evidence.md,
  distinguishing verified behaviours from deliberate Margam choices.
- The project is a PINN builder across physics, not a heat-transfer-only tool.
  Keep demo-specific assumptions in templates rather than shared UI contracts.
- Basic interval/rectangle/3D-box/sphere-obstacle sampling must work directly.
  Advanced CAD/geometry editing belongs to Jorvik integration. An optional
  dedicated Solution Explorer window supports interactive 1D, 2D, 3D viewport,
  transient playback, and parametric solution post-processing.
- Views are optional and owned by their blocks or opened explicitly in the
  dedicated Solution Explorer window. Do not show global default figure panes
  at startup or assume an analytical solution.
- Local sample/geometry previews currently belong to Rectangle blocks only.
  Keep operational progress/errors separate from numerical output views.
- In-app 2D/3D field visualization, vector overlays, probes, and parametric
  exploration are actively supported in the dedicated Solution Explorer window
  (using native PySide6/OpenGL with a deterministic software fallback so
  offscreen tests work cleanly).
- ParaView file export remains supported through its own opt-in block alongside
  the in-app Solution Explorer. Concrete 2D and 3D transient/steady flows and
  continuous parametric PINNs are supported where explicitly configured.

## Validation and commits

- Check the current branch and working tree before editing; preserve user changes.
- Keep `dev` untouched unless the user explicitly authorizes changes to it.
- Run checks appropriate to the change. Verify UI interactions and lifecycle where
  relevant; store screenshots under assets/screenshots. Label CPU-only validation
  and CUDA skips accurately. Do not claim numerical accuracy from UI startup alone.
- Make regular, coherent commits at verified milestones. Do not leave completed
  work uncommitted or push changes unless the user requests it.
- Every commit created or amended by Codex must include this exact trailer,
  separated from the commit body by a blank line:

  ```text
  Co-authored-by: Codex <codex@openai.com>
  ```

  This includes documentation-only commits. Preserve the human author's identity.
  Do not rewrite existing history unless explicitly authorized for that purpose.

## Current data and desktop contracts

- Core parameters are owned by blocks. Assemble supported physics through the
  typed graph; the inspector cache is not numerical source-of-truth.
- Keep restricted expressions interpreted through the allowlisted AST. Do not
  replace this with Python eval or executable model-file payloads.
- Run archives load with weights_only=True and verify architecture, Adam options
  and data fingerprints. Do not silently resume against changed external data.
- Unit tests must use Window(state_directory=False), or a dedicated temporary
  directory when testing desktop state. Do not overwrite personal layout/recovery
  files during validation. Release test clipboard data before offscreen Qt teardown.
- Coupled u/v/p and 3D u/v/w/p results support tables/CSV, ParaView export, and
  interactive inspection in the dedicated Solution Explorer window. ParaView
  export is opt-in; never infer physical time from training updates. Evaluation
  points and Reference data blocks accept CSV/NPZ independently of training.
  Preserve exact coordinates, row order and integer/text IDs. Align references
  explicitly without interpolation. Supplied clouds export as VTU vertices and
  physical-time PVD.

- Prepared 2D uses separate Interior data and Boundary data blocks, with coordinate
  bounds for scaling only. Keep free boundary labels and target/operator physics
  separate. Require explicit unit outward normals for normal-dependent conditions
  and explicit evaluation points; never fill an unknown domain with box samples.
  The current non-rectangular contract is steady planar scalar baseline/LRA/fixed-SA.

- Residual samplers own initial points and schedules; trainer hooks perform scoring.
  Preserve inverse-probability MSE for importance sampling and checkpoint the active
  set/history. Keep BCs/export grids separate. Reject SA masks and imported-domain
  candidate generation until their explicit data/transfer contracts are implemented.

- Integral loss integrates squared pointwise errors with physical measures; it
  does not square a signed residual integral. Keep mean MSE the default, sum
  vector components in integral mode, and retain unit endpoint/gauge penalties.
  Apply volume once with importance correction. Reject RAR/imported quadrature
  until appropriate point weights exist; persist mode/measures in checkpoints.

- Credit community implementation references in docs/community-methods.md and
  each method guide. Distinguish algorithm inspiration from copied/adapted source,
  retain exact upstream references/notices, and state implementation differences.

- Burgers uses explicit SpaceTime1D geometry and (x,t) coordinates, two spatial
  boundary traces plus an initial set; never impose a final-time BC. Keep its
  solution in the existing table view and export physical-time PVD line frames,
  not a VTI plane treating time as space. Do not assume an exact later-time reference.

- Expert PINN is Margam’s label for Expert’s Guide v1, not an author-proposed
  name or mixture of experts. Follow paper sum-L2-gradient-norm balancing and
  old-weight EMA; do not substitute existing max/mean LRA. Causal weights are
  detached and (x,t) pairs stay intact. Preserve same-device sampling/decay resume.
  The inspected JAX-PI licence restricts copying; use independent paper-based code.

## User documentation

- Keep task-based help under docs/start, docs/modeling and docs/use, with docs/index.md
  as the user entry. Keep method/API/evidence pages separate as reference.
- Update screenshots/captions when their controls change; use the state-isolated
  examples.capture_manual helper, with images in assets/screenshots.
- Build/check with examples.documentation. Generated source/site stay under the
  ignored build/manual directory; never edit generated copies. Local builds do not
  publish. The dedicated Pages workflow deploys the manual when main is pushed;
  do not push or change remote Pages settings unless the user requests it.
- F1 opens the local built guide. Zensical is an optional docs extra, not a numerical
  or desktop startup dependency. Preserve offline assets and their licenses.

- Public documentation, snippets, screenshots and capture manifests must not
  contain local usernames or home-directory paths. Use repository-relative demo
  paths, and inspect capture inputs before saving screenshots. Keep real local
  filesystem paths confined to local tool operations and diagnostic UI when needed.

- Expert suite examples use concrete paper-case contracts. Torus Taylor–Green and cylinder startup are explicitly labelled variations. Preserve obstacle exclusion, outlet traction, cavity corner handling and physical-time PVD frames. Torus output w is scalar vorticity, not vertical velocity. Do not claim paper accuracy or Reynolds curriculum support. NTK and sequential time marching are opt-in Expert controls; preserve detached teacher ICs, per-window Adam/EMA resets, local coordinate bounds and piecewise physical-time predictions. Reject predictions beyond trained windows and validate the complete teacher/checkpoint chain.

- Scale-PINN is an independent paper-based trainer. Keep w^(k-1) frozen, compute its spatial derivatives before detaching the correction, and snapshot w^k before the optimizer step. Time is never a smoothing dimension. Flow correction maps velocity to momentum and pressure change to continuity; reject undefined mappings. Preserve both iterates in archives and distinguish physical PDE MSE from corrected training loss. The upstream repository declares all rights reserved; do not copy code/data.

- Scale high-Re cavity entries are independent starting presets, not solved benchmarks or a curriculum. Keep ν and default γ at 1/Re for unit lid speed/length and display Re from current parameters. KdV uses third-order dispersion and periodic matching through second derivative; KS fourth-order coefficient is 100/16⁴. Preserve physical-time exports and label borrowed tuning settings.

- Jorvik geometry documents are explicitly enabled for steady planar 2D scalar
  imported samples. Keep the editor in a separate process using the root environment.
  New documents use `.jorvik`; exports use `domain.nc` beside `domain.jorvik`.
  Export saved generated arrays automatically on editor close and Update, using
  Jorvik project validation and atomic export; never regenerate samples or silently
  fall back to stale exports. An explicitly browsed NetCDF copy overrides this.
  Prefill dataset roles from metadata and boundary names from labels; Ignore is
  opt-in. Apply updates explicitly with dataset-role review, stable region IDs,
  verified planar conormals and immutable owned snapshots. Never assign PDE/BC
  physics from labels or infer mesh topology. Preserve relative owned references,
  Save As isolation, owned-file recovery and the editor lock during save/close.
  Broader Jorvik/3D integration remains deferred.
- New Scale examples use notebook-specific affine inputs and a trainable sine
  projection in the separate Affine Fourier NN architecture. Preserve physical
  units by default, keep normalization explicit, and do not silently migrate old
  fixed-Fourier checkpoints or claim upstream trunk/head equivalence.

- gPINN differentiates the full scalar residual (or each supported steady-flow component) including forcing/coordinate-callable
  targets, not its square or just the solution. Preserve higher-order parameter
  gradients. Keep coordinate penalties explicit; zero disables a direction. Log
  raw gradient terms separately, preserve BC/IC losses and score RAR on physical
  residuals. Mean/fixed/RAR only; do not silently compose advanced trainers.

- RBA uses detached normalized absolute PDE residuals and squared weighted
  residual losses. Keep point identities fixed, both initialization modes and
  zero-residual safety. Commit attention only after a successful update; checkpoint
  masks separately from optimizer parameters. Never silently move or interpolate
  attention under resampling. Keep raw and weighted PDE diagnostics distinct.

- PDE collocation follows point-source arrows. The optional boundary input adds
  geometric boundary/initial rows; without that wire do not append them. Allow a
  boundary source on the primary collocation input too. Keep BC operators separate.
  New example graphs wire both sources, while old documents/API defaults remain
  interior-only. Preserve duplicates, forcing derivatives, adaptive boundary
  anchors with unit importance weights, and physical-time causal bins. Reject
  extra boundary rows as domain-volume integral quadrature; never silently mix
  measures. Exclude pressure gauges and mean-velocity anchors from geometry rows.

- Desktop execution device is an application-wide preference (Automatic/CPU/CUDA),
  overriding legacy per-block values. Keep CLI options explicit and independent.
  Preserve trained weights on device changes; rebuild prediction weights on the
  chosen target when needed. Resume must honor the global selection. Data and UI
  preparation stay on CPU. Lock device changes during running jobs/editors.
- Inspector and library information buttons open scrollable descriptions with
  actual choices/limits. Reuse existing widgets; avoid inventing unsupported
  parameter semantics in fallback help.

- Add a dedicated setting-help description when introducing numerical controls;
  explain its meaning and interactions. Available choices and limits are read
  from the live widget, including disabled choices. Keep fallback descriptions
  limited to actual widget values and block capabilities.

- Run archive version 4 embeds project-owned files in a bounded, validated relative
  file map. Restore into a new private workspace before replacing the graph;
  preserve older archives and external-input semantics. Never extract traversal,
  absolute, ambiguous or conflicting paths, or follow symlinks when capturing.

- BC blocks own positive per-constraint `bc_weights`, default 1, multiplying the
  existing normalization/global weight without renormalization. Keep raw losses
  separate from `weighted:<name>` objective contributions. Record adaptive
  contributions using the weights actually used before the optimizer/mask update;
  never reconstruct older missing history from current coefficients. Model 27
  migrates old BC blocks to defaults. Reject changed objectives on continuation
  and clear removed Jorvik aliases' coefficients alongside their physics settings.

- CSV/NPZ mapping dialogs offer explicit Copy into project / Link existing file.
  Default new sources to Copy, preserve existing links, and copy only after an
  accepted validated mapping. Validate copied bytes before installing references;
  retain fixed snapshots for undo and include them in existing recovery/run payloads.
  Canonicalize owned mapping paths in editor parameters and undo values. Resolve
  all training/evaluation/reference mappings at use; rebase cached trained configs
  across Save As without treating file relocation as changed physics. Never turn
  an owned URI into an absolute external link or silently bundle external files.

- Project data overview is a read-only inventory of block-held file mappings,
  including disconnected/unmapped source blocks. Inspection must not mutate model,
  undo/results or filesystem; Refresh must not replace established baselines. Use non-creating owned-path
  resolution for missing-file checks. Use saved accepted-import fingerprints when available, and label only older
  unrecorded mappings as session-only; use Jorvik's applied fingerprint where present. Mapping edits are explicit,
  guarded while busy/stale, and use the existing owning block's dialogs. Disconnect
  observers on window shutdown; do not retain stale Qt node references.

- Explicit desktop Validate opens a clickable multi-issue report; programmatic
  validation remains boolean and non-modal. Keep the supported numerical execution checks
  required; active data guards may reject cases older lazy reference callbacks
  deferred until prediction. Require explicit evaluation points on prepared-domain
  outputs, validate mapped legacy reference alignment before training, and resolve
  inputs without creating missing owned folders. Reuse actual expression/data/network/method validators; avoid a
  second numerical executor or promises of exhaustive diagnostics on incomplete
  graphs. Unused optional sources/outputs are advisory. Preserve model/history,
  files and results while inspecting; selecting an issue only opens its owner.
  Use IDs instead of retaining Qt nodes, guard stale boundary contexts, and
  disconnect report observers on window shutdown.

- Model 28 persists `data_fingerprints` separately from numerical mappings.
  Record them only on an explicitly accepted dataset mapping, bound to that exact
  mapping; never fabricate historical checksums or rebase them during inspection,
  validation, save or archive reopening. Keep mapping/fingerprint edits undoable
  together, and clear cached training weights when reaccepting training data.
  Valid changed imports are warnings for fresh training, while checkpoint data
  identity and Jorvik snapshot checks remain strict. Keep hashing bounded and
  metadata decoding independent of Qt.


- Format layout commands change positions only, as one undo entry; preserve graph,
  cached numerical results and selection. Auto arrangement includes disconnected
  blocks and terminates on cycles without claiming they are executable. Equal-gap
  distribution must not introduce overlap. Keep shortcuts scoped to the canvas.
- Model search stores block IDs rather than graphics objects and resolves them
  again before navigation. Refresh after edits; stale/deleted results must remain
  safe. Do not treat search as insertion or a numerical mutation.

- Recent model paths belong only to local desktop preferences (ten unique entries).
  Record successful model opens/saves, never recovery drafts, failed operations or
  run archives. Preserve discard prompts and busy locks; removing history must
  never delete model files. State-disabled tests use ephemeral memory only.


- Port/wire explanations describe problem dependencies, not time signals or the
  autograd graph. Wire inspection selects the wire, with optional ID-based endpoint
  navigation; do not modify graph topology while inspecting it.
- Output freshness is ephemeral availability of current in-memory data, not a
  numerical accuracy claim or external-file watcher. Preserve running/evaluation,
  failure and partial states; settings edits still clear obsolete data. Export
  readiness does not imply a file was written. Keep badges out of model settings.
- Settings search indexes block-owned parameters, uses existing inspector controls
  and stable IDs, and never expands recipes or changes options automatically.
  Hidden/disabled controls must be reported honestly. Side selection is UI context.
- GUI Train/F5 reviews a run summary before starting; direct start_training and
  checkpoint continuation keep their existing contracts. Summaries use actual
  supported initial sample builders and explicit collocation selection, never build
  a neural network or guess bounding-box samples on prepared geometry. Label initial,
  adaptive and sequential-window counts accurately and revalidate before training.


- Model 29 adds plain-text annotation nodes with no ports, bounded to 4096 characters.
  Keep annotation text out of numerical assembly and URI/file-ownership inference.
  Note-only add/edit/delete/clipboard operations must preserve weights, predictions
  and freshness; mixed graph edits retain normal invalidation. Escape note text in
  rich-capable tooltips and keep canvas/editor text literal. Preserve older models.
- Blank models use File → New blank model / Ctrl+N with the usual discard and busy
  guards. They are saveable drafts, not executable implicit Poisson problems.
- Dependency highlights are ephemeral, ID/edge-based graph reachability; handle
  branches/cycles without changing selection, settings or numerical state. Recompute
  on selection/topology changes and guard callbacks during scene teardown.
- Grid/snap and dependency mode are local preferences. Snap drag groups by one
  common offset, record one undo step and preserve exact loaded/undo/layout positions.
  Right-clicking a selected item must preserve multi-selection and expose group
  commands with busy guards. Auto-arrange the computational graph without moving notes.


- Inverse Burgers is the concrete continuous-time identification problem: two
  jointly trained constants and fixed measured `(x,t,u)` rows. Both MSE terms use
  the same observation instance; do not add hidden BC/IC losses or reference answers.
  The dedicated PDE owns initial guesses; learned values never overwrite config.
  Observation loss uses the trainer's existing boundary_weight slot with its own UI
  label, not BC normalization. Log diffusion ensures a positive physical lambda_2.
- Keep inverse code and recipe in their own inverse_burgers files. Reuse pointwise
  networks, explicit data mapping/copy/link and output blocks; reject unsupported
  trainer/sampler compositions. Model 30 adds inverse_settings, with old defaults
  migrated in models and advanced checkpoint signatures. Persist physical parameters,
  Adam state and after-update parameter history, and retain strict observation/data
  identity and objective checks on resume. Inference restores learned parameters.
- Credit the original mathematical formulation separately from the MIT observation
  dataset. Retain its pinned source/hash/licence and never publish local paths in
  captures/evidence. Short coefficient-update tests are not accuracy reproductions.

- MCP automation is optional, starts off, and reuses the official SDK with an
  authenticated loopback-only listener. Keep server imports lazy. All graph/Qt
  commands cross the bounded queued bridge; recheck permissions and listener
  generation at dispatch. Do not execute GUI operations on the HTTP thread.
  Require window session and exact model revision for edits/training; preserve
  atomic validation/one Undo entry and existing data/output/device contracts.
  Track asynchronous jobs separately; reject results from changed graphs or
  replacement desktop jobs. Stop/revoke the listener on accepted window close.
  No arbitrary code, file-management or implicit output tools. Credit the adapted
  PyniteGUI framework and retain its licence. Use state-isolated real-client tests.

- `margam.research` is the script-facing API inside the existing package, separate
  from UI registration. Reuse the shared PINN loop rather than duplicating it.
  Keep plain PyTorch module/callable/factory replacement points and document fixed
  sampling, shapes and differentiability. Default research updates reject
  non-finite gradients; custom update callbacks own their optimizer work/state.
  Research checkpoints are a distinct tensor/primitive format: require a versioned
  caller identity and fixed data, validate actual weights/options/tensors/history,
  retain Torch RNG for same-device continuation, and reject unsupported custom
  update/stateful-loss/custom-optimizer state instead of claiming exact resume.
  Never load research functions from a desktop model or require GUI registration
  for ordinary Python experiments. Check an external-folder script and CPU/CUDA
  continuation when changing these interfaces.

- Model comparison is read-only and matches stable block IDs after normal model
  migrations. Compare block-owned settings/connections, not the inspector config
  cache or inferred numerical equivalence. Keep names/notes/positions optional.
  Read only bounded model JSON: do not resolve/import datasets or create owned
  folders. Preserve results/history; resolve navigation IDs at click time. Compare
  snapshots refresh explicitly, and selecting a saved comparison never replaces
  the current canvas. Render full values literally, without executable payloads.

- Model format 31 stores optional presentation `label` and compact `network_label`
  metadata; retain formats 1–30. Keep labels separate from kind, ID, boundary
  aliases and numerical parameters. A label equal to an automatic type name is
  still explicit until cleared. Show actual type/ID and render labels literally.
  Renaming must preserve trained data/freshness and be undoable; retain root and
  network labels across compact/expanded recipes and recovery/copy/run files.
  Ignore labels when detecting owned-file references or default numerical model
  differences. Never dispatch physics or identify entities by the display label.

- Unused import cleanup is explicit, limited to recognised single-file CSV/NPZ
  snapshots under datasets/<UUID>/. Record mapping references before/after edits,
  retain pins across Undo/Redo/macros/merges, and collect only path fields, never
  labels/expressions or obsolete fingerprint bindings. Recompute current, saved,
  cached and history protection before removal; refuse unknown history or ambiguous
  references. Never clear Undo automatically, prune linked originals/CAD/NC files,
  follow links or create owned folders during inspection. Use descriptor-anchored
  checks and preserve changed/unknown files. Cleanup must not alter model/results.

- Selected error norms are read-only diagnostics on stored aligned comparison
  arrays. Do not invoke inference/importers or invent a reference. Default the GUI
  to one field; vector selections combine compatible raw field values explicitly.
  Use pointwise Euclidean magnitudes, unweighted rows/all stored physical times,
  explicit zero-reference policy and finite/None range statuses. Keep legacy pooled
  metrics/export/archive formats intact. Report choices are ephemeral, never loss
  settings; clear stale reports on semantic/result changes and resolve owners by ID.
  Reuse the same function in research scripts and retain detached tensor inputs.

- Research StepDecay uses completed updates and closed-form initial-group rates,
  advances only after a successful default optimizer update, and logs actual rates.
  Reject external rate edits, zero-rate underflow and custom-step composition.
  Scheduled research checkpoints use format 2 with exact schedule/history checks;
  preserve unscheduled format 1 and copy optimizer tensors when restoring snapshots.

- Optional wire routing is a local visual preference, not model metadata. Use
  bounded orthogonal lane detours around up to 64 block/note rectangles, with
  explicit curved fallback; preserve topology, selection, numerical state and
  Undo history when toggling. Recompute on moves/insertion/deletion and Undo/Redo.
  Shared lanes/crossings are not graph junctions. Keep pending-wire drawing and
  the default curved mode intact; test painted strokes, not implicit path fills.

- Transient table time selection filters exact stored t values only, with sorted
  frame choices and original within-frame row order/IDs. Use a lazy list model
  and retain the 200-row preview bound. This is ephemeral view state, never an
  inference call, export filter or model/archive setting. Label existing metrics
  as full-data metrics; Error norms retain their complete stored-row scope. Reset
  on new results and release data on invalidation; preserve steady-table behavior.

- Adaptive point inspection belongs to the connected RAR/importance sampler and
  reads retained idle-model/checkpoint data only. Never read a worker's tensors,
  regenerate samples or rescore residuals. Copy at most a 200-row page to CPU;
  retain order/duplicates, actual correction weights and the existing RAR versus
  importance anchor ordering. Derive only initial grid counts for archive roles,
  not coordinates/BC labels. Clear on busy/semantic/result changes, resolve owners
  by ID and close deleted owners. Keep evaluation points and archive formats intact.

- Keep examples.validate_cuda's explicit module/desktop-case manifest aligned
  with actual CUDA regressions; mocked GPU-unavailable checks do not certify
  CUDA execution. Only executed passes may produce a passed audit. Skips, expected
  failures and empty suites are incomplete. Public audit JSON includes source/runtime
  metadata and outcomes, never local tracebacks, usernames or absolute paths.
  Do not weaken numerical validators to accommodate stale small test presets.

- Research IntegralConstraint compares a scalar weighted field/flux total with a
  fixed scalar target, reducing quadrature before squaring. This differs from
  integral-of-squared-error mode. Require explicit finite nonnegative weights
  with positive finite mass, keep measure=1, detach/clone compiled quadrature and
  preserve scalar operator parameter gradients. Fingerprint actual weights/points/
  targets on continuation; keep point-only research checkpoint identity intact.
  Ordinary PINN/desktop methods must reject this research-only constraint. Never
  infer normals, units, mesh weights or a full-flow benchmark from the API example.

- Resource reviews use allocation-free architecture arithmetic and clearly label
  weight/Adam/hidden/encoded-input references, never a guaranteed peak/fit estimate.
  Available host/driver memory is a snapshot, not a reservation; quotas/caches
  matter. Preserve RNG, model/results, history and numerical settings. Warnings
  are advisory and do not silently resize experiments or bypass validation.
  Stage-specific OOM handling must preserve ordinary error diagnostics, release
  failed worker references and avoid presenting failed updates as checkpoints.
  Test injected failures rather than exhausting real memory. Supervisor QThreads
  do not enforce hardware quotas or guarantee protection from an OS-level kill.

- Native desktop jobs use isolated headless numerical processes; supervisor
  threads only transfer bounded data and Qt signals. Keep callbacks/GUI objects
  out of IPC: explicit configuration/reference declarations and CPU tensors only,
  restricted loading and atomic snapshots. Preserve session/event ordering, child
  identity/exit checks, 256 MiB transfer bounds and CPU retained handles. Never
  adopt partial or force-discarded results, including late transfer races.
  Keep cooperative Stop/checkpoint behavior, explicit force-stop consequences,
  parent-pipe death cleanup and workspace locks until the child is reaped.
  Retained GUI state is not a live PyTorch trainer; research APIs remain direct.
  Process isolation is not a hardware memory quota or a malicious-code sandbox.

- DSGNAR follows the explicitly documented paper Algorithm 1 acceptance rule,
  fresh CountSketch/SRCT operators, SVD trust-region probes and condition balancing.
  Preserve dedicated sketch RNG, attempt/rejection history, positive coefficients,
  stopping/stage state and strict physics/data/dtype continuation. No Adam step or
  implicit advanced-trainer composition. Keep fixed built-in scalar/steady rectangular 2D flow scope explicit.
  Derivative helpers must preserve the solution dtype rather than global defaults;
  check fresh-process float32/float64 CPU/CUDA continuation. No upstream source
  licence is declared in the inspected checkout; keep the implementation independent.

- DSGNAR probe interpolation fits ratio against log radius and locates its
  crossing without collapsing equal-ratio plateaus. Preserve the independent
  PCHIP oracle/bracket regression. Stage grace defaults to zero for old models;
  migrate absent values in config, node and checkpoint settings consistently.
  Distinguish matching an error threshold on an independent grid from reproducing
  the paper's complete protocol or hardware/runtime. Preserve superseded results
  with explicit labels, and validate qualifying checkpoints by fresh inference.

- The DSGNAR smooth-lid cavity is Re=100 with u=4*x*(1-x) on the lid.
  Keep the native pressure gauge/grouped-flow variation distinct from the
  research five-condition ungauged Appendix C.7 case. Label the actual SIREN
  parameter count; the paper caption and notebook disagree. Independent finite-
  difference references require steady and spatial-refinement checks and must
  not be labelled exact/Dedalus or used to claim the full paper protocol.

- Standing wave owns two-mode initial displacement and zero initial velocity,
  with separate losses sharing one initial geometric set; no final-time BC.
  Preserve explicit PDE-source wiring without duplicating initial rows. Keep
  fixed built-in baseline/DSGNAR scope and the single-window research variation
  explicit; future window transfer needs both displacement and velocity.
  Retain unsuccessful adaptive solves and raw condition losses alongside errors.

- DSGNAR phase hooks belong only in standalone isolated profiling workers, not
  active GUI jobs. Restore patched methods/functions on failure, verify checkpoint
  parity against an uninstrumented continuation and distinguish nested phase
  durations. Batch changes alter the realized sketch; do not claim cross-batch
  trajectory equivalence. Allocator peaks and lifetime RSS have different scopes.

- CountSketch CPU accumulation reduces only sorted occupied buckets, preserving
  each hash’s stable sum/order/normalization and private RNG. Keep CUDA dense
  below 4,000; full-count paired runs qualify occupied rows at ≥4,000. Check
  float32/float64 exact reference sums and full continuation before changing
  dispatch. Kernel-only 4,000-row measurements do not certify SVD or trainer speed.

- Cache unchanged PCHIP coefficients per radius crossing without altering its
  arithmetic, plateau treatment or 60-step bisection. Require exact legacy radii
  and same-checkpoint training comparisons. Large SVD profiles must record actual
  architecture/precision/counts, preserve whole-attempt peaks across phase resets,
  and reject publication when source changes during the run.

- Native pointwise inference uses at most 2,048 rows per forward with full CPU
  assembly. Preserve input order/IDs/duplicates, physical times and trained dtype;
  analytical output expressions use CPU float64. Research prediction/output APIs
  stay one-shot unless explicitly opted in, since custom models can couple rows.
  Require exact checkpoint/RNG preservation and tolerance-based output parity;
  GEMM shapes can change rounding. Keep total host/IPC costs and training
  microbatching limitations explicit; do not promise that batching prevents OOM.

- Owned-file clipboard snapshots use a versioned data-only MIME payload bound to
  the model fragment, with 16 MiB/128-file limits. Copy only referenced regular
  files with no-follow reads; paste into a fresh descriptor-anchored namespace.
  Preserve adjacent automatic Jorvik export versus explicit override, applied
  fingerprints and source alias metadata. Rewrite mapping bindings, never labels
  or expressions. Cross-project paste must survive source closure; Undo/Redo
  retains independent files. Reject incomplete/malformed bundles before graph
  mutation, keep external links external and forbid owned copies while busy.

- Importance sampling's optional pool='fixed' generates one uniform seeded box
  pool; rescore it at each event and keep per-event draw seeds and 1/(M*p) weights.
  Absent/refresh keeps historical candidate generation and checkpoint layout.
  Store and exactly verify fixed pool coordinates/dtype against regeneration;
  reject out-of-pool adapted draws and changed modes on resume. Preserve initial
  warm-up and unit boundary-anchor weights; do not infer imported quadrature or
  allow moving SA masks. Qualify each supported trainer's CPU/CUDA continuation.

- Flow gPINN explicitly differentiates momentum-x, momentum-y and continuity
  residuals separately, then averages their squared coordinate derivatives.
  Never differentiate a component sum before squaring. Preserve scalar arithmetic
  and signature; identify the three-component contract in flow checkpoints.
  Keep steady rectangular u/v/p scope, component BCs/gauge and physical-residual
  RAR scoring; do not infer transient/3D/imported vector support. Require an
  opposing-component cancellation oracle and higher-order parameter gradients.

- Coupled gPINN continuation checks keep weights, active samples and ordinary
  histories exact. CUDA raw/weighted residual-gradient diagnostic reductions may
  differ by a ULP between fresh graphs; qualify only those fields with at most
  two dtype epsilons relative tolerance and zero absolute tolerance. Keep the
  failed audit and measured discrepancy visible; never loosen solver validators
  or disguise a diagnostic tolerance as universal bitwise reproducibility.

- SCALE cavity reproduction lives in the separate research case, not the generic
  GUI presets. Preserve paper-shaped shared/three-head SiLU architecture and
  affine sine features; explicit derivative jets must match native ScalePINN
  losses and every parameter gradient. The compiled path is case-specific.
  Keep fixed-cloud mixed draws, interior-only PDE mask, summed velocity BC loss,
  no pressure gauge, cosine schedule and previous-before-update timing explicit.
  Reference interior velocities are evaluation-only; never replace raw boundary
  predictions in the primary metric. Do not bundle upstream code/reference data.
  Research checkpoints retain both networks, Adam, sampler RNG, schedule and
  reference/backend identity. Do not edit runtime/driver sources during active
  benchmark jobs; source guards refuse publication after changes. Report every
  seed/failed accuracy attempt and separate synchronized training, compilation,
  evaluation/I/O and wall time; no RTX3090/JAX parity claim from a local run.

- SCALE cavity refinements are explicitly changed protocols. Keep weights-only
  warm restart distinct from resume; record reset Adam/sampling, source checkpoint
  hash and cumulative updates/cost. Reference-selected snapshots must stay separate
  from final iterates and disclose full selection search cost. Curate every attempt,
  reload weights independently and preserve exact executed driver snapshots.
  Standalone centerline interpolation is a labelled benchmark diagnostic only;
  full-field metrics and raw VTU exports retain supplied coordinates/predictions.

- Importance sampling optionally scores the spatial-gradient magnitude of one
  selected steady output, without evaluating PDE residuals for those scores.
  Preserve residual defaults/legacy histories, detached scores, inverse-probability
  PDE correction and boundary anchors. Keep gradient statistics labelled separately
  in logs/retained inspection and strictly match the measure/field on resume.
  Validate field indices and reject transient field gradients until their
  coordinate/time scaling contract is specified; never pool velocity and pressure.

- Model format 33 adds a boolean `enabled` setting only to optional outputs;
  missing flags in old models mean enabled. Disabled outputs retain wiring but
  request no data/monitor recording and cannot export. An enabled export needs
  its upstream Comparison enabled. Reject flags on required physics, malformed
  flags and archives carrying disabled-output results/monitor records. Keep
  toggles guarded while busy, undoable and portable; invalidate old results while
  preserving matching weights for explicit evaluation without optimizer updates.
  The Disabled badge is derived from the setting, not serialized UI freshness.

- Research CosineDecay is closed-form in completed updates with a positive
  relative floor per optimizer group and power applied before adding that floor.
  Clamp at the horizon, then hold; never auto-stop or restart a cosine cycle.
  Keep StepDecay and unscheduled identities unchanged, use scheduled format 2
  with exact kind/settings/rate-history validation, and retain same-device Torch
  RNG/optimizer continuation. Custom steps still own their schedules/state.

- Research pointwise_microbatch declares deterministic row-independent network,
  residual and point operators on fixed data, built-in mean MSE and default
  Trainer updates only. Scale each chunk by its fraction of that term's rows;
  accumulate before one optimizer/schedule step. Reject global integrals,
  custom losses/steps/subclasses, stochastic/batch-dependent layers and adaptive
  hooks before sampling. Validate complete target lengths before slicing. On a
  bad chunk/gradient, clear accumulated gradients without committing a step.
  Use research format 3 with exact chunk-size identity; preserve default formats
  1/2. Full/chunked arithmetic is tolerance-qualified, while same-chunk resume is
  tested exactly. Never auto-size after OOM or infer support for advanced trainers.

- Job time limits are local application preferences, default zero/unlimited,
  snapped into each isolated train/resume/evaluate transport. Keep physics,
  model/archive formats, results and undo history unchanged when editing the
  preference, and lock edits while busy. Limit child launch-to-exit time only;
  parent packet work and explicit file exports/editors are excluded. On expiry
  kill/reap and discard the entire job without adopting partial state; retain
  prior saved files. Cancel/join timers on all paths, including thread-launch
  failure, and never expire an already-finished child during parent decoding.
  Never hold the deadline lock across control-pipe writes; broken-pipe cleanup
  must not mask the primary diagnostic or skip closing other streams. This is not
  a RAM/VRAM quota or a policy for direct research training loops.

- The optional Burgers reference example owns a fixed evaluation-only Cole–Hopf
  CSV shared by coordinate/reference mappings. Never feed its values into training
  or imply it changes with edited physics. Preserve the ordinary reference-free
  example, explicit comparison outputs, exact IDs and physical-time VTU/PVD
  frames. Distinguish reference-evaluator validation from trained PINN accuracy.

- Use Jorvik as the current product name in docs and help. Preserve historical
  upstream source paths/licence references and legacy schema keys; keep old
  planar-adapter imports/commands working through compatibility aliases.

- MCP resume accepts only a checkpoint already retained/open in the window. Require
  fresh session/revision and a separate ephemeral run ID, since manual training
  can replace weights without changing graph settings. Read metadata without
  copying tensors or retaining obsolete checkpoints; reuse native Resume, its
  undoable total-update change and isolated-worker validation. Track the resulting
  revision/new job ID, keep Training permission gates and strict integer bounds,
  and expose no checkpoint file loading or implicit export.

- Parameter studies support bounded Cartesian preview and version-1 portable
  save/open and serial fresh execution through `margam-cli`. Resolve actual block IDs/parameters through the
  existing decoder/graph/config validators; never retarget deleted IDs. Preserve
  model, Undo, retained results and RNG during preview/save; allocate no networks
  or samples. File writes are limited to explicit save and private restore.
  Keep invalid trials with reasons and no metric/budget, bound enumeration before
  copying trials, and label update/parameter counts as workload rather than fit
  or runtime predictions. Use the shared executable graph/output gate and native
  isolated child for CLI trials; do not introduce a second training loop. Portable bundles freeze inputs, graph, axes and execution intent;
  never implicitly include external links or rewrite accepted-import history.
  Opening explicitly replaces the graph using a validated private workspace;
  preserve app preferences. Verify bounded schema/checksum/path/ZIP handling,
  atomic no-clobber save and stable content/trial identities.

- CLI studies require fresh output folders, stable definition/trial identities and
  actual resolved settings/source/runtime/input provenance. Overrides are explicit.
  Stage native archives/requested exports before publication; no fake metrics on
  invalid/incompatible/failed/interrupted trials. Export files only with --export
  and enabled connected export blocks. Source changes prevent uncertain artifact
  publication and abort remaining trials. Ctrl-C reaps/discards the active job and
  preserves completed artifacts. POSIX grid resume requires unchanged frozen study,
  Python sources and recorded runtime/device/settings. Verify all completed payloads
  before mutation; restart unfinished numerical attempts fresh, keep prior attempts,
  retry failed trials explicitly and retain the process lease in native children.
  Do not imply mid-trial checkpoint recovery or Slurm preemption integration.
  Keep CLI/child Qt-free and reuse the native archive/data restoration contract.


- Study collection is read-only version-2 metadata inspection, independent of the
  current source/runtime/device. Label outcomes/errors as reported; do not rank
  different fields/problems or fabricate missing metrics. Keep model/results/RNG
  unchanged while browsing. Explicit native opening must verify frozen provenance,
  selected publication and payloads, then use a verified private archive copy and
  the normal discard/data restoration workflow. Reject stale progress/unsafe paths.
  Cancel/join background readers on browser/window close and ignore stale queued
  thread callbacks; never deserialize numerical archives in metadata readers.


- Slurm export is a data-only, no-submission single-node serial bundle. Require
  explicit CPU/memory/time budgets, sanitize partition/account tokens, freeze
  study/script hashes and preserve fresh-output refusal. Use SLURM_SUBMIT_DIR,
  never the batch spool script path; launch only the existing native runner.
  CPU/GPU intent is explicit. Resume/retry remain opt-in and use original strict
  provenance. Do not claim actual allocation enforcement, automatic requeue,
  signal checkpoints, mid-trial recovery or real scheduler validation from local
  Bash tests. Preserve the existing root environment and never embed local paths.


- Saved-prediction comparison is explicit and read-only. Require exact coordinate,
  field and typed-ID order; preserve duplicate rows and physical times. Do not
  interpolate, rank or infer units/physics equivalence. Use per-field differences
  with A as relative denominator and separate saved-reference errors; retain
  undefined/overflow states. Metrics cover all rows, previews at most 200.
  Read stable bounded private CPU snapshots without network/optimizer restoration,
  preserve CPU/CUDA RNG/model/results/history, and cancel/join readers on close.
  Keep native full archive restoration unchanged; container inspection is not a
  claim that an optimizer checkpoint has been qualified for continuation.


- Local GUI studies launch the existing CLI with a reviewed private input copy;
  never create another numerical executor. Preserve model/Undo/retained results/RNG
  and application preferences; block overlapping jobs/editors using existing busy
  guards. Resume binds reviewed execution metadata under the results lease and
  uses recorded settings. Stop interrupts the CLI; force-stop targets only its
  newly owned POSIX session/group, never PIDs read from files. An owner pipe must
  stop jobs after GUI loss. Bound progress/log reads, join review threads and clean
  controller-owned temporary files after execution. Hiding controls leaves a job
  running; main-window close stops it before the normal discard check. Do not
  imply parallel scheduling, OS quotas or mid-trial checkpoint recovery.


- History overlays use explicitly saved Training Monitor records only; never
  substitute checkpoint history, renumber retained indices or reconstruct old
  weighted contributions from current coefficients. Split lines at absent updates,
  missing terms and nonpositive log values; retain exact values in the preview.
  Keep raw/weighted/total semantics distinct, bound rendering and preview, preserve
  model/results/RNG and the shared restricted CPU-reader lifecycle. Do not rank
  objectives or infer physical time/accuracy from training-update indices.


- Network-family study axes are concrete pointwise `network_type` string choices,
  Fixed/Candidates only. Explicit choices reset base custom options, including the
  base family, using complete frozen family snapshots. Share `apply_choices` between
  preview and native trials. Version 2 protects selected-family defaults in study
  identity; preserve them on open/preview/resave and reset with Clear draft. Reject
  incomplete/unknown schemas instead of reinterpreting them. Numeric-only version-1
  studies keep their identities/custom options. Retain invalid trial reasons with
  no fake budget/metric; do not imply recipe/topology or neural-operator alternatives.


- Desktop Slurm export consumes an explicitly selected saved study, never the
  current canvas or unsaved study draft. Reuse the data-only export_job writer,
  require explicit nonzero resource budgets and a fresh output directory, and
  preserve model/Undo/results/device/RNG. Export never submits, launches training
  or installs packages. Keep local Bash qualification distinct from real cluster
  submission and allocation enforcement.


- Study table CSV exports the collected metadata snapshot, not fresh disk state
  or numerical payloads. Include all trials and all output diagnostics, preserve
  absent/undefined versus zero, choices, reasons, attempts and snapshot provenance.
  Encode arbitrary text as JSON cells to preserve content without spreadsheet
  formulas. Publish atomically with no-clobber and a bounded size; never rank
  trials, infer compatibility or imply payload verification. Disable export while
  reading or after a failed refresh, without changing model/results/RNG.


- Paired studies use version 3 with `combination="paired"` and a complete selected
  family-default map (empty without architecture axes). Equal-length non-singleton
  lists pair by position; singletons repeat. Repeated values/rows are explicit
  paired trials; keep Cartesian duplicate rejection. Reject mismatches instead of zip
  truncation. Share combination resolution across preview/native trials and use
  the same total in run/resume/collection. Preserve Cartesian v1/v2 bytes and trial
  identities. Invalid pairs retain row numbers and reasons. Bound paired grids to
  64 rows/8 axes without applying a Cartesian-product limit; conditional/topology
  alternatives remain deferred.


- History overlays accept two–eight explicit archive/monitor selections. Reuse
  retained_curve and the restricted CPU snapshot reader, preserve every run's
  original indices/gaps and raw/weighted semantics, and bound plot/preview. Slot
  changes invalidate the overlay and are blocked during reads. Removing the last
  slot releases its snapshot; close clears all retained runs and joins the reader.
  Keep two-run prediction comparison unchanged and never rank objectives.


- Resource probes use the native isolated worker's training loop with an explicit
  bounded update/attempt stop budget; preserve the original configured schedule
  horizon. Never restore/adopt a checkpoint or evaluate outputs. Measure fresh
  construction/update CUDA allocator peaks and whole-process CPU RSS with their
  distinct scopes. Count only observed residual-sampler events and never extrapolate
  later growth/outputs. Preserve inputs/model/RNG, source guards and transport
  timeout/cancellation; publish no measurement after failure/cancel/source change.
  Expert multi-window probes use version 2: the runtime-only transition observer
  wraps the unchanged native handoff, with exact CPU/CUDA state parity. Keep
  construction/training/transfer phases ordered and bounded to 64. Include
  internal window snapshots and retained teacher device copies, aggregate CUDA
  peaks by maximum, and label RSS cumulative. Never sum phase peaks or imply
  synchronization leaves overlap/allocator reuse unchanged. A boundary-only stop
  does not claim a transfer; preserve pending status and original horizon. The
  update cap is 1000 single-window or 100000 marching, bounded by that horizon.


- Desktop resource probes review an immutable saved-study trial, not the canvas.
  Reuse StudyProcess and the probe CLI, including owner-stdin monitoring; allocate
  no network during review. Use the application device preference, require new
  review after settings/device changes, and keep the explicit probe limit local.
  Include the probe in shared busy/Stop/Force-stop/main-close handling; hiding its
  controls leaves it running. Discard reports after stop/failure/mismatched
  provenance. Release frozen copies/readers and preserve current weights/results/
  model/Undo/RNG; never adopt probe numerical state or write a results folder.


- Prediction groups contain two–eight explicit archive/output selections against
  baseline A. Reuse compare_predictions for every pair with its exact coordinate,
  field, duplicate-row and typed-ID contract; reject the whole group on any later
  mismatch, naming the run. Preserve separate own-reference diagnostics and A
  denominators, never rank or infer unit/physics equivalence. Group math runs in
  the shared CPU reader with cancellation between pairs and bounded previews;
  discard cancelled/failed results. Share concrete run-slot widgets with history
  overlays, release snapshots/reader closures on main-window/dialog close, and
  preserve current model/results/Undo/RNG without network/optimizer restoration.
