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
- Basic interval/rectangle sampling must work directly. Advanced CAD/geometry
  workflows belong to future GeoSampler integration. Do not add a 3D viewport.
- Views are optional and owned by their blocks. Do not show global default figure
  panes or assume an analytical solution. Only expose supported context actions.
- Local sample/geometry previews currently belong to Rectangle blocks only.
  Keep operational progress/errors separate from numerical output views.
- Do not continue polishing or expanding in-app field heatmaps without a concrete
  user request. Preserve existing behaviour unless the user requests removal.
- ParaView file export was explicitly requested and is supported through its own
  opt-in block. Viewer embedding/launching and arbitrary mesh export remain deferred. Concrete
  2D transient flow is supported by the Expert suite; do not infer generic 3D support.

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
- Coupled u/v/p results currently use tables/CSV. Preserve the pause on heatmap
  development. Full 3D/non-planar GeoSampler remains deferred. ParaView export is
  opt-in; never infer physical time from training updates. Supplied-node evaluation
  and richer mesh-aware exports are planned in docs/todo.md, not implemented yet.

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
