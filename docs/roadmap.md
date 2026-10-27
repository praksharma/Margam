# Margam: a visual framework for physics-informed neural networks

Discussion roadmap · 26 October 2026

This file preserves the original discussions and staged intentions. Historical
statements below such as “not implemented” describe their discussion stage.
For current support and limits, use [README](../README.md),
[components](components.md), [UI implementation](ui-prototype.md), and
[remaining TODOs](todo.md). The 27 October implementation includes owned graph
parameters, scalar/steady-u-v-p physics, explicit outputs, model/run persistence,
checkpoint resume, clipboard/history, recovery and a planar GeoSampler bridge.


## Purpose

Margam will let users construct, train, and inspect physics-informed neural
networks (PINNs) through connected visual components, in the spirit of Simulink.
The visual interface should expose an understandable numerical model: geometry,
sampling, networks, physics residuals, constraints, losses, and training.

Before building the canvas, we need an implementation structure and documentation
that remain understandable as methods and components accumulate. A developer
returning after three months should be able to answer:

- Is this PINN method already implemented, perhaps under another name?
- Which sampling strategies, optimisers, and other components already exist?
- Can an existing component support this method, and what are its limitations?
- Where does the method's distinctive behaviour belong?
- Is an entry working, experimental, or only planned?

## Agreed principles

### One file per PINN method

Each method has one clearly named file containing its distinctive preparation,
loss evaluation, sampling decisions, training procedure, and prediction behaviour
where applicable. Someone reading that file should be able to understand what
makes the method different without following a chain of method-specific files.

This is a unit of ownership, not a requirement to duplicate the entire engine.
Methods reuse shared components and retain control over their own execution.

Examples below are proposed names, not claims that these methods exist:

```text
methods/
    baseline_pinn.py
    adaptive_sampling_pinn.py
    gradient_enhanced_pinn.py
```

### Shared components live separately

Provide reusable sampling, optimisation, loss, network, geometry, and
differentiation components in separate files or existing packages. Methods use
these defaults where suitable and can replace them when necessary.

For example, an adaptive sampling method could reuse the standard network and
optimiser while defining its own sampling updates and training loop in its
method file.

Replacement does not require inheritance. Functions, factories, and objects can
provide interchangeable behaviour. Avoid forcing every method through a rigid
base trainer or adding method-name switches throughout shared code.

Extract shared behaviour when there is a concrete reason to share it. Do not
build abstractions for every hypothetical future PINN.

### Methods and templates are different

- A **method** defines how a PINN operates, including distinctive losses,
  sampling, or training behaviour.
- A **component** supplies a reusable capability such as a sampler or optimiser.
- A **template** is a configured example combining a method, components, and a
  physical problem.

“Poisson with baseline PINN” and “heat conduction with baseline PINN” are two
templates using the same method. A different parameter value does not by itself
create a new method. A reusable sampler does not automatically require its own
method either.

The exact boundary for combinations of network, loss, sampler, and training
changes remains a design question. Document the reason whenever introducing a
new method rather than a component or template.

## Documentation and discovery

### A catalogue is part of the implementation

Maintain a searchable inventory of methods and shared components. The future UI
should display the same catalogue information that developers consult; avoid
maintaining an independent UI list that can drift from the implementation.

The initial catalogue can be lightweight. Its storage format and discovery
mechanism remain to be chosen. Automatic discovery is not a prerequisite for
documenting what exists.

Every catalogue entry should identify its implementation location and distinguish
implemented, experimental, planned, and deprecated capabilities. Planned entries
must not appear as usable implementations.

### Method documentation

Each method file should start with a concise description. Its catalogue entry
should include:

| Field | What it answers |
| --- | --- |
| Stable identifier, name, aliases | Is this already here under another name? |
| Description and reference, where applicable | Which formulation is implemented? |
| Difference from baseline | What makes this a distinct method? |
| Status and implementation location | Can I use it, and where is its code? |
| Shared components used | What is reused? |
| Custom behaviour | What does this method replace or add? |
| Supported problems and limitations | Does it fit my use case? |
| Settings and defaults | How do I configure it? |
| Runnable example and validation | What demonstrates that it works? |

An implemented method should have a runnable example. Claims of numerical
correctness should link to relevant checks or recorded evidence, with their
limits stated.

### Shared component documentation

Every reusable component needs a discoverable entry explaining its purpose,
location, interface, configuration, restrictions, and an example of use.

For samplers, record in particular:

- Supported geometry, dimensions, and regions: interior, boundary, initial, or data.
- Inputs and output representation, including coordinate shape.
- Whether sampling is fixed, repeatable, or refreshed during training.
- Seed behaviour, device and dtype expectations, and gradient requirements.
- Any adaptive state, required residual information, and known limitations.
- Methods or examples that use the sampler.

For optimisation components, record the accepted parameters, state ownership,
step or closure requirements, and which training procedures support them.
Offering an optimiser does not imply every method can use it.

The catalogue must describe actual supported behaviour. An importable class or a
completed checkbox alone is not evidence of compatibility.

### Workflow for adding a method

1. Search the method catalogue by name, aliases, reference, and behaviour.
2. Search the component catalogue for required capabilities.
3. Check interfaces and limitations before deciding anything is missing.
4. Reuse suitable components. Keep behaviour unique to the method in its file.
5. If a reusable capability is missing, add it to the shared components and
   document it in the same change.
6. Add the method file, catalogue entry, runnable example, and appropriate
   validation. State unsupported features explicitly.
7. Add a visual template once the method is usable through the visual builder.

Updating documentation is part of adding or changing a capability, not a later
cleanup task. Remove or update stale catalogue claims when interfaces change.

## Proposed organisation

This is a conceptual layout. Reuse the current engine's suitable packages rather
than relocating working code merely to match this tree.

```text
methods/                 One implementation file per PINN method
components/              Reusable defaults and contracts
    sampling.py
    optimisation.py
    losses.py
networks/                Reusable network implementations
geometry/                Domains and geometry operations
docs/
    methods/             Method catalogue and usage documentation
    components/          Component inventory and compatibility information
templates/               Configured problem examples and, later, saved diagrams
examples/                Runnable examples
tests/                   Behaviour and numerical checks
```

The catalogue's authoritative metadata location is still undecided. Choose one
source and derive other views from it where practical.

## Roadmap

### 1. Inventory the existing engine

- Catalogue existing methods, samplers, optimisation support, networks, losses,
  geometry, and differentiation utilities.
- Separate current implementations from unfinished refactors and legacy APIs.
- Reconcile documentation and TODO claims with actual code and examples.
- Record what is missing instead of treating planned features as available.

**Done when:** a developer can check whether a needed capability exists and find
its implementation, usage, and limitations without opening the UI.

### 2. Establish method and component contracts

- Use the existing baseline PINN as the first documented method.
- Define the minimum interfaces between problems, sampling, networks, residuals,
  constraints, training, and results.
- Make ownership of sampling updates and optimiser state explicit.
- Describe how a method supplies custom behaviour without changing unrelated
  methods or forcing everything through one training loop.
- Check the design against a concrete second method before generalising it.

**Done when:** another method can be added in one method file, with documented
reuse and any necessary new shared components.

### 3. Define the visual model

- Define blocks and the meaning of their connections.
- Specify validation for coordinate dimensions, network outputs, derivatives,
  constraints, and method/component compatibility.
- Define how a saved diagram becomes an executable PINN configuration.
- Define how custom Python residuals and method-specific settings are exposed.
- Decide the saved-model format and how changes to it will be handled.

**Done when:** the existing Poisson example can be described unambiguously as a
diagram, with clear errors for invalid connections.

### 4. Build one complete visual workflow

Create a 1D Poisson template covering interval geometry, interior and boundary
sampling, a network, PDE residual, boundary conditions, training, and a solution
plot. Provide the method catalogue and a template browser with distinct roles.

**Done when:** a user can load, configure, train, inspect, save, and reopen this
model, and its execution agrees with the equivalent Python configuration.

### 5. Expand through real use cases

Add methods, components, and templates as concrete requirements arise. Keep the
catalogue current and verify that new methods can be added without repeatedly
changing the visual builder or unrelated method implementations.

## Geometry, data import, and boundary-condition workflow

### Simple domains belong in Margam

A user solving a problem on a square or rectangle should not need an external
geometry file or another application. Provide built-in domain creation from
bounds, interior sampling, and separately selectable sides for boundary sampling
and condition assignment. Retain the interval workflow for 1D examples.

Generate meshless point samples; a mesh is not a required training input. Expose
sampling settings supported by the chosen sampler, such as point count and seed,
and preview interior and boundary sets before training.

### Boundary point sets come before boundary conditions

Represent boundary data as a collection of named point sets. The number of sets
determines how many boundary entries are available; do not assume a fixed number
of BCs. Each entry initially contains its points. The user assigns the condition
type, field/operator, and target afterward, for example Dirichlet temperature or
Neumann flux. A boundary label alone does not prescribe physics.

For built-in rectangles, let users select sides and create the required sets.
For imported samples, preserve supplied labels. Check additional requirements
when a condition is assigned: a normal-derivative condition needs normals, which
must be computed from known geometry or explicitly supplied, not guessed from
arbitrary coordinates. Define corner handling for conflicting side conditions.

### Prepared data and translation scripts

Accept prepared PDE and boundary point sets as another input route. File formats
such as NPZ, NetCDF, and CSV do not determine what their arrays mean. Define a
small documented internal contract and allow a short Python adapter to read a
user's files, map coordinates and labels, and return that representation.

Provide adapter examples and validate dimensions and point-set names. Preview
the interpreted data. Keep file translation separate from assigning BC types or
defining the PDE. Convenient import forms can follow for common layouts; do not
require the initial UI to infer arbitrary file contents.

### Complex geometry: future GeoSampler route

Use [GeoSampler](https://github.com/praksharma/GeoSampler) for the future complex
geometry workflow: bring geometry with selectable edges/faces, select and label
regions, and generate meshless interior and boundary samples. Assign the physics
in Margam afterward. This is a future integration, not an implemented feature or
a requirement for simple rectangular problems.

The integration mechanism remains open: importing GeoSampler's labelled exports
is one possible first step; an embedded or connected geometry interface can be
considered later. Inspect the actual export contract, including normal support,
before committing to an implementation.

## Initial PDE data scope and deferred work

Start with a simple PDE interface:

- **PDE definition:** the residual function describing the equation.
- **PDE points:** the coordinates at which the residual is evaluated.

Additional PDE inputs, such as force values supplied at each point, are deferred.
This decision concerns the PDE interface; boundary-condition data remains a
separate concern.

### TODO

- [ ] Reuse suitable GeoSampler foundations in stages: theme/dock persistence,
  unit-domain sampling functions and catalogue, then region metadata and project
  lifecycle patterns. See `geosampler-reuse.md` for inspected sources, adaptation
  requirements and provenance. Keep the 3D/WebEngine/CAD stack out of the basic UI.
- [ ] Provide built-in square/rectangle creation, interior sampling, selectable
  sides, boundary sampling, and point previews without external geometry files.
- [ ] Define the boundary-point-set collection and subsequent BC assignment UI,
  including targets, normal requirements, and corner handling.
- [ ] Document the prepared-data contract and provide small translation-script
  examples for external datasets.
- [ ] Add the future GeoSampler route for complex geometry with selectable
  edges/faces and meshless sampling; decide the integration mechanism separately.
- [ ] Add optional PDE input data, including pointwise forcing, after the initial
  PDE-definition-and-points workflow is established. Define how input values align
  with PDE points and how the residual receives them.

## Current baseline and boundaries

The repository is `/home/hell/Desktop/Margam`, with the Python package `margam`.
The baseline PINN, benchmark runner, numerical components and behaviour tests are
retained. The older borrowed geometry stack and compatibility API are retired;
new simple-domain sampling is planned. The desktop builder is a layout prototype
and does not yet execute a PINN graph.

Keep `dev` untouched. Source lives in `margam/`, UI code in `margam/ui/`, planning
and usage documents in `docs/`, screenshots in `assets/screenshots/`, and the
single active development environment at the repository-root `.venv`.

## Decisions still open

### Desktop UI proposal

Use PySide6 with Qt Widgets and a Graphics View canvas for the initial desktop
prototype. The layout has a left block library, central 2D connection canvas,
right properties panel, and bottom 2D domain/sample preview and messages. No 3D
viewport is needed for this project. See `ui-prototype.md` and `margam/ui/app.py`
in this repository. This is a layout study; executable port semantics,
model persistence, and backend integration remain future work.

- What precise changes warrant a new named method?
- Where should authoritative catalogue metadata live, and how will the UI read it?
- What is the smallest useful component interface for the first two methods?
- How should method-specific behaviour appear in the visual model?
- How will user-defined residuals and components be represented and saved?

The immediate next step is the component inventory and catalogue, followed by
the contracts. Canvas layout comes after those foundations are clear.

## Implementation update: package cleanup

The package is now `margam`, with the baseline and benchmark engine retained.
The old geometry/domain compatibility API is retired. UI source is in
`margam/ui/`, planning documents in `docs/`, and the environment is root `.venv`.
The migration history is recorded separately.
See `cleanup.md` for the migration record and `todo.md` for current work.

## Baseline UI integration update

The block layout now runs a template-backed baseline Poisson problem on intervals
and rectangles. Properties edit geometry, sampling, constant source, side BCs,
network and training settings. Named point sets are generated before BC assignment.
The UI validates the supported connections and runs the existing method in a
background thread, with live losses, cancellation and safe close handling.
Intervals have analytical reference curves and RMSE; rectangles have a predicted
field heatmap without a reference error claim. Grid/random samplers are implemented.
General graph compilation, custom PDE/data adapters, persistence and broader
sampling remain future work. See ui-prototype.md and components.md for the current
contracts. Completed stages should be checkpointed in small, verified commits.

## UI refinement: explicit output and comparison blocks

The next output design should follow the model's declared intent. Training must
be valid without a plot or results block. It produces a trained model and run
information; connected output blocks decide which predictions to evaluate,
visualise or export. Loss plots should also be requested explicitly. Result panes
should be created or shown for those outputs, rather than being mandatory default
views. Operational progress, validation messages and failures remain available.

An exact solution is not assumed. A reference source may supply an analytical
function, measurements or another solver's data. A separate comparison block
pairs compatible reference values and predictions at explicit evaluation points
and requests metrics such as RMSE. A solution plot can show predictions alone.
A demo template may include reference/comparison blocks, but must make them visible
and optional. The existing automatically constructed interval reference is demo
behaviour to replace, not a general PINN interface contract.

These are planned refinements: the current UI still pre-creates results tabs,
requires the Solution block, and automatically compares the interval preset.
Add standard File, Edit, View and Help menus alongside this work; see todo.md.

## Component library and view organisation

Margam is a PINN builder across physical problems, not a heat-transfer application.
The left pane should be a searchable component library grouped by responsibility:
Geometry / Data, Sampling, Physics / Conditions, Networks, PINN Methods, and
Outputs / Comparison. Templates remain a separate discovery concept.

Sampling has two independent choices: the point-set target/purpose (interior or
boundary) and its numerical method. Each point set should own its recipe and
settings. For example, a rectangle interior can use random sampling while one
boundary uses a grid and another uses a different supported recipe. Boundary
sampling creates coordinates; the BC block subsequently assigns the physical
condition. Method choices and compatible targets come from the documented
component catalogue rather than separate hard-coded UI inventories.

Margam must supply basic interval/rectangle grid and random sampling directly,
including bounds/side selection, counts, seed and preview. Advanced CAD sampling,
face/edge selection and richer region preparation are reserved for GeoSampler.
The current template shares one sampler selection; independent recipes are planned.

Training Monitor, Solution Plot and Reference Comparison are explicit optional
outputs. Open their views from their blocks. Start with bottom-docked tabs, allow
floating windows or docking elsewhere, and separate closing a view from removing
its model block or ending training. Do not automatically create result views.
Reference comparison must remain optional across every physics template. View
visibility and retention of sampled output data need separate policies.

## Simulink evidence baseline

See [simulink-evidence.md](simulink-evidence.md) for the official R2026b source
register and proposed interaction contract, collected 26 October 2026. Modern
Simulink's toolstrip and standalone/scope-container plots differ from our chosen
conventional menu bar and initially bottom-docked output views. Preserve familiar
interactions while specifying PINN dataflow independently; do not equate training
iterations with physical simulation time. The document separates observed
product behaviour, Margam adaptations and gaps requiring hands-on validation.

## Implemented: explicit output ownership and standard menus

The next output stage is now implemented: the default model has no plot blocks or
views. Optional monitor/prediction/comparison requests are validated independently
from the seven required Poisson core blocks. An explicit Reference block supplies
the analytical interval demo; the generic worker never assumes an exact solution.
Output views open only on request and can hide, float, redock and reopen. Per-output
history/evaluation settings are independent; graph/model edits invalidate stale
runs. File/Edit/View/Run/Help expose only implemented commands. Persistence, export,
general references, a grouped catalogue and instance-owned core sampling remain
future work. Current behaviour is documented in ui-prototype.md.

## Implemented: narrow block context actions and rectangle previews

Blocks now expose supported actions through right-click menus. Rectangle blocks
provide optional local geometry/sample previews; output blocks provide their own
Open actions. The default figure pane is removed. Messages are independent of
figures. The rectangle previews float initially, can be hidden/reopened or docked,
and are removed when their owner disappears or changes to an unsupported shape.
No interval/general/3D sample renderer or generic capability framework was added.
Build further abstractions only when a concrete second use case justifies them.

## Implemented: canvas navigation and ready-wired heat demo

Canvas controls now include pointer-centred wheel zoom, mouse/Space panning, 100%
and explicit fit. The separate Heat demo loads a baseline steady heat problem and
connected optional monitor, prediction, reference and comparison blocks. Its 2D
reference is a labelled truncated sine series for equal Dirichlet edges, not an
assumed solution for arbitrary problems. The basic template remains output-free.
See heat-demo.md for the formulation and ui-prototype.md for controls.
