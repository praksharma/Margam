# Adding a PINN method

[Documentation index](README.md) · [Capabilities](components.md) · [Repository rules](../AGENTS.md)

A new parameter value or PDE does not automatically mean a new method. First check
whether existing components and a new physics example suffice. A method owns a
substantive change in network, loss, sampling, constraints or training procedure.

## Ownership

| Work | Location / existing example |
| --- | --- |
| Distinctive numerical method | One file in `margam/methods/`; [SA example](../margam/methods/self_adaptive.py) |
| Construction in both UI forms | One file in `margam/recipes/`; [SA recipe](../margam/recipes/self_adaptive.py) |
| Reusable implementation | Existing shared problem, sampling, loss, network or differentiation module |
| Concrete PDE/BC/data/reference setup | `margam/templates/`, exposed in File → Examples |
| Numerical/equivalence/persistence checks | `tests/` |
| Runnable comparison | `examples/` |
| Method guide and measured evidence | `docs/`; curated JSON in `assets/benchmarks/` |

Keep the distinctive algorithm understandable in its one method file. Reuse actual
shared behaviour; do not copy the baseline engine or create a universal extension
framework. The shared baseline `loss_samples()` supplies physical predictions and
targets, allowing SA to own its per-point loss/min-max loop independently.

## What the recipe file defines

The baseline and SA files define names/root kind, component-owned fields, component
placement, internal connections, external ports, defaults, compact exposed settings,
and split/combine operations. Both insertion forms read this same definition.
A collapse must preserve component parameters and external physics/output wires.
A shared component cannot be hidden if doing so removes another block's input.

There are currently explicit maps for the two supported compact/trainer types in
`margam/recipes/__init__.py`, plus concrete UI kinds/ports/parameter ownership in
`margam/ui/`. Adding a file does **not** automatically register a new UI method.
Extend those concrete contracts only as required by the next real method. Numeric
runtime selection currently lives in the scalar/flow template builders.

## Before calling the method implemented

1. Document its equations and primary-source attribution, defaults, supported
   replacement points, differentiability requirements and unsupported features.
2. Check the distinctive update mathematically on a small problem. Add a concrete
   accuracy check with a valid explicit reference; do not rely on UI startup.
3. Verify equal settings in compact/expanded forms, actual execution, graph history,
   model persistence and external connections. Keep outputs optional.
4. If there is additional training state, persist and validate it for exact resume.
   If resume is unsupported, expose that limit explicitly rather than saving an
   incomplete checkpoint. Reject changed samples/options where state depends on them.
5. Compare to a stated baseline at equal data, initial seed and update budget. Save
   per-seed records and failures, not only an attractive average. Report runtime
   separately if making an efficiency claim.
6. Update the capability row, example catalogue, method/recipe guide, validation,
   results index and TODO. Keep proposed scope distinct from working support.

## Recording a study

Save equation, domain, BCs, explicit reference, training/evaluation points, sampler
and seed, network/initialisation, all learning rates/weights, stopping budget,
mask/transformation options, dtype/device/thread count and environment. Include
per-seed RMSE/relative L2/max error where meaningful and explain the reduction.
State whether losses are pre-update and predictions post-update. For coupled
quantities, report per-field errors and scaling.

Record source revision and whether it was modified; source hashes help identify
changed files but are not a complete source archive. New measurements get new
filenames. Preserve older records and identify what was rerun, imported or only
checked against a regression threshold. State the tested limitations, especially
unreproduced paper protocols and unavailable hardware.
