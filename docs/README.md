# Margam reference catalogue

For task-based help with current screenshots, start with the [illustrated user manual](index.md)
and [Your first PINN](start/first-pinn.md). F1 opens the built manual in the app.

Current implementation and recorded evidence, reviewed 28 October 2026. Start with
[setup and commands](../README.md); historical design discussions are labelled below.

## Using Margam

| Question | Read |
| --- | --- |
| Can I follow a screenshot walkthrough? | [Your first PINN](start/first-pinn.md) |
| How do I build, train, inspect and save a model? | [Desktop guide](ui-prototype.md) |
| Which complete problems are available? | [Examples and predefined PDEs](problem-examples.md) |
| How do compact/expanded PINNs work? | [Recipes](pinn-recipes.md) |
| Which pointwise networks are available? | [Neural networks](neural-networks.md) |
| How does learning-rate annealing work? | [LRA method and safeguards](lra-pinn.md) |
| How do I integrate squared PDE/BC errors over the geometry? | [Integral loss formulation](integral-loss.md) |
| How do residual refinement and importance sampling work? | [Adaptive sampling](adaptive-sampling.md) |
| How does SA-PINN work, and which settings matter? | [SA-PINN method](self-adaptive-pinn.md) |
| What is the paper Helmholtz example? | [Fourier/LRA Helmholtz](helmholtz.md) |
| What is the steady heat problem/reference? | [Heat demo](heat-demo.md) |
| How do I export predictions to ParaView? | [Explicit export and playback](paraview-export.md) |
| How do prepared GeoSampler points enter Margam? | [Planar data contract](geosampler-contract.md) |
| What is implemented and what are the limits? | [Component capability index](components.md) |

## Results and verification

| Question | Read |
| --- | --- |
| What numbers did we actually measure? | [Results catalogue](results.md) |
| Where are the exact records? | [Curated evidence files](../assets/benchmarks/README.md) |
| What do tests establish? | [Validation scope and commands](validation.md) |
| How did hard Dirichlet compare with soft BCs? | [Heat accuracy study](heat-accuracy.md) |

A measured error, a test's acceptance threshold, and an implementation claim are
different kinds of evidence. Result pages state the problem, method, seed, training
budget, evaluation/reference, runtime and limitations. Historical GPU data does not
replace testing the current CPU build on CUDA.

## Extending the project

- [Development and experiment commands](commands.md)
- [Numerical API and benchmark runner](engine.md)
- [Adding a PINN method and its one-file recipe](adding-a-pinn.md)
- [Agent/repository instructions](../AGENTS.md)
- [Remaining work](todo.md)

Keep these terms distinct:

| Term | Responsibility | Location |
| --- | --- | --- |
| Method | Distinctive numerical behaviour/training | `margam/methods/` |
| Recipe | Components, connections, defaults and exposed settings for one method construction | `margam/recipes/` |
| Component | Reusable sampler, network, loss, operator or optimiser helper | Shared source modules; see capability index |
| Example | A physical problem wired to a method and explicit outputs/reference | `margam/templates/`; File → Examples |

## Design history and provenance

[Roadmap](roadmap.md) preserves the original discussions; its stage-specific
“not implemented” notes are historical. [Simulink evidence](simulink-evidence.md)
distinguishes sourced interactions from Margam choices. [Cleanup record](cleanup.md)
and [GeoSampler reuse review](geosampler-reuse.md) are dated records, not the current
capability catalogue. [Third-party attribution](../THIRD_PARTY.md) remains separate.

When adding or changing a method, update its guide, capability row, example,
validation evidence and results index together. Do not mark a proposed feature as
implemented or a regression threshold as a measured benchmark result.
