# Portable study examples

[poisson-grid.margam-study](poisson-grid.margam-study) is a native baseline Poisson
study created with the actual study dialog. It has 12 design points: update
budgets 1000/2000/3000, Adam rates 0.001/0.003 and layers [16,16]/[32,32].
It requests 24,000 updates in total; its largest network has 1,185 parameters.
No reference data, trained weights, results or paper-accuracy claim is included.

Open it using Run → Parameter study… → Open study…. Opening replaces the current
model and table after normal discard prompts. Preview/save/open works; the [headless CLI](../../docs/use/study-cli.md) executes
the finite grid serially. HPC scheduling and grid resume remain future work.

The [capture manifest](../screenshots/manual-study-preview.json) records the
file hash and study identity. ZIP timestamps may differ on regeneration; study
identity describes the canonical definition and input hashes. The generated
native block IDs are part of that definition.

See [the user guide](../../docs/use/parameter-studies.md) and
[format contract](../../docs/study-format.md).
