# When something does not look right

Start with **Validate** and the message shown by the application. A validation
message identifies a configuration issue; it is not a verdict on a PINN method.

## Training finished but I see no result

Check that an output block is connected to the trainer. Double-click it to open
its view. Startup has no output blocks, and complete examples do not open their
result windows automatically. Export blocks write a file only when you request it.

## Comparison reports a missing input

Comparison needs both the PINN run and a Reference connection. A Solution Plot
does not need a reference. Choose a reference that matches the problem and fields;
not every problem has an analytical solution.

## A connection will not attach

Check port types and whether the target input is occupied. Use one PDE producer.
An expanded trainer needs a network input; a compact recipe already owns one.
The current executor handles one problem root, not a study of multiple trainers.

## I cannot find a pane or all my blocks

Use **Fit model** to frame the graph. Show hidden panes from View, or use
**View → Reset panel layout**. This is different from resetting the Poisson model
under Examples. [Workspace controls](../start/workspace.md) lists navigation keys.

## The checkpoint will not resume

Check whether the architecture, settings or imported file contents changed.
Resume checks data fingerprints; reopening a file with the same name does not make
changed points identical. Start a new run for a changed problem. Archived output
may still be viewed without reconstructing missing external files.

## My imported points or VTI export fail validation

Preview the input mapping and check coordinate columns, dimensions, finite values,
units and any required outward unit normals. Interior imports use coordinates.
VTI export expects the complete regular **evaluation** grid; random **training**
points are supported. Arbitrary scattered-node export is future work.

## I expected a time animation

Current problems are steady snapshots. The writer can store supplied physical-time
frames in a PVD series; it does not manufacture a time axis from training updates.
Use the synthetic dataset in [Export](export.md) to check a reader independently.

## The manual does not open or looks old

Build it again with `.venv/bin/python -m examples.documentation build`, then press
F1. Screenshot captures are dated, real application states; old development images
remain in the repository but this user guide uses the current `manual-*` set.
See [documentation maintenance](../documentation.md) if you are updating the guide.

## I want to report a bug

Include the command/action, the exact message or traceback, the relevant model and
input mappings, and whether the problem repeats after reopening. Separate software
failures from a run that simply needs different scientific settings. The project
prioritises correct equations/data/state handling; solution tuning is your choice.
