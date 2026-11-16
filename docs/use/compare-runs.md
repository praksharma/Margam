# Compare saved predictions against a baseline

Choose **View → Compare saved predictions…**, then choose native
`.margam-run.pt` archives for **A** and **B**. These can be standalone saved runs
or completed trial archives collected through a [study results folder](study-results.md).
Choose the desired saved output from each dropdown and press **Compare selected outputs**.
Nothing is trained or loaded into your current model.

Use **Add run** for C through H, up to eight selections total. Choose an archive
and output for every added row. A remains the baseline: B−A, C−A and later
differences are separate comparisons. **Remove last run** releases that snapshot
and clears the old comparison. You may choose the same archive again as a control.
Slot/file/output changes require another comparison; slots cannot change during
a read or group comparison.

![Saved prediction differences and separate reference errors](../../assets/screenshots/manual-prediction-group.png)

*Actual saved Burgers predictions after two, three and five updates. B and C
are compared separately with A. The small budgets
exercise the workflow, not solution accuracy.
[Open full size](../../assets/screenshots/manual-prediction-group.png).*

## Matching is explicit

Every selected output must have exactly matching coordinate names/order, coordinate rows,
field names/order and point IDs, including ID types when supplied. No interpolation,
sorting, nearest-neighbor matching or automatic field mapping is performed. Physical
time is part of the coordinate rows for transient problems, so the same time samples
must be present too. Duplicate rows stay duplicated.

A mismatch in any run rejects the entire group; the diagnostic names that run
and no partial metric table is adopted. If matching fails, evaluate the models on the same explicit evaluation dataset
and save new runs. If an archive has no saved prediction outputs, choose another
archive or evaluate it first. Merely having trained weights is insufficient.

## Read the numbers

Each field of each non-baseline run has its own diagnostics over **all matching
rows**. For a group, the Run column identifies B through H; every difference
and relative denominator uses A:

- **RMS B−A** and **Max |B−A|** describe the difference between predictions.
- **Relative L2 B−A / A** uses prediction A as its denominator. This is a
  difference diagnostic, not error against an exact solution.
- **A/B vs own reference L2** recomputes each run's error against its own explicitly
  saved reference values. Missing references remain missing; zero-reference
  denominators retain an explicit undefined status.

Diagnostics use float64 discrete rows, without quadrature or unit normalization.
The table beneath previews the first 200 rows, including coordinates, original
point IDs and A, then each other run and its difference from A for every field. Preview truncation does not truncate metrics.
Overflow is reported explicitly rather than presented as a finite value.

The view states whether all saved references match A on these rows. Details also
report reference equality and changed configuration keys for each run versus A,
with every snapshot name and SHA-256. Matching rows and fields do **not**
establish matching units, PDEs, domains, reference definitions or accuracy protocols.
Inspect these before drawing scientific conclusions. Margam does not rank the runs,
combine fields with different units or infer which solution is better.

## Snapshots and safety

Archive reads and comparisons of three or more runs happen in a cancellable
background worker. **Cancel reading** also cancels group comparison. Cancellation
is checked between complete pair evaluations; a current CPU reduction finishes
before the group stops. Failed/cancelled comparisons leave no partial metric
table. The existing two-run comparison remains immediate after reading. Each read makes a bounded
private copy, checks the native container/data contract and uses restricted CPU
loading. It constructs no neural network, restores no optimizer and retains no
checkpoint/owned-file payload for this view. Full checkpoint restoration remains
part of **File → Open saved run…**, rather than comparison inspection.

Browsing/comparing preserves your current model, Undo, results, application device
and CPU/CUDA RNG. Files remain unchanged, and no referenced input file is reopened.
Snapshots stay fixed if the original archive later changes: choose it again to
refresh. Failed or cancelled reads clear the affected side and invalidate the old
comparison. Closing the dialog releases its snapshots and joins any reader.

This first comparison covers two saved prediction outputs. For retained loss curves use [Compare saved training histories](compare-histories.md).
Automatic study ranking, comparison-data export and further multi-run plotting remain
separate work. It adds no default plots or heatmaps.
