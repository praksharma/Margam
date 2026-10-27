# Training and optional results

Click **Validate** before **Train PINN** (F5). Validation checks the supported
problem graph and settings. It does not establish that the trained solution will
reach a particular accuracy.

During training, progress and operational messages remain available. Stop requests
are checked between batches of 25 updates. A stopped partial run can retain
results and a checkpoint; changing the numerical model then invalidates them.

## Ask for the views you need

| Block | What it requests | How to open |
| --- | --- | --- |
| Training Monitor | Recorded loss history | Double-click or Open monitor |
| Solution Plot | Predicted field values on an evaluation grid | Double-click or Open plot |
| Comparison | Predictions and an explicitly supplied reference | Connect both inputs, then Open comparison |
| CSV Export / ParaView Export | Evaluation data for an external file | Explicit export action after training |

The startup graph has no outputs. Complete examples wire useful optional outputs,
but their windows remain closed until opened. You can keep training without a
figure on screen. Closing a view is different from deleting its block.

![The block-owned monitor appears as a dockable results tab](../../assets/screenshots/manual-monitor.png)

*The block-owned monitor appears as a dockable results tab. [Open full size](../../assets/screenshots/manual-monitor.png).*

## Read the losses in context

Per-term PDE/constraint curves describe physical errors at training locations.
The total includes the configured method's weights; SA and LRA change their
objectives during training. A smaller total across two different methods need
not mean a better field solution. A reference is not required to see losses.

## Predictions and comparisons

Prediction resolution controls an independent regular evaluation grid. It does
not change the training samples. For scalar examples the existing plot views are
available. Coupled `u,v,p` results use tables and multi-column exports. A preview
table can show only the first rows while export includes the requested grid.

![An explicit scalar comparison tab; the reference is a separate graph input](../../assets/screenshots/manual-comparison.png)

*An explicit scalar comparison tab; the reference is a separate graph input. [Open full size](../../assets/screenshots/manual-comparison.png).*

References can be supported presets, restricted expressions or mapped data, under
the documented contracts. An imported reference must match evaluation coordinates;
it is not silently interpolated from an arbitrary FEM mesh. The heat rectangle
reference is a labelled truncated series, not a universal exact solution.

**Next:** [Save or resume](saving.md), or [export predictions](export.md).
