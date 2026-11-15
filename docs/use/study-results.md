# Collect a study's results in Margam

After running a [saved study from the CLI](study-cli.md), choose **File → Open
study results…**, then **Open results folder…**. Select the folder containing
`summary.json`, `execution.json` and `study.margam-study`. Copy the whole results
folder back from a remote machine before opening it; no remote connection is
required. Version-2 results are supported. Older run archives can still be opened
individually with **File → Open saved run…**.

![Study collection with completed and invalid design points](../../assets/screenshots/manual-study-results.png)

*Real CLI outcomes from a small Burgers network study: two valid trials and an
invalid network choice. Two updates test the workflow, not solution accuracy.
[Open full size](../../assets/screenshots/manual-study-results.png).*

## Review reported outcomes

Browsing preserves your current model, Undo history, trained weights, device
preference and files. It starts no training and loads no neural networks. You can
browse runs made with different recorded source versions or devices; those strict
runtime matches are required for **CLI resume**, rather than historical inspection.

Select a row to see its parameter choices, failure reason, previous attempts and
recorded runtime. **Refresh** explicitly rereads progress. There is no file watcher
or automatic retry. Reported `running` can be stale after a killed runner.

The reference-error selector chooses an output block's recorded discrete relative
L2 diagnostic. A dash means that trial has no completed diagnostic for the chosen
output. Undefined zero-reference errors retain their explicit status. Details
include the selected fields and row count. Margam does not rank trials or assume
that errors from different fields, units, coordinates or problems are comparable.

## Open one completed trial

Select a completed row and choose **Verify and open completed trial**. Margam
checks the frozen definition, execution record, published model/configuration and
all payload hashes, then makes a verified private copy of the native run archive.
Changed progress requires a refresh; changed, missing or unsafe files are refused.
Browsing alone reports metadata and does not verify all numerical payloads.

The normal saved-run workflow then asks about unsaved model changes and restores
that trial's owned inputs, weights and explicit outputs. The application's device
preference stays yours. Open individual block views as needed; no default plots
appear. You can continue the loaded checkpoint through the existing Resume action.

Reading and hashing run in the background. **Cancel reading** or closing the
browser stops the read without changing the model or result folder. Opening is
disabled during a training job or geometry edit and checked again before model
replacement. This browser is read-only: grouped execution, multi-run plotting and
Slurm submission/collection automation remain separate TODOs.
