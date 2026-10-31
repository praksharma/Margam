# Save, reopen and resume

A model definition and a trained run are different files. Save the one you need;
use both if you want an editable setup and a recorded training snapshot.

| File | Contains | Command |
| --- | --- | --- |
| `.margam.json` | Graph, block IDs, settings, connections and positions | File → Save model / Save model as |
| `.margam-run.pt` | Model, requested results, monitor records and training checkpoint | File → Save run and checkpoint |
| `.csv` / `.vti` | Explicitly exported predictions | Connected export block's action |

![After training, File provides separate model and run/checkpoint actions](../../assets/screenshots/manual-save-menu.png){ .portrait }

*After training, File provides separate model and run/checkpoint actions. [Open full size](../../assets/screenshots/manual-save-menu.png).*

## Reopen a setup

Choose **File → Open model…**. It restores the graph and settings. It does not
load trained weights or recreate old predictions. Loading another setup uses the
normal Save / Discard / Cancel check for your current unsaved work.

## Continue a trained run

Choose **File → Open saved run…**. Result views stay closed until requested.
Use **Run → Resume checkpoint…** to add more training updates. The archive carries
the state required by its method, including SA masks or LRA balancing history.

Resume validates architecture, optimiser settings and input-data identity.
Changing imported files, the architecture or numerical settings is not silently
accepted as the same run. Copied inputs are included in new run archives; linked
CSV/NPZ files must remain available for actual resume. Archived predictions can
still be viewed if external input files moved.

## What undo and layout remember

Undo/redo changes model definitions; it does not certify old results after a
physics change. Numerical edits clear stale results/checkpoints. Moving blocks or
renaming outputs preserves the trained result. Desktop layout/recovery preferences
are stored separately from your model; no result view opens from layout recovery.

For exact formats and validation rules, see [desktop persistence](../ui-prototype.md)
and [validation scope](../validation.md).

**Next:** [Export predictions](export.md).

Adaptive sampler runs also retain active points, importance weights and the sampling
schedule. Resume verifies the original data and settings; it does not restart the
sampler. See [adaptive sampling and resume](../adaptive-sampling.md#saved-models-and-runs).

## Project-owned files

Unsaved models have a private temporary working folder, created only when a
component needs files. First Save copies these files beside the model into
`<name>_files`; Save As copies them for the new model and preserves the original.
Keep the model and its companion folder together when moving a project.
Ordinary models without owned files do not create an empty companion folder.
A failed save retains the temporary working files so saving can be retried.
Discarding or closing the model removes its temporary working folder.

[Jorvik geometry documents](../jorvik.md) and their NetCDF snapshots use this owned
folder. Their paths are stored relative to the project workspace. CSV/NPZ files
can also be copied there when importing; existing links remain external unless
you explicitly change File handling. Desktop recovery copies owned files alongside
the recovery model; it pauses during external editing and resumes when the editor
closes. Close Jorvik before saving, switching projects or closing Margam.


## Choose copied or linked input data

Every CSV/NPZ mapping dialog now has **File handling**:

| Choice | Behaviour |
| --- | --- |
| **Copy into project** | A fixed snapshot travels with the model's companion folder, Save As, recovery and run archives. Changes to the original file are not applied automatically. |
| **Link existing file** | The project reads the file at its external location. Keep that file available when moving the project or resuming training. |

New sources default to **Copy into project**. Existing linked sources keep
**Link existing file** when reopened, so upgrading or previewing an old model
does not silently copy or replace its inputs. Choose Copy explicitly to convert
one. Preview the mapping and click **OK** to apply it. Cancel and failed
validation leave the mapping unchanged and do not install a new dataset.

![A mapped project copy with its preserved IDs and rows](../../assets/screenshots/manual-data-copy.png)

*The file field identifies a project copy by name; no temporary folder path needs
to be managed by the user. The mapping and data readers are unchanged.*

![An explicit external link](../../assets/screenshots/manual-data-link.png)

The choice applies independently to interior points, each named boundary set,
evaluation points, Reference data and the older Reference block's imported values.
CSV and NPZ contents are copied intact; coordinate/normal/value/ID mappings are
preserved. The inspector shows **Project copy** or **Linked file**, with the linked
location shown explicitly. To replace a copied snapshot with newer data, browse
the desired original file and apply Copy again. To create a link from a copied
source, choose an external file and select Link; Margam will not create a fragile
absolute link into its own temporary folder.

![A moved project using managed inputs throughout the graph](../../assets/screenshots/manual-data-project.png)

Save a copied-data model and move **both** `<name>.margam.json` and `<name>_files`.
Save As copies the owned data for the new model and retains the originals. It
preserves trained weights for evaluation without extra optimizer updates.
Opening a run archive restores copied inputs into a private workspace, allowing
resume without the original model folder. Linked inputs remain external and are
not silently bundled. Existing model/archive formats remain readable.

Each imported file is limited to 64 MiB, including the existing expanded-NPZ
checks. Run archives retain their 256 MiB/4096-file limits. Copies retained for
undo/redo are kept in the workspace; automatic pruning is not implemented.
Copy/paste of blocks containing owned files remains disabled. CLI configuration
continues to use explicit ordinary file paths; this copy/link choice belongs to
the desktop project workflow.

### Reproduce the portability check

```sh
.venv/bin/python -m examples.portable_data --device cpu
.venv/bin/python -m examples.portable_data --device cuda
```

Each command leaves a fresh `run-*` folder below `build/portable-data` and prints
its location. Open `portable.margam.json` with its companion folder to inspect the
finished model. The check copies six training/evaluation/reference sources,
deletes the originals, verifies recovery, saves twice without changing weights,
relocates and trains the model, then restores/resumes a run archive after removing
the relocated model's inputs. Continuation matches uninterrupted training exactly
on the same device. It also checks reference alignment, original point IDs/order,
and CSV/VTU export. See the [CPU record](../../assets/benchmarks/portable-data-cpu.json)
and [CUDA record](../../assets/benchmarks/portable-data-cuda.json). This is a
portability check, not a solution-accuracy benchmark.
