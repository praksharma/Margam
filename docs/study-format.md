# Portable finite-grid study format

Version 1 `.margam-study` files are data-only ZIP bundles. This is the serialization
and deterministic trial-resolution milestone; there is no study executor yet.
See [the illustrated workflow](use/parameter-studies.md).

## Contents and identity

`study.json` has exactly these fields:

| Field | Meaning |
| --- | --- |
| `format`, `version` | `margam-study`, integer `1` |
| `model` | Current versioned Margam graph, including explicit output blocks |
| `axes` | Ordered parameter rows: block ID, parameter, mode, original JSON text |
| `execution` | Device intent `auto`/`cpu`/`cuda`, wall-time seconds 0–604800 |
| `files` | Relative owned filenames mapped to byte length and SHA-256 |
| `study_id` | SHA-256 of canonical JSON for all preceding fields |

Each input is stored at `files/<relative-owned-name>`. All active input references
must be owned and covered by the file map; only declared files and optional saved
Jorvik document/NetCDF siblings are allowed. Existing Jorvik files are carried
through the current contract; this does not expand geometry support. Linked
CSV/NPZ/NetCDF files require explicit inclusion before capture and become
`margam-owned:study_inputs/<content-sha256>.<extension>` in the saved copy. Original
model mappings and historical accepted-import fingerprint records are not rebound.
Labels, notes and fingerprint history are never interpreted as active file paths.

The cached model configuration is assembled from actual block parameters, with a
neutral cached CPU device. Execution intent is authoritative for the future
runner; the inspector cache and local app preference do not silently define it.
The model graph retains its real block IDs/settings, including seeds, network
options, conditions, enabled flags, reference mappings and requested resolutions.
Outputs are declarations, not predicted results.

Canonical JSON sorts keys, uses compact separators and ASCII escapes, rejects
non-finite values and normalizes tuples to JSON arrays. ZIP timestamps/compression
metadata are outside study identity. Changing any manifest content or input bytes
changes the identity. Equivalent-looking edits are not semantically deduplicated.
Checksums detect corruption, not authorship or a trusted numerical solution.

## Deterministic expansion without execution

```python
from margam.study_document import read, trial_model, restore
from margam.studies import config_from_document
from types import SimpleNamespace

study, files = read("study.margam-study")
trial_id, model = trial_model(study, 0)  # zero-based index
workspace = restore(files)
try:
    context = SimpleNamespace(project_workspace=lambda: workspace,
                              execution_device=lambda: "cpu")
    config = config_from_document(model, context)
    config.validate()
finally:
    workspace.close()
```

`read` and `trial_model` require neither Qt nor Torch and do not import executable
model payloads. Numerical configuration validation may import Torch for existing
physics checks, such as coordinate-dependent corner targets. It still requires
no Qt/display or GUI service. This example does not train, predict or export.
A future executor must resolve device intent and apply the normal full execution,
data/output and checkpoint contracts rather than trusting preflight alone.

Expansion reuses `margam.studies.resolve_axes`: 1–8 actual block-held controls,
1–64 values each, at most 512 Cartesian trials, last axis varying fastest.
`trial_model` preserves the base model and returns a fresh graph with that trial's
choices. The stable ID is `DP0001-<full-study-sha256>` for index zero. Invalid
numerical combinations remain part of the grid and need explicit invalid/skipped
reporting by the future executor; no objective values are fabricated.

## IO and restoration limits

Readers reject duplicate JSON keys, unsupported versions, altered identities,
invalid axis references/counts, active external links, malformed file maps,
traversal/ambiguous/conflicting paths, undeclared or missing ZIP entries,
nonregular/symlink/encrypted entries, unsupported compression and input byte/hash
mismatches. Only stored and deflate ZIP entries are accepted. Expanded inputs are
bounded to 64 MiB each, 256 MiB total and 256 files; the manifest is at most 8 MiB.
Archive-size allowance includes bounded ZIP overhead. There is no `extractall`.

`restore` writes validated relative inputs to a fresh private project workspace;
it never extracts over the original project. Save validates first, writes a
private temporary archive and publishes atomically in the destination directory.
Default publication refuses existing files, including publication races; explicit
replacement uses atomic replace. No temporary file is retained after failure.
Source symlinks and changing files are rejected by the shared descriptor-based
reader. Secure copying requires no-follow/descriptor filesystem operations;
unsupported platforms report unavailable instead of silently following links. No arbitrary neighboring workspace files are copied.

This bundle does not include software environments, checkpoints, training logs,
results, credentials or arbitrary code modules. Historical fingerprint records
may retain original source paths; inspect private models before publishing them.
