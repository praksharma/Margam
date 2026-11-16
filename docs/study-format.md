# Portable finite-grid study format

Version 1/2/3 `.margam-study` files are data-only ZIP bundles. This is the serialization
and deterministic trial-resolution contract; the [serial CLI runner](use/study-cli.md)
now consumes it through the shared native execution gate.
See [the illustrated workflow](use/parameter-studies.md).

## Contents and identity

`study.json` has these exact base fields; version 2 additionally requires
`network_defaults`, described below. Version 3 requires both `network_defaults`
and `combination`:

| Field | Meaning |
| --- | --- |
| `format`, `version` | `margam-study`, integer `1`, `2` or `3` |
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

## Frozen network alternatives (version 2)

A `network_type` axis uses Fixed/Candidates string keys from the pointwise network
catalogue, never numeric ranges. Version 2 freezes `network_defaults` as a map of
exactly the selected families to complete, schema-validated option dictionaries.
It is included in study identity. Materialization applies these literal options,
including when a candidate equals the base family; base custom options are not
implicitly carried across. Preview and execution share this application rule.

Cartesian numeric-only studies remain version 1 and retain their identities. Version 1 cannot
contain architecture axes; version 2 requires them. Unknown/missing option fields,
undeclared families and malformed maps are rejected. Changed library default values
do not replace saved values; future unsupported option schemas must fail explicitly
rather than silently changing an experiment. Model/input/software compatibility
and runtime provenance remain necessary for reproduction.

See [the architecture workflow](use/study-networks.md). Configured PINN recipe or
arbitrary graph alternatives are not encoded by this field.

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
The CLI resolves device intent and applies the normal full execution/data/output
contracts rather than trusting preview alone. It records effective overrides and
runtime provenance separately from the saved definition identity.

Expansion reuses `margam.studies.resolve_axes`: 1–8 actual block-held controls,
1–64 values each, at most 512 Cartesian trials, last axis varying fastest.
`trial_model` preserves the base model and returns a fresh graph with that trial's
choices. The stable ID is `DP0001-<full-study-sha256>` for index zero. Invalid
numerical combinations remain part of the grid and need explicit invalid/skipped
reporting by the executor; no objective values are fabricated.

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


## Results and grid continuation

Study bundles use version 1 for Cartesian numeric-only grids, version 2 for
Cartesian frozen network-family alternatives, and version 3 for paired lists. CLI result folders independently use version-2
`summary.json` records and a separate frozen `execution.json`. Per-trial attempts
have independent directories and an atomically published `artifacts/publication.json`
manifest binding trial identity, configuration, code and payload checksums.
The [CLI guide](use/study-cli.md#continue-the-grid) describes POSIX grid resume,
explicit failed retries, prior-attempt retention and process leases. This restarts
unfinished trials fresh; it does not continue their partial optimizer state.


## Paired studies (version 3)

Version 3 adds `combination: "paired"` and requires `network_defaults` (empty
when no network-family axis exists). Only this exact mode is accepted in version
3; Cartesian studies continue to use version 1 or 2. All these fields participate
in the content hash. Missing/unknown fields or modes are rejected.

Resolve each axis using the existing Fixed/Candidates/Range rules. All lists
longer than one must have the same length. Trial i takes element i from each such
list and repeats each singleton value. The total is the longest list, at most 64,
without computing or imposing the Cartesian product limit on paired grids.
The shared combination resolver feeds preview and native trial materialization;
runner, resume and collection use the same total and retain invalid row indices.
Repeated axis values and complete rows are explicit separate paired trials;
Cartesian duplicate-value rejection is unchanged. No conditional skipping or
topology changes are implied. Frozen family defaults
follow the version-2 policy. Version-1/2 serialization and identities are unchanged.
