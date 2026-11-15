# Export a study for Slurm

Save a [portable study](parameter-studies.md) in Margam, then export a single-node
serial job bundle from the repository environment:

```bash
.venv/bin/margam-cli study slurm-export my-study.margam-study \
  --output jobs/my-study --cpus 4 --memory-mb 8192 --minutes 60
```

All three resource budgets are required. Optional `--partition compute` and
`--account research` record cluster-specific names. Export needs a **new folder**;
existing folders are refused. It does not install software, contact a cluster or
submit a job, and does not start training.

For one GPU and explicit output-block exports:

```bash
.venv/bin/margam-cli study slurm-export my-study.margam-study \
  --output jobs/my-gpu-study --cpus 4 --memory-mb 16384 --minutes 120 \
  --gpu --partition gpu --export
```

Without `--gpu`, execution explicitly uses CPU even if the saved study requested
auto/CUDA. With it, the script requests one GPU and execution explicitly uses CUDA;
unavailable CUDA remains incompatible, with no CPU fallback. `--export` writes
CSV/ParaView files only for enabled, connected export blocks. Native archives are
always retained for completed trials.

## Review and transfer

The bundle contains:

```text
my-study/
  study.margam-study    # frozen model, parameter grid and owned inputs
  job.json             # requested resources, execution choice and file checksums
  job.sbatch           # reviewable batch script
  README.md            # standalone launch/resume/collection instructions
```

Copy the whole folder to cluster storage visible to its compute node. Prepare an
existing environment containing the desired Margam version and its dependencies;
the bundle does not contain code or an environment. Activate that environment,
or point `MARGAM_PYTHON` to its Python executable. A CPU-only Torch installation
cannot execute a GPU bundle.

```bash
cd my-study
export MARGAM_PYTHON=python
sbatch job.sbatch
```

Submit **from inside the bundle directory**. The script resolves its files through
`SLURM_SUBMIT_DIR`, rather than the script's scheduler spool location. That variable
identifies the directory where submission occurred. [Slurm sbatch reference](https://slurm.schedmd.com/sbatch.html)

The generated script requests one node/task, the selected CPUs, node memory in
MiB (`--mem=…M`) and a batch time in minutes. GPU jobs use `--gpus=1`, which requires
appropriate cluster support, including `select/cons_tres`. Check partition/account,
GPU policy and any additional site requirements with your cluster. CPU counts set
OMP/MKL thread environment values; they are not Margam memory/CPU isolation quotas.
These directives and GPU-selection requirements are documented by
[Slurm sbatch](https://slurm.schedmd.com/sbatch.html) and its
[GPU scheduling guide](https://slurm.schedmd.com/gres.html).

Budgets are user requests, not measured fit or runtime predictions. The Slurm
allocation clock includes the entire batch; the saved study's per-child wall-time
limit covers each numerical child separately. The launcher does not verify the
scheduler's actual allocation or record every site override. Use scheduler logs
for that evidence. Re-export to change bundle settings; the launcher checks the
frozen study/script checksums and their agreement with `job.json`.

## Continue and collect

A new execution writes **`results/`** beside the bundle files. Repeating the fresh
command refuses existing results. To explicitly resume that same grid:

```bash
sbatch job.sbatch --resume
sbatch job.sbatch --resume --retry-failed
```

[Grid resume](study-cli.md#continue-the-grid) requires unchanged recorded source,
runtime/device and execution settings. It verifies completed publications, preserves
old attempts and restarts interrupted trials **fresh**. Failed retries are explicit;
invalid/incompatible trials remain recorded. Keep the same environment and requested
resources for a resumed submission.

Automatic requeue is disabled in the generated directives. Slurm administrators
can override that policy; the launcher still refuses fresh overwrites.
[Slurm requeue options](https://slurm.schedmd.com/sbatch.html)
There is no signal-triggered checkpoint, mid-trial optimizer recovery, array
scheduler or parallel execution. A scheduler kill can leave stale `running`
metadata; retry only after the original job/child has exited. Never remove a lock
file to bypass a live runner.

Copy the **entire `results/` folder** back, then choose **File → Open study results…**
to [inspect reported outcomes and verify/open a completed trial](study-results.md).
Keep archives, attempt folders, frozen provenance and PVD frame directories together.
No references or accuracy metrics are invented for failed or reference-free trials.

## Validation scope

The generated scripts are syntax-checked and exercised with real CPU and CUDA
numerical jobs through local Bash, including an emulated submission directory and
a different script/CWD location. Owned reference inputs, CSV IDs, physical-time
PVD frames and native GUI reopening are checked. **A real Slurm cluster has not
been used for this qualification.** Site submission, scheduling, resource enforcement
and preemption remain unqualified; small training budgets test software behavior,
not solution accuracy.
