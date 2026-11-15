"""Export and launch a single-node serial Slurm study; never submit jobs."""
import argparse
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from . import study_document as sd
from .execution.study_state import read_json,safe_path,file_identity


TOKEN=re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z')


def resources(cpus,memory_mb,minutes,gpu,partition=None,account=None):
    for name,value,maximum in [('CPUs',cpus,1024),('Memory MiB',memory_mb,16777216),('Minutes',minutes,10080)]:
        if type(value) is not int or not 1<=value<=maximum:raise ValueError(f'{name} must be an integer from 1 to {maximum}.')
    if type(gpu) is not bool:raise ValueError('GPU request must be boolean (zero or one GPU).')
    for value in (partition,account):
        if value is not None and (not isinstance(value,str) or TOKEN.fullmatch(value) is None):raise ValueError('Partition/account must be a single 1–64 character letter/digit/dot/underscore/hyphen token.')
    return dict(cpus=cpus,memory_mb=memory_mb,minutes=minutes,gpu=gpu,partition=partition,account=account)


def script(settings):
    lines=['#!/bin/bash','#SBATCH --job-name=margam-study','#SBATCH --nodes=1','#SBATCH --ntasks=1',
        f"#SBATCH --cpus-per-task={settings['cpus']}",f"#SBATCH --mem={settings['memory_mb']}M",f"#SBATCH --time={settings['minutes']}",
        '#SBATCH --output=slurm-%j.out','#SBATCH --error=slurm-%j.err','#SBATCH --no-requeue']
    if settings['gpu']:lines.append('#SBATCH --gpus=1')
    for key in ('partition','account'):
        if settings[key] is not None:lines.append(f'#SBATCH --{key}={settings[key]}')
    lines.extend(['','set -euo pipefail','cd -- "${SLURM_SUBMIT_DIR:?Submit from the exported bundle directory}"',
        ': "${MARGAM_PYTHON:=python3}"',f"export OMP_NUM_THREADS={settings['cpus']}",f"export MKL_NUM_THREADS={settings['cpus']}",
        'exec "$MARGAM_PYTHON" -m margam.slurm run . "$@"',''])
    return '\n'.join(lines)


GUIDE='''# Run this exported Margam study

This bundle contains data and a reviewable single-node serial job script, not
Margam, Python, dependencies or credentials. Nothing has been submitted.

1. Copy this entire folder to cluster storage accessible to the compute node.
2. Activate an existing environment containing the desired Margam checkout and
   matching PyTorch/CUDA dependencies. Do not create an environment per trial.
   Set MARGAM_PYTHON to that environment's Python executable if it is not python3.
3. Change into THIS folder, inspect job.sbatch and job.json, then submit:

   sbatch job.sbatch

The script uses SLURM_SUBMIT_DIR, not its spool-file location. A GPU bundle
requests one GPU with --gpus=1 (Slurm select/cons_tres required); CPU bundles force
CPU execution. Partition/account and site GPU policies must suit your cluster.
Resource requests are explicit budgets, not measured fit/runtime predictions.
The batch allocation clock differs from the study's per-child wall-time limit.

Results go into the NEW results/ folder. Existing results are never overwritten.
To explicitly continue the same grid with unchanged code/runtime/settings:

   sbatch job.sbatch --resume
   sbatch job.sbatch --resume --retry-failed

Automatic requeue, job arrays, parallel trials, signal-triggered checkpoints and
mid-trial recovery are not implemented. A scheduler termination can leave running
metadata; resume verifies completed publications and restarts unfinished trials
FRESH. Do not delete the .study.lock file or bypass a live numerical child.
Use the original environment/source and resources when resubmitting. This
launcher does not verify that a scheduler actually granted the requested limits.

Copy the entire results/ folder back and choose File > Open study results in
Margam. Keep the frozen study, execution/summary records, attempt directories,
archives and any PVD frame folders together. CSV/VTK files are written only when
the bundle's export setting is enabled AND matching output blocks are connected.
No reference solution is invented. Short workflow checks do not establish accuracy.

The launcher verifies the frozen study/script manifest. Re-export to change its
settings rather than editing files in place. No packages are installed or commands
submitted by export. Official option reference: https://slurm.schedmd.com/sbatch.html
'''


def export_job(study_path,output,*,cpus,memory_mb,minutes,gpu=False,partition=None,account=None,exports=False):
    settings=resources(cpus,memory_mb,minutes,gpu,partition,account)
    if type(exports) is not bool:raise ValueError('Exports must be boolean.')
    output=Path(output).absolute()
    if output.exists() or output.is_symlink():raise FileExistsError('Choose a fresh Slurm bundle directory.')
    from .execution.study_runner import snapshot
    with TemporaryDirectory(prefix='margam-slurm-source-') as temporary:
        frozen=Path(temporary)/'study.margam-study';snapshot(study_path,frozen);study,_=sd.read(frozen)
        content=frozen.read_bytes();job=script(settings).encode()
        manifest=dict(format='margam-slurm-job',version=1,study_id=study['study_id'],resources=settings,
            execution=dict(device='cuda' if gpu else 'cpu',exports=exports),
            files={name:dict(size=len(data),sha256=sd.digest(data)) for name,data in [('study.margam-study',content),('job.sbatch',job)]})
        payloads={'study.margam-study':content,'job.sbatch':job,'job.json':sd.canonical(manifest)+b'\n','README.md':GUIDE.encode()}
        output.mkdir(mode=0o700,parents=True,exist_ok=False);created=[]
        try:
            for name,data in payloads.items():
                with (output/name).open('xb') as stream:created.append(output/name);stream.write(data)
        except BaseException:
            for path in created:path.unlink(missing_ok=True)
            try:output.rmdir()
            except OSError:pass
            raise
    return manifest


def read_job(root):
    root=Path(root).absolute();job=read_json(root,'job.json')
    if not isinstance(job,dict) or set(job)!={'format','version','study_id','resources','execution','files'} or job['format']!='margam-slurm-job' or type(job['version']) is not int or job['version']!=1:raise ValueError('Unsupported Slurm bundle.')
    settings=job['resources']
    if not isinstance(settings,dict) or set(settings)!={'cpus','memory_mb','minutes','gpu','partition','account'}:raise ValueError('Invalid Slurm resources.')
    resources(**settings)
    if not isinstance(job['execution'],dict):raise ValueError('Invalid Slurm execution settings.')
    if job['execution']!={'device':'cuda' if settings['gpu'] else 'cpu','exports':job['execution'].get('exports')} or type(job['execution']['exports']) is not bool:raise ValueError('Invalid Slurm execution settings.')
    if set(job['files'])!={'study.margam-study','job.sbatch'}:raise ValueError('Invalid Slurm file manifest.')
    for name,identity in job['files'].items():
        if file_identity(root,name)!=identity:raise ValueError('Slurm bundle checksum changed: '+name)
    if safe_path(root,'job.sbatch').read_text()!=script(settings):raise ValueError('Slurm script differs from recorded resource settings.')
    study,_=sd.read(safe_path(root,'study.margam-study'))
    if study['study_id']!=job['study_id']:raise ValueError('Slurm study identity changed.')
    return job


def run_job(root,*,resume=False,retry_failed=False):
    if type(resume) is not bool or type(retry_failed) is not bool or retry_failed and not resume:raise ValueError('Failed retries require explicit resume.')
    root=Path(root).absolute();job=read_job(root);output=root/'results'
    from .execution.study_runner import run_study,resume_study,snapshot
    if resume:
        report=read_json(output,'summary.json')
        if report['study_id']!=job['study_id'] or report['study_archive_sha256']!=job['files']['study.margam-study']['sha256'] or report['effective_execution']['requested_device']!=job['execution']['device'] or report['effective_execution']['exports']!=job['execution']['exports']:raise ValueError('Existing results differ from this Slurm bundle.')
        return resume_study(output,retry_failed=retry_failed)
    with TemporaryDirectory(prefix='margam-slurm-run-') as directory:
        frozen=Path(directory)/'study.margam-study'
        if snapshot(root/'study.margam-study',frozen)!=job['files']['study.margam-study']['sha256']:raise ValueError('Slurm study changed before launch.')
        return run_study(frozen,output,device=job['execution']['device'],exports=job['execution']['exports'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('run',));parser.add_argument('bundle');parser.add_argument('--resume',action='store_true');parser.add_argument('--retry-failed',action='store_true');args=parser.parse_args(argv)
    try:
        report=run_job(args.bundle,resume=args.resume,retry_failed=args.retry_failed)
        print(json.dumps({key:report[key] for key in ('study_id','state','counts')},indent=2));return 130 if report['state']=='interrupted' else 0 if report['state']=='completed' else 1
    except KeyboardInterrupt:return 130
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,ImportError) as error:
        print('Slurm bundle failed: '+str(error),file=__import__('sys').stderr);return 2


if __name__=='__main__':raise SystemExit(main())
