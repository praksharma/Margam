"""Headless finite-grid study inspection/execution, without a GUI service."""
import argparse
import json
import sys


def main(argv=None):
    parser=argparse.ArgumentParser(prog='margam-cli',description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    study=commands.add_parser('study');actions=study.add_subparsers(dest='action',required=True)
    inspect=actions.add_parser('inspect',help='Read the portable definition; no execution.')
    inspect.add_argument('file')
    run=actions.add_parser('run',help='Run all design points fresh, serially, in isolated numerical processes.')
    run.add_argument('file');run.add_argument('--output',required=True,help='Fresh results directory; existing directories are refused.')
    run.add_argument('--device',choices=('auto','cpu','cuda'),help='Override saved device intent, recorded in results.')
    run.add_argument('--wall-time',type=float,help='Override saved per-job seconds (0 unlimited, maximum 604800).')
    run.add_argument('--export',action='store_true',help='Write files only for enabled, connected CSV/ParaView Export blocks.')
    resume=actions.add_parser('resume',help='Verify completed trials and restart unfinished trials fresh.')
    resume.add_argument('output',help='Existing version-2 study results directory.')
    resume.add_argument('--expected-execution-sha256',help='Reject execution metadata changed after an explicit launcher review.')
    resume.add_argument('--retry-failed',action='store_true',help='Also restart failed trials; invalid/incompatible trials remain recorded.')
    batch=actions.add_parser('slurm-export',help='Export a reviewable single-node Slurm bundle; no submission.')
    batch.add_argument('file');batch.add_argument('--output',required=True)
    batch.add_argument('--cpus',type=int,required=True);batch.add_argument('--memory-mb',type=int,required=True);batch.add_argument('--minutes',type=int,required=True)
    batch.add_argument('--gpu',action='store_true');batch.add_argument('--partition');batch.add_argument('--account');batch.add_argument('--export',action='store_true')
    probe=actions.add_parser('probe',help='Measure disposable complete native updates; no trained state or fit prediction.')
    probe.add_argument('file');probe.add_argument('--trial',type=int,default=1,help='One-based design point.')
    probe.add_argument('--updates',type=int,default=1,help='Complete updates/attempts within the original budget: up to 1000 for single-window or 100000 for Expert time marching.')
    probe.add_argument('--device',choices=('auto','cpu','cuda'));probe.add_argument('--wall-time',type=float,default=60,help='Isolated child limit in seconds; default 60, 0 unlimited.')
    for owned in (run,resume,probe):owned.add_argument('--watch-owner-stdin',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv);owner_watch=None
    try:
        if getattr(args,'watch_owner_stdin',False):
            from .execution.owner_watch import watch_owner_stdin
            owner_watch=watch_owner_stdin()
        from .execution.study_runner import inspect as inspect_study,run_study,resume_study
        if args.action=='probe':
            from .execution.probe import probe_study
            result=probe_study(args.file,trial=args.trial,updates=args.updates,device=args.device,wall_time_seconds=args.wall_time);code=0
        elif args.action=='slurm-export':
            from .slurm import export_job
            job=export_job(args.file,args.output,cpus=args.cpus,memory_mb=args.memory_mb,minutes=args.minutes,gpu=args.gpu,partition=args.partition,account=args.account,exports=args.export)
            result=dict(study_id=job['study_id'],resources=job['resources'],submitted=False);code=0
        elif args.action=='inspect':result=inspect_study(args.file);code=0
        else:
            result=resume_study(args.output,retry_failed=args.retry_failed,expected_execution_sha256=args.expected_execution_sha256) if args.action=='resume' else run_study(args.file,args.output,device=args.device,wall_time_seconds=args.wall_time,exports=args.export)
            code=130 if result['state']=='interrupted' else 0 if result['state']=='completed' else 1
            result={key:result[key] for key in ('study_id','state','counts')}
        if any(name=='PySide6' or name.startswith('PySide6.') for name in sys.modules):raise RuntimeError('Headless CLI unexpectedly imported Qt.')
        print(json.dumps(result,indent=2,allow_nan=False));return code
    except KeyboardInterrupt:
        print('Interrupted before the study could finish recording.',file=sys.stderr);return 130
    except (OSError,ValueError,TypeError,KeyError,RuntimeError,ImportError) as error:
        print('Study command failed: '+str(error),file=sys.stderr);return 2
    finally:
        if owner_watch is not None:owner_watch.set()


if __name__=='__main__':raise SystemExit(main())
