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
    resume.add_argument('--retry-failed',action='store_true',help='Also restart failed trials; invalid/incompatible trials remain recorded.')
    args=parser.parse_args(argv)
    try:
        from .execution.study_runner import inspect as inspect_study,run_study,resume_study
        if args.action=='inspect':result=inspect_study(args.file);code=0
        else:
            result=resume_study(args.output,retry_failed=args.retry_failed) if args.action=='resume' else run_study(args.file,args.output,device=args.device,wall_time_seconds=args.wall_time,exports=args.export)
            code=130 if result['state']=='interrupted' else 0 if result['state']=='completed' else 1
            result={key:result[key] for key in ('study_id','state','counts')}
        if any(name=='PySide6' or name.startswith('PySide6.') for name in sys.modules):raise RuntimeError('Headless CLI unexpectedly imported Qt.')
        print(json.dumps(result,indent=2,allow_nan=False));return code
    except KeyboardInterrupt:
        print('Interrupted before the study could finish recording.',file=sys.stderr);return 130
    except (OSError,ValueError,TypeError,KeyError,RuntimeError,ImportError) as error:
        print('Study command failed: '+str(error),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
