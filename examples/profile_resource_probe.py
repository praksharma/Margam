"""Measure disposable updates of the shipped study; no accuracy/fit claim."""
import argparse
import hashlib
import json
from pathlib import Path
from margam.execution.probe import probe_study
from examples.documentation import validate_public_text


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--devices',nargs='+',choices=('cpu','cuda'),default=['cpu']);parser.add_argument('--output',default='build/resource-probe.json');parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];source=root/'assets/studies/poisson-grid.margam-study';target=Path(args.output)
    if target.exists() and not args.overwrite:parser.error('Output exists; use --overwrite or choose a new path.')
    reports=[probe_study(source,updates=2,device=device) for device in args.devices]
    value=dict(generator='examples.profile_resource_probe',generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fixture='assets/studies/poisson-grid.margam-study',scope='Actual disposable native updates, not a fit prediction, performance benchmark or solution-accuracy claim. No checkpoint/output retained.',measurements=reports)
    text=json.dumps(value,indent=2,allow_nan=False)+'\n';validate_public_text(text,'Resource probe evidence');target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w' if args.overwrite else 'x') as stream:stream.write(text)
    print('Recorded '+str(len(reports))+' disposable probe measurements.')


if __name__=='__main__':main()
