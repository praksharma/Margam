"""Opt-in cold-child Expert window measurements, not a fit/speed/accuracy benchmark."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from margam.templates.examples import example_config
from margam.templates.poisson import BoundarySampling
from margam.execution.probe import validate_probe
from margam.execution.probe_report import validate_window_report
from margam.execution.codec import encode_config
from margam.execution.transport import JobTransport
from margam.execution.study_runner import source_identity,source_matches
from examples.documentation import validate_public_text


def case_config(case,device,windows=3,steps=16,mode='ntk'):
    c=example_config(case);c.device=device;c.points=32;c.boundary_points=4;c.layers=(8,8)
    c.network_options['features']=4
    c.boundary_sampling={side:BoundarySampling(4,'random',c.seed+i+1) for i,side in enumerate(c.sides)}
    c.expert_settings.update(windows=windows,window_steps=steps,chunks=4,balance_mode=mode,balance_every=4,ntk_points=2)
    c.iterations=windows*steps;c.validate();return c


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs='+',choices=('expert_burgers','expert_cylinder'),default=['expert_burgers','expert_cylinder'])
    p.add_argument('--devices',nargs='+',choices=('cpu','cuda'),default=['cpu'])
    p.add_argument('--windows',type=int,default=3);p.add_argument('--steps',type=int,default=16)
    p.add_argument('--updates',type=int,help='Default: complete configured schedule; never fast-forward a handoff.')
    p.add_argument('--mode',choices=('gradient','ntk'),default='ntk');p.add_argument('--wall-time',type=float,default=120)
    p.add_argument('--output',type=Path,default=Path('build/window-resource-probes.json'));p.add_argument('--overwrite',action='store_true');a=p.parse_args()
    if not 2<=a.windows<=32 or not 1<=a.steps<=100000 or a.windows*a.steps>100000:p.error('Choose 2–32 windows and a positive per-window schedule totaling at most 100000 updates.')
    if a.output.exists() and not a.overwrite:p.error('Choose a fresh output or use --overwrite.')
    source=source_identity();records=[]
    for case in a.cases:
        for device in a.devices:
            if not source_matches(source):raise RuntimeError('Source changed; audit not published.')
            config=case_config(case,device,a.windows,a.steps,a.mode);updates=config.iterations if a.updates is None else a.updates;validate_probe(config,updates)
            row=dict(case=case,device=device,configuration=asdict(config),requested_updates=updates)
            try:
                packet=JobTransport(wall_time_seconds=a.wall_time).run(dict(operation='probe',config=encode_config(config),requests=[],checkpoint=None,metadata={},dtype='float32',probe_updates=updates))
                if packet['checkpoint'] is not None or packet['metadata'] or not packet['headless']:raise ValueError('Unexpected retained native probe state.')
                validate_window_report(packet['result']);row.update(status='completed',measurement=packet['result'])
            except Exception as error:row.update(status='failed',reason=type(error).__name__+': '+str(error),measurement=None)
            if not source_matches(source):raise RuntimeError('Source changed during numerical work; audit not published.')
            records.append(row)
    result=dict(format='margam-window-resource-audit',version=1,generator='examples.profile_window_resources',generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source=source,
        scope='Cold isolated native Expert probes including reached teacher transfers. Fixed small case settings; cylinder is the self-contained startup variation. No numerical accuracy, full-run capacity or paper/hardware speed claim. Synchronization can alter overlap and allocator reuse.',records=records)
    content=json.dumps(result,indent=2,allow_nan=False)+'\n';validate_public_text(content,'Window resource audit');a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w' if a.overwrite else 'x') as stream:stream.write(content)
    failed=sum(row['status']=='failed' for row in records);print(f'Recorded {len(records)-failed} complete probes and {failed} failures.');return 1 if failed else 0


if __name__=='__main__':raise SystemExit(main())
