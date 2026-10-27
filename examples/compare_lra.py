"""CPU smoke comparison on interval Poisson, not a reproduction of paper results.

Run: .venv/bin/python -m examples.compare_lra --iterations 2000
"""
import argparse
import json
from dataclasses import replace, asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import platform
import subprocess
import numpy as np
import torch
from margam.templates.examples import example_config
from margam.templates.poisson import build_pinn


def compare(iterations,seeds):
    torch.set_num_threads(1)
    points=torch.linspace(0,1,201)[:,None];reference=points*(1-points)
    results=[]
    for seed in seeds:
        base=replace(example_config('lra_poisson'),network_seed=seed,iterations=iterations)
        for variant,method,cap in (('baseline','baseline',100.),('lra_cap100','lra',100.),('lra_cap1e6','lra',1e6)):
            model=build_pinn(replace(base,method=method,lra_max=cap));model.train(iterations,display_every=0)
            error=model.predict(points)-reference
            record=dict(seed=seed,variant=variant,method=method,coefficient_cap=cap,updates=iterations,rmse=error.square().mean().sqrt().item(),relative_l2=(error.norm()/reference.norm()).item(),max_abs_error=error.abs().max().item(),final_pre_update_loss=model.loss_history[-1])
            if method=='lra':record['loss_balancing']=model.balancing_summary();record['balance_events']=model.lra_history
            results.append(record)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations',type=int,default=2000)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,11,19])
    parser.add_argument('--output',type=Path,help='Save a provenance/settings/results JSON record to a new file')
    args=parser.parse_args()
    if not 1<=args.iterations<=100000:parser.error('iterations must be 1–100000')
    if args.output is not None and args.output.exists():parser.error('output must be a new file; preserve earlier results')
    records=compare(args.iterations,args.seeds)
    if args.output is not None:
        root=Path(__file__).resolve().parents[1]
        sources=['examples/compare_lra.py','margam/templates/examples.py','margam/templates/poisson.py',
                 'margam/methods/pinn.py','margam/methods/learning_rate_annealing.py','margam/components/losses.py',
                 'margam/components/sampling.py','margam/nn/FCNN.py','margam/nn/factory.py','margam/recipes/learning_rate_annealing.py','margam/backend/utils.py']
        try:
            revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
            dirty=subprocess.run(['git','diff','--quiet','HEAD'],cwd=root).returncode!=0
        except (OSError,subprocess.CalledProcessError):revision=None;dirty=None
        config=example_config('lra_poisson');config.iterations=args.iterations
        report=dict(schema_version=1,study='baseline_vs_lra_interval_poisson',
                    recorded_utc=datetime.now(timezone.utc).isoformat(),
                    environment=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
                                     platform=platform.platform(),device='cpu',dtype='float32',cpu_threads=torch.get_num_threads(),
                                     cuda_available=torch.cuda.is_available()),
                    provenance=dict(git_head=revision,tracked_changes=dirty,source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources}),
                    settings=dict(base_config=asdict(config),network_seeds=args.seeds,
                                  variants=['baseline','lra_cap100','lra_cap1e6'],evaluation_points=201,evaluation_range=[0.,1.],
                                  includes_boundary=True,reference='x*(1-x)',
                                  metric_reduction='RMSE and relative L2 over 201 scalar evaluation values',
                                  loss_measurement='last pre-update physical MSE terms and method-specific total; field errors after final update'),
                    records=records)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(records,indent=2))
