"""One-seed pointwise PINN compatibility study; not an architecture ranking."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import torch
import numpy as np
from margam.components.network_config import NETWORK_NAMES
from margam.templates.poisson import PoissonConfig,build_pinn


def smoke_options(kind):
    if kind in ('fourier','highway_fourier','modified_fourier'):return {'frequencies':[0.,.5,1.]}
    if kind=='fourier_mlp':return {'modes':3,'scale':.1}
    if kind=='multiscale_fourier':return {'scales':[.1,.5],'features':8}
    if kind=='siren':return {'first_omega':10.,'omega':10.}
    if kind in ('fourier_filter','gabor_filter'):return {'input_scale':.5}
    return {}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations',type=int,default=700)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if not 1<=args.iterations<=100000 or not 0<=args.seed<2**31:parser.error('Invalid update budget/seed.')
    if args.output.exists():parser.error('Output must be a new file; preserve earlier evidence.')
    torch.set_num_threads(1);points=torch.linspace(0,1,201)[:,None];reference=points*(1-points);records=[]
    for kind in NETWORK_NAMES:
        for method in ('baseline','self_adaptive'):
            config=PoissonConfig(shape='Interval',points=32,layers=(16,16),network_seed=args.seed,
                                 lr=.001 if kind=='siren' else .003,boundary_weight=10.,method=method,
                                 network_type=kind,network_options=smoke_options(kind),iterations=args.iterations)
            record=dict(network=kind,method=method,config=asdict(config))
            try:
                model=build_pinn(config);model.train(args.iterations,display_every=0)
                prediction=model.predict(points);error=prediction-reference
                record.update(status='completed',rmse=error.square().mean().sqrt().item(),relative_l2=(error.norm()/reference.norm()).item(),
                              max_abs_error=error.abs().max().item(),final_pre_update_loss=model.loss_history[-1],
                              predictions=prediction[:,0].tolist())
                if method=='self_adaptive':record['adaptive_weights']=model.adaptive_summary()
            except Exception as error:record.update(status='failed',error=f'{type(error).__name__}: {error}')
            records.append(record)
            print(kind,method,record['status'],record.get('rmse',record.get('error')),flush=True)
    root=Path(__file__).resolve().parents[1]
    sources=[Path(__file__).relative_to(root),Path('margam/components/network_config.py'),
             Path('margam/templates/poisson.py'),Path('margam/methods/pinn.py'),Path('margam/methods/self_adaptive.py'),
             *sorted(Path('margam/nn').glob('*.py'))]
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        dirty=subprocess.run(['git','diff','--quiet','HEAD'],cwd=root).returncode!=0
    except (OSError,subprocess.CalledProcessError):commit=None;dirty=None
    report=dict(schema_version=1,study='pointwise_network_compatibility_interval_poisson',recorded_utc=datetime.now(timezone.utc).isoformat(),
                environment=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,device='cpu',dtype='float32',cpu_threads=1),
                provenance=dict(git_head=commit,tracked_changes=dirty,source_sha256={str(p):hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources}),
                protocol='Different architectures/parameter counts; explicit low-frequency settings and lower SIREN rate; one seed. Compatibility smoke study, not equal-capacity ranking.',
                evaluation_points=points[:,0].tolist(),reference=reference[:,0].tolist(),records=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if any(r['status']=='failed' for r in records):raise SystemExit(1)


if __name__=='__main__':main()
