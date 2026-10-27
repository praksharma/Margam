"""Measure the LRA-paper Helmholtz problem using Margam's existing Fourier NN.

This is a changed-architecture fixed-sample example, not a paper reproduction.
"""
import argparse,hashlib,json,platform,subprocess,time
from dataclasses import asdict,replace
from datetime import datetime,timezone
from pathlib import Path
import torch
from margam.templates.examples import example_config
from margam.templates.poisson import build_pinn


def measure(iterations,seeds):
    torch.set_num_threads(1)
    axis=torch.linspace(-1,1,101);points=torch.cartesian_prod(axis,axis)
    reference=torch.sin(torch.pi*points[:,0:1])*torch.sin(4*torch.pi*points[:,1:2])
    records=[]
    for seed in seeds:
        config=replace(example_config('helmholtz_lra_2d'),iterations=iterations,network_seed=seed)
        model=build_pinn(config);start=time.perf_counter();model.train(iterations,display_every=0)
        seconds=time.perf_counter()-start;error=model.predict(points)-reference
        residuals=[]
        for batch in points.split(512):
            batch=batch.clone().requires_grad_()
            residuals.append(model.problem.residual(batch,model.network(batch)).detach())
        residual=torch.cat(residuals)
        boundaries={condition.name:model.predict(x).abs().max().item() for condition,x,_ in model._constraints}
        records.append(dict(network_seed=seed,updates=iterations,training_seconds=seconds,
                            rmse=error.square().mean().sqrt().item(),relative_l2=(error.norm()/reference.norm()).item(),
                            max_abs_error=error.abs().max().item(),residual_rmse=residual.square().mean().sqrt().item(),
                            boundary_max_abs=boundaries,final_pre_update_loss=model.loss_history[-1],
                            balancing_summary=model.balancing_summary(),balance_events=model.lra_history))
        print(json.dumps({k:v for k,v in records[-1].items() if k!='balance_events'}),flush=True)
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations',type=int,default=5000)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,11,19])
    parser.add_argument('--output',type=Path,help='Write a new evidence JSON; existing files are preserved')
    args=parser.parse_args()
    if not 1<=args.iterations<=100000:parser.error('iterations must be 1–100000')
    if any(not 0<=s<2**31 for s in args.seeds):parser.error('use non-negative 32-bit seeds')
    if args.output is not None and args.output.exists():parser.error('output must be a new filename')
    records=measure(args.iterations,args.seeds)
    if args.output is not None:
        root=Path(__file__).resolve().parents[1];config=example_config('helmholtz_lra_2d');config.iterations=args.iterations;config.validate()
        sources=['examples/run_helmholtz.py','margam/templates/examples.py','margam/templates/predefined_pdes.py',
                 'margam/templates/poisson.py','margam/methods/pinn.py','margam/methods/learning_rate_annealing.py',
                 'margam/components/sampling.py','margam/components/expressions.py','margam/components/losses.py',
                 'margam/nn/factory.py','margam/nn/fourier.py','margam/nn/pointwise.py','margam/components/network_config.py']
        report=dict(schema_version=1,study='lra_paper_helmholtz_existing_fourier_network',recorded_utc=datetime.now(timezone.utc).isoformat(),
                    environment=dict(python=platform.python_version(),torch=str(torch.__version__),platform=platform.platform(),
                                     device='cpu',dtype='float32',cpu_threads=1,cuda_available=torch.cuda.is_available()),
                    provenance=dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                                    tracked_changes=subprocess.run(['git','diff','--quiet','HEAD'],cwd=root).returncode!=0,
                                    source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources}),
                    settings=dict(base_config=asdict(config),network_seeds=args.seeds,evaluation_resolution=101,evaluation_points=10201,
                                  evaluation_domain=[[-1.,-1.],[1.,1.]],includes_boundary=True,reference='sin(pi*x)*sin(4*pi*y)',
                                  metric_reduction='Unweighted RMSE, relative L2 and maximum error on evaluation grid; PDE residual RMS on same grid',
                                  limitations='Fixed samples and existing Fourier architecture; no paper architecture, minibatch resampling, rate-decay schedule, 40000-update or ten-trial reproduction; timings exclude construction and evaluation'),
                    records=records)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
