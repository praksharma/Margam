"""Fixed-data pointwise microbatch continuation and one-update allocator evidence.

Run full and chunked modes in separate fresh processes/folders. This is a small
research API/allocator experiment, not an accuracy or hardware-capacity claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from margam.research import Trainer,PINNProblem,PointConstraint,derivative,CosineDecay


def build(device,points,steps,size):
    torch.manual_seed(83)
    network=torch.nn.Sequential(torch.nn.Linear(1,32),torch.nn.Tanh(),torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,1)).double()
    locations=torch.linspace(0,1,points,dtype=torch.float64)[:,None]
    problem=PINNProblem(lambda x,u:derivative(u,x,0,2)+2,locations,
        [PointConstraint('boundary',torch.tensor([[0.],[1.]],dtype=torch.float64))])
    return Trainer(problem,network,weights={'boundary':10.},checkpoint_id='microbatch-poisson-example-v1').compile(
        device=device,lr=.001,schedule=CosineDecay(steps,.1),pointwise_microbatch=size)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda','auto'),default='auto')
    parser.add_argument('--points',type=int,default=4096)
    parser.add_argument('--steps',type=int,default=6)
    parser.add_argument('--microbatch',type=int,help='Deterministic pointwise rows per derivative graph; absent means full batch.')
    parser.add_argument('--output',type=Path,default=Path('build/research-microbatch'))
    args=parser.parse_args()
    if not 1<=args.points<=100000:parser.error('--points must be 1..100000')
    if not 2<=args.steps<=100000:parser.error('--steps must be 2..100000')
    if args.microbatch is not None and not 1<=args.microbatch<=100000:parser.error('--microbatch must be 1..100000')
    if args.output.exists():parser.error('Choose a fresh output folder; existing experiments are not overwritten.')
    torch.set_num_threads(1)
    model=build(args.device,args.points,args.steps,args.microbatch)
    memory=None
    if model.device.type=='cuda':
        torch.cuda.synchronize(model.device);torch.cuda.reset_peak_memory_stats(model.device)
        memory=dict(baseline_allocated_bytes=torch.cuda.memory_allocated(model.device),baseline_reserved_bytes=torch.cuda.memory_reserved(model.device))
    started=time.perf_counter();model.train(1,0)
    if memory is not None:
        torch.cuda.synchronize(model.device)
        memory.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(model.device),peak_reserved_bytes=torch.cuda.max_memory_reserved(model.device),
            after_allocated_bytes=torch.cuda.memory_allocated(model.device))
        memory['peak_above_baseline_bytes']=memory['peak_allocated_bytes']-memory['baseline_allocated_bytes']
    update_seconds=time.perf_counter()-started
    args.output.mkdir(parents=True)
    phase=max(1,args.steps//2);model.train(phase-1,0);model.save_checkpoint(args.output/'split-checkpoint.pt')
    model.train(args.steps-phase,0)
    resumed=build(args.device,args.points,args.steps,args.microbatch)
    resumed.load_checkpoint(args.output/'split-checkpoint.pt');resumed.train(args.steps-phase,0)
    exact=model.loss_history==resumed.loss_history and all(torch.equal(value,resumed.network.state_dict()[key]) for key,value in model.network.state_dict().items())
    if not exact:raise RuntimeError('Same-device microbatch continuation changed the trajectory.')
    resumed.save_checkpoint(args.output/'final-checkpoint.pt')
    report=dict(scope='Fixed-data deterministic research Poisson; one fresh complete update including Adam initialization; not accuracy/capacity evidence.',
        device=str(model.device),python=__import__('platform').python_version(),torch=str(torch.__version__),dtype=str(model.dtype),
        points=args.points,boundary_points=2,parameters=sum(p.numel() for p in model.network.parameters()),steps=model.iter,
        pointwise_microbatch=args.microbatch,checkpoint_version=resumed.training_state()['version'],exact_resume=exact,
        first_update_seconds=update_seconds,cuda_allocator=memory,last_preupdate_loss=resumed.loss_history[-1],
        measurement_scope='First complete optimizer update in this process; includes Adam moments, excludes model construction, checkpoint copies, predictions and later updates. CUDA figures cover the PyTorch allocator, not driver/system memory. CPU memory peaks are not measured.')
    root=Path(__file__).resolve().parents[1]
    names=('examples/research_microbatch.py','margam/research/trainer.py','margam/research/microbatch.py','margam/research/checkpoint.py','margam/research/schedules.py','margam/methods/pinn.py')
    report['source_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    (args.output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
