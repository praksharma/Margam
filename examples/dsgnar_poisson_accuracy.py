"""Train Margam's 1D Poisson DSGNAR example against its analytical solution."""
import argparse,json,time
from pathlib import Path
import torch
from margam.templates.examples import example_config
from margam.templates.poisson import build_pinn


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--checkpoint',type=Path);args=parser.parse_args()
    torch.set_num_threads(1);c=example_config('dsgnar_poisson_1d');c.device='cpu';c.network_seed=c.seed;c.points=256;c.layers=(32,32);c.pde_weight=1.;c.boundary_weight=1.
    c.dsgnar_settings.update(sketch_size=700,batch_size=64,reweight_exponent=0.,min_radius=1e-12,precision='float64',probes=12)
    model=build_pinn(c);points=torch.linspace(0,1,1001,dtype=torch.float64)[:,None];exact=points*(1-points)
    def error():return (torch.linalg.vector_norm(model.predict(points)-exact)/torch.linalg.vector_norm(exact)).item()
    start=time.perf_counter();history=[]
    for _ in range(200):
        model.train(1,0)
        if model.iter%10==0 or model.converged:
            row=dict(attempt=model.iter,relative_l2=error(),seconds=time.perf_counter()-start);history.append(row);print(json.dumps(row),flush=True)
        if model.converged or error()<1e-8:break
    report=dict(case='Margam 1D Poisson: u_xx + 2 = 0, u(0)=u(1)=0',scope='Analytical validation of an existing example; not a DSGNAR paper benchmark.',
        points=c.points,evaluation_points=1001,device='cpu',precision='float64',layers=list(c.layers),settings=c.dsgnar_settings,
        attempts=model.iter,seconds=time.perf_counter()-start,relative_l2=error(),history=history,summary=model.dsgnar_summary())
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if args.checkpoint:
        from margam.execution.codec import save_packet,cpu_tree
        args.checkpoint.parent.mkdir(parents=True,exist_ok=True);save_packet(args.checkpoint,cpu_tree(model.training_state()))
    print(json.dumps(dict(done=True,relative_l2=report['relative_l2'],attempts=model.iter)),flush=True)


if __name__=='__main__':main()
