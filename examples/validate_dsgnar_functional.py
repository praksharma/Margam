"""Check compiled paper-sized Burgers derivatives against generic autograd."""
import argparse,json,time
from pathlib import Path
from copy import deepcopy
import torch
from examples.dsgnar_accuracy import PeriodicMLP,FunctionalBurgersTrainer
from margam.methods.dsgnar import DSGNARPINN
from margam.components import PINNProblem,PointConstraint
from margam.components.differential import derivative
from margam.components.dsgnar_config import dsgnar_defaults


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for this compiled benchmark check.')
    torch.set_num_threads(1);torch.manual_seed(0)
    points=torch.rand(128,2);points[:,0]=points[:,0]*2-1
    initial=points.clone();initial[:,1]=0.
    def residual(x,u):return derivative(u,x,1)+u*derivative(u,x,0)-.01*derivative(u,x,0,2)
    problem=PINNProblem(residual,points,[PointConstraint('initial',initial,lambda x:-torch.sin(torch.pi*x[:,0:1]))])
    settings=dsgnar_defaults();settings.update(batch_size=128,precision='float32')
    network=PeriodicMLP();reference=DSGNARPINN(problem,network,settings=settings).compile(device='cuda')
    tested=FunctionalBurgersTrainer(problem,deepcopy(network),settings=settings).compile(device='cuda')
    tested.functional_blocks={name:torch.compile(function,fullgraph=True) for name,function in tested.functional_blocks.items()}
    tested.functional_values={name:torch.compile(function,fullgraph=True) for name,function in tested.functional_values.items()}
    rows=[];start=time.perf_counter()
    for a,b in zip(reference._conditions(),tested._conditions()):
        ja,ra=next(reference._jacobian_batches(a));jb,rb=next(tested._jacobian_batches(b))
        torch.testing.assert_close(ja,jb,rtol=2e-4,atol=3e-6)
        torch.testing.assert_close(ra,rb,rtol=2e-5,atol=1e-6)
        probe=next(tested._batches(b,batch_size=128))
        torch.testing.assert_close(probe,ra,rtol=2e-5,atol=1e-6)
        rows.append(dict(condition=a[0],rows=len(ra),parameters=ja.shape[1],jacobian_max_abs=(ja-jb).abs().max().item(),residual_max_abs=(ra-rb).abs().max().item()))
    report=dict(scope='Compiled functional PDE/IC derivatives versus generic reverse autograd; not an accuracy or speed reproduction.',device='cuda',precision='float32',torch=str(torch.__version__),passed=True,seconds=time.perf_counter()-start,cases=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
