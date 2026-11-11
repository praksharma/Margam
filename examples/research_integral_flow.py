"""Explicit total-flow quadrature with a scalar parabolic axial-velocity ansatz.

This is a research API/gradient/checkpoint example, not a full Navier-Stokes case.
"""
import argparse
import json
from pathlib import Path
import torch
from margam.research import Trainer,PINNProblem,PointConstraint,IntegralConstraint,StepDecay,derivative


class AxialVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__();self.amplitude=torch.nn.Parameter(torch.tensor(.2,dtype=torch.float64))
    def forward(self,points):
        y=points[:,1:2]
        return self.amplitude*y*(2-y)


def make_trainer(device='cpu',quadrature=None,target=4/3):
    points=torch.tensor([[1.,0.],[1.,1.],[1.,2.]],dtype=torch.float64)
    weights=torch.tensor([1/3,4/3,1/3],dtype=torch.float64) if quadrature is None else quadrature
    problem=PINNProblem(lambda x,u:derivative(u,x,0),torch.tensor([[.5,.5],[.5,1.5]],dtype=torch.float64),[
        PointConstraint('walls',torch.tensor([[.5,0.],[.5,2.]],dtype=torch.float64)),
        IntegralConstraint('flow_rate',points,target=target,quadrature_weights=weights),
    ])
    return Trainer(problem,AxialVelocity(),checkpoint_id='scalar-axial-integral-flow-v1').compile(
        lr=.05,device=device,schedule=StepDecay(every=5,gamma=.5))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda','auto'),default='cpu')
    parser.add_argument('--output',type=Path,help='Optional JSON evidence; no files by default.')
    args=parser.parse_args();torch.set_num_threads(1)
    full=make_trainer(args.device);full.train(10,0)
    split=make_trainer(args.device);split.train(4,0);state=split.training_state()
    continued=make_trainer(args.device);continued.load_training_state(state);continued.train(6,0)
    exact=full.loss_history==continued.loss_history and all(torch.equal(v,continued.network.state_dict()[k]) for k,v in full.network.state_dict().items())
    if not exact:raise RuntimeError('Integral-constraint continuation changed the trajectory.')
    points=full._constraints[-1][1];weights=full._integral_weights['flow_rate']
    flux=(weights*full.predict(points)).sum().item()
    report=dict(scope='Scalar parabolic axial-velocity ansatz; fixed Simpson quadrature and continuation check, not a full flow or accuracy benchmark.',
        device=str(full.device),torch_version=torch.__version__,steps=full.iter,quadrature_points=3,
        quadrature_measure=weights.sum().item(),target_flow=4/3,predicted_flow=flux,
        amplitude=full.network.amplitude.item(),exact_resume=exact,last_preupdate_loss=full.loss_history[-1])
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output is not None:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
