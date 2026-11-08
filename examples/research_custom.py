"""A standalone research script: custom PyTorch pieces, shared Margam trainer."""
import argparse
import csv
from functools import partial
import json
from pathlib import Path
import torch
from margam.research import Trainer, PINNProblem, PointConstraint, derivative, StepDecay


class MyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=torch.nn.Sequential(torch.nn.Linear(1,16),torch.nn.Tanh(),
            torch.nn.Linear(16,16),torch.nn.Tanh(),torch.nn.Linear(16,1))

    def forward(self,x):return self.layers(x)


def my_sampler():
    # No Margam sampler subclass or UI registration. Explicitly add boundary PDE rows.
    interior=torch.quasirandom.SobolEngine(1,scramble=True,seed=13).draw(64)
    return torch.cat((interior,torch.tensor([[0.],[1.]])))


def my_residual(x,u):return derivative(u,x,axis=0,order=2)+2


def my_loss(prediction,target):
    error=prediction-target
    return error.square().mean()+.01*error.pow(4).mean()


def clipped_step(trainer,terms,total):
    total.backward()
    torch.nn.utils.clip_grad_norm_(trainer.network.parameters(),1.,error_if_nonfinite=True)
    trainer.optimizer.step()


def make_trainer(device='auto',custom_step=False,decay=False):
    torch.manual_seed(7)
    problem=PINNProblem(my_residual,my_sampler,
        [PointConstraint('boundary',torch.tensor([[0.],[1.]]))])
    return Trainer(problem,MyNetwork(),loss={'pde':my_loss},weights={'boundary':10.},
        step=clipped_step if custom_step else None,
        checkpoint_id='research-poisson-example-v1').compile(
            optimizer=partial(torch.optim.AdamW,weight_decay=1e-4),lr=.003,device=device,
            schedule=StepDecay(every=2,gamma=.5) if decay else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--steps',type=int,default=8)
    parser.add_argument('--custom-step',action='store_true')
    parser.add_argument('--step-decay',action='store_true',help='Halve the learning rate every two completed updates.')
    parser.add_argument('--output',type=Path,default=Path('build/research-custom'))
    args=parser.parse_args()
    if not 2<=args.steps<=100000:parser.error('--steps must be 2..100000')
    if args.custom_step and args.step_decay:parser.error('--step-decay requires the default update')
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=True)
    model=make_trainer(args.device,args.custom_step,args.step_decay);resume=None
    if args.custom_step:model.train(args.steps,display_every=0)
    else:
        split=args.steps//2;model.train(split,display_every=0)
        model.save_checkpoint(args.output/'checkpoint.pt');model.train(args.steps-split,display_every=0)
        continued=make_trainer(args.device,decay=args.step_decay);continued.load_checkpoint(args.output/'checkpoint.pt')
        continued.train(args.steps-split,display_every=0)
        resume=model.loss_history==continued.loss_history and all(
            torch.equal(value,continued.network.state_dict()[key]) for key,value in model.network.state_dict().items())
        if not resume:raise RuntimeError('Same-device continuation differs from uninterrupted updates.')
        model=continued
    points=torch.linspace(0,1,101)[:,None];prediction=model.predict(points).cpu();reference=points*(1-points)
    with (args.output/'predictions.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(('x','predicted_u','reference_u'))
        writer.writerows(zip(points[:,0].tolist(),prediction[:,0].tolist(),reference[:,0].tolist()))
    result=dict(scope='Research API smoke example; not an accuracy benchmark.',steps=model.iter,
        device=str(model.device),torch_version=torch.__version__,pde_rows=len(model.collocation_point_sample),
        custom_network=True,custom_sampler=True,custom_pde_loss=True,optimizer='AdamW',
        custom_step=args.custom_step,step_decay=args.step_decay,exact_resume=resume,
        next_learning_rates=[group['lr'] for group in model.optimizer.param_groups],last_preupdate_loss=model.loss_history[-1],
        postupdate_rmse=((prediction-reference).square().mean().sqrt().item()))
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
