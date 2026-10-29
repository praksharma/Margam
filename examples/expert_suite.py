"""Short correctness evidence for all seven Expert examples; no accuracy claim."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path


def main():
    from margam.templates.expert_examples import CASES,case_config,samples
    from margam.templates.poisson import build_pinn
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--steps',type=int,default=10)
    parser.add_argument('--case',choices=['all',*CASES],default='all');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not 2<=args.steps<=100000:parser.error('Use 2–100000 steps.')
    torch.set_num_threads(1);cases={}
    for key in CASES if args.case=='all' else [args.case]:
        c=case_config(key);c.pde_points='both';c.device=args.device;c.points=32;c.boundary_points=4;c.layers=(16,16);c.network_options['features']=4
        c.iterations=args.steps;c.expert_settings.update(chunks=4,balance_every=2)
        m=build_pinn(c);phase=args.steps//2;m.train(phase,display_every=0)
        n=build_pinn(c);n.load_training_state(m.training_state());m.train(args.steps-phase,display_every=0);n.train(args.steps-phase,display_every=0)
        identical=m.loss_history==n.loss_history and all(torch.equal(v,n.network.state_dict()[name]) for name,v in m.network.state_dict().items())
        if not identical:raise RuntimeError('Continuation differs: '+key)
        points,sets=samples(c)
        cases[key]=dict(label=CASES[key]['label'],configuration=asdict(c),interior_count=len(points),constraint_counts={name:len(rows) for name,rows in sets.items()},fields=c.field_names,coordinates=c.coordinate_names,steps=m.iter,initial_losses=m.loss_history[0],final_losses=m.loss_history[-1],checkpoint_resume_identical=identical)
        print(key+': finite training and identical continuation')
    report=dict(scope='Short software correctness checks; not paper accuracy reproduction or convergence evidence.',torch=str(torch.__version__),device=args.device,gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,paper='https://arxiv.org/abs/2308.08468v1',cases=cases)
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    else:print(text)


if __name__=='__main__':main()
