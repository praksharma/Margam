"""CPU smoke comparison on interval Poisson, not a reproduction of paper results.

Run: .venv/bin/python -m examples.compare_self_adaptive --iterations 700
"""
import argparse
import json
from dataclasses import replace
import torch
from margam.templates.examples import example_config
from margam.templates.poisson import build_pinn


def compare(iterations,seeds):
    torch.set_num_threads(1)
    points=torch.linspace(0,1,201)[:,None];reference=points*(1-points)
    results=[]
    for seed in seeds:
        base=replace(example_config('interval_poisson'),network_seed=seed,iterations=iterations)
        for method in ('baseline','self_adaptive'):
            model=build_pinn(replace(base,method=method));model.train(iterations,display_every=0)
            error=model.predict(points)-reference
            record=dict(seed=seed,method=method,updates=iterations,rmse=error.square().mean().sqrt().item(),relative_l2=(error.norm()/reference.norm()).item())
            if method=='self_adaptive':record['adaptive_weights']=model.adaptive_summary()
            results.append(record)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations',type=int,default=700)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,11,19])
    args=parser.parse_args()
    if not 1<=args.iterations<=100000:parser.error('iterations must be 1–100000')
    print(json.dumps(compare(args.iterations,args.seeds),indent=2))
