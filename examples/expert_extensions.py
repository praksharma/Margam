"""Small NTK/time-window correctness records, with no accuracy claim."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path


def main():
    import torch
    from margam.templates.examples import example_config
    from margam.templates.expert_examples import CASES
    from margam.templates.poisson import build_pinn
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--output',type=Path)
    args=parser.parse_args();torch.set_num_threads(1);records={}
    for key in ('expert_burgers',*CASES):
        c=example_config(key);c.device=args.device;c.points=16;c.boundary_points=3;c.layers=(8,8);c.network_options['features']=4
        windows=3 if c.is_time_dependent else 1
        c.expert_settings.update(balance_mode='ntk',ntk_points=2,windows=windows,window_steps=3,chunks=2,balance_every=2)
        c.iterations=3*windows
        m=build_pinn(c);phase=c.iterations//2;m.train(phase,0);n=build_pinn(c);n.load_training_state(m.training_state());m.train(c.iterations-phase,0);n.train(c.iterations-phase,0)
        identical=m.loss_history==n.loss_history and all(torch.equal(value,n.network.state_dict()[name]) for name,value in m.network.state_dict().items())
        parameter_difference=max((value-n.network.state_dict()[name]).abs().max().item() for name,value in m.network.state_dict().items())
        a=torch.tensor([[row[name] for name in row] for row in m.loss_history],dtype=torch.float64)
        b=torch.tensor([[row[name] for name in row] for row in n.loss_history],dtype=torch.float64)
        loss_difference=(a-b).abs().max().item()
        close=torch.allclose(a,b,rtol=1e-5,atol=1e-7) and all(torch.allclose(value,n.network.state_dict()[name],rtol=1e-5,atol=1e-7) for name,value in m.network.state_dict().items())
        if not close or args.device=='cpu' and not identical:raise RuntimeError('Continuation differs beyond the documented tolerance: '+key)
        records[key]=dict(configuration=asdict(c),checkpoint_resume_identical=identical,checkpoint_resume_close=close,parameter_max_abs_difference=parameter_difference,loss_max_abs_difference=loss_difference,summary=m.expert_summary(),final_losses=m.loss_history[-1],balance_histories=[state['expert']['history'] for state in m.completed]+[m.active.balance_history] if windows>1 else [m.balance_history])
        print(key+': finite NTK training and '+('identical' if identical else 'float32-close')+' continuation'+(' through three windows' if windows>1 else ' (steady)'))
    report=dict(scope='Software correctness only, not convergence or paper accuracy reproduction.',resume_tolerance=dict(rtol=1e-5,atol=1e-7,cpu_requires_bitwise_identity=True,cuda_requires_numerical_equivalence=True),torch=str(torch.__version__),device=args.device,gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,cases=records)
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    else:print(text)


if __name__=='__main__':main()
