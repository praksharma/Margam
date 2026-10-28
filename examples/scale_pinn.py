"""Short Scale-PINN implementation evidence; not a paper accuracy benchmark."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path


def main():
    import torch
    from margam.templates.examples import example_config
    from margam.templates.scale_examples import SCALE_EXAMPLES
    from margam.templates.poisson import build_pinn
    from copy import deepcopy
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--steps',type=int,default=20);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if not 2<=args.steps<=100000:parser.error('Use 2–100000 updates.')
    torch.set_num_threads(1);cases={}
    for key in SCALE_EXAMPLES:
        c=example_config(key);c.device=args.device;c.points=64;c.boundary_points=8;c.layers=(16,16);c.network_options['features']=8;c.iterations=args.steps
        m=build_pinn(c);half=args.steps//2;m.train(half,0);n=build_pinn(c);n.load_training_state(m.training_state());m.train(args.steps-half,0);n.train(args.steps-half,0)
        identical=m.loss_history==n.loss_history and all(torch.equal(value,n.network.state_dict()[name]) for name,value in m.network.state_dict().items())
        a=torch.tensor([[row[name] for name in row] for row in m.loss_history],dtype=torch.float64);b=torch.tensor([[row[name] for name in row] for row in n.loss_history],dtype=torch.float64)
        close=torch.allclose(a,b,rtol=1e-5,atol=1e-7) and all(torch.allclose(value,n.network.state_dict()[name],rtol=1e-5,atol=1e-7) for name,value in m.network.state_dict().items())
        if not close or args.device=='cpu' and not identical:raise RuntimeError('Scale continuation differs: '+key)
        control=deepcopy(c);control.scale_settings['strength']=0.;baseline=build_pinn(control);baseline.train(args.steps,0)
        cases[key]=dict(configuration=asdict(c),loss_weights=c.loss_weights,steps=m.iter,checkpoint_resume_identical=identical,checkpoint_resume_close=close,parameter_max_abs_difference=max((value-n.network.state_dict()[name]).abs().max().item() for name,value in m.network.state_dict().items()),initial_diagnostics=m.diagnostics[0],final_diagnostics=m.scale_summary(),uncorrected_same_batch_control=baseline.scale_summary(),initial_losses=m.loss_history[0],final_losses=m.loss_history[-1])
        print(key+': finite corrected training, '+('identical' if identical else 'float32-close')+' continuation, uncorrected control recorded')
    report=dict(scope='Software correctness and short training diagnostics only; no convergence, solution accuracy or speedup claim.',paper='https://arxiv.org/abs/2602.19475v1',torch=str(torch.__version__),device=args.device,gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,resume_tolerance=dict(rtol=1e-5,atol=1e-7,cpu_requires_bitwise_identity=True),cases=cases)
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    else:print(text)


if __name__=='__main__':main()
