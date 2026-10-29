"""Short residual-based attention evidence with exact point-mask continuation."""
import argparse,json
from pathlib import Path
from dataclasses import asdict


def main():
    import torch
    from margam.templates.examples import example_config
    from margam.templates.rba_examples import RBA_EXAMPLES
    from margam.templates.poisson import build_pinn
    from margam.components.outputs import OutputRequest,ReferenceSource,evaluate_outputs
    from margam.components.expressions import Expression
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--steps',type=int,default=6);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if not 2<=args.steps<=100000:parser.error('Use 2–100000 updates.')
    torch.set_num_threads(1);cases={}
    for key,(_,_,reference) in RBA_EXAMPLES.items():
        for mode in ('zero','normalized'):
            c=example_config(key);c.device=args.device;c.points=16;c.boundary_points=4;c.layers=(8,8);c.iterations=args.steps;c.rba_settings['initialization']=mode
            if c.network_type!='fcnn':c.network_options['features']=4
            m=build_pinn(c);half=args.steps//2;m.train(half,0);n=build_pinn(c);n.load_training_state(m.training_state());m.train(args.steps-half,0);n.train(args.steps-half,0)
            identical=m.loss_history==n.loss_history and torch.equal(m.attention,n.attention) and all(torch.equal(value,n.network.state_dict()[name]) for name,value in m.network.state_dict().items())
            if not identical:raise RuntimeError('RBA continuation differs: '+key)
            ref=None
            if reference:
                expression=Expression(reference,len(c.bounds[0]),target=True,coordinates=c.coordinate_names)
                ref=ReferenceSource('Manufactured sine Helmholtz',lambda points:expression(torch.tensor(points)).tolist())
            data=evaluate_outputs(m,c,[OutputRequest('result','comparison' if ref else 'solution',5,ref)])['result']
            cases[key+'_'+mode]=dict(configuration=asdict(c),steps=m.iter,checkpoint_resume_identical=identical,attention_requires_grad=m.attention.requires_grad,initial_losses=m.loss_history[0],final_losses=m.loss_history[-1],attention_summary=m.attention_summary(),point_rows=len(m.collocation_point_sample),evaluation_rows=len(data['points']),rmse=data['rmse'])
            print(key+' '+mode+': finite raw/weighted losses, detached attention and identical continuation')
    report=dict(scope='Short software correctness checks, not tuned convergence or paper accuracy.',paper='https://doi.org/10.1016/j.cma.2024.116805',device=args.device,torch=str(torch.__version__),gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,cases=cases)
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    else:print(text)


if __name__=='__main__':main()
