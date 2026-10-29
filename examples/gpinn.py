"""Short gPINN correctness and checkpoint evidence; no tuned accuracy claim."""
import argparse,json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path


def main():
    import torch
    from margam.templates.examples import example_config
    from margam.templates.gpinn_examples import GPINN_EXAMPLES
    from margam.templates.poisson import build_pinn
    from margam.components.outputs import evaluate_outputs,OutputRequest,ReferenceSource
    from margam.components.expressions import Expression
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--steps',type=int,default=6);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if not 4<=args.steps<=100000:parser.error('Use 4–100000 updates.')
    torch.set_num_threads(1);cases={}
    for key,(_,_,reference) in GPINN_EXAMPLES.items():
        c=example_config(key);c.device=args.device;c.points=16;c.boundary_points=4;c.layers=(8,8);c.iterations=args.steps
        if c.adaptive_sampling:c.adaptive_sampling.update(interval=2,candidates=32,batch=2,maximum=64,chunk=16)
        m=build_pinn(c);half=args.steps//2;m.train(half,0);resumed=build_pinn(c);resumed.load_training_state(m.training_state());m.train(args.steps-half,0);resumed.train(args.steps-half,0)
        identical=m.loss_history==resumed.loss_history and all(torch.equal(value,resumed.network.state_dict()[name]) for name,value in m.network.state_dict().items())
        if not identical:raise RuntimeError('Same-device gPINN continuation differs: '+key)
        ref=None
        if reference:
            expr=Expression(reference,len(c.bounds[0]),target=True,coordinates=c.coordinate_names)
            ref=ReferenceSource('Manufactured sine variation',lambda points:expr(torch.tensor(points)).tolist())
        data=evaluate_outputs(m,c,[OutputRequest('result','comparison' if ref else 'solution',5,ref)])['result']
        cases[key]=dict(configuration=asdict(c),steps=m.iter,checkpoint_resume_identical=identical,effective_loss_weights=m.weights,initial_losses=m.loss_history[0],final_losses=m.loss_history[-1],active_points=len(m.collocation_point_sample),sampling_history=[] if m.sampling_controller is None else m.sampling_controller.history,evaluation_rows=len(data['points']),rmse=data['rmse'])
        print(key+': finite residual-gradient training, identical continuation and requested evaluation passed')
    report=dict(scope='Short software correctness checks, not paper reproduction or tuned accuracy.',paper='https://arxiv.org/abs/2111.02801',torch=str(torch.__version__),device=args.device,gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,cases=cases)
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    else:print(text)


if __name__=='__main__':main()
