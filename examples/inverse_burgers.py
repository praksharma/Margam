"""Short, reproducible inverse-method evidence; no accuracy tuning or sweep."""
import argparse
from copy import deepcopy
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=('cpu','cuda'),default='cpu');parser.add_argument('--updates',type=int,default=4);parser.add_argument('--output',type=Path,default=Path('build/inverse-burgers/evidence.json'));args=parser.parse_args()
    if not 2<=args.updates<=100000:parser.error('Use 2–100000 updates.')
    import torch
    from margam.templates.inverse_burgers import inverse_example_config
    from margam.templates.poisson import build_pinn
    if args.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA is unavailable in this environment.')
    torch.set_num_threads(1);c=inverse_example_config();c.device=args.device;c.iterations=args.updates
    whole=build_pinn(c);whole.train(args.updates,0)
    first=build_pinn(c);split=args.updates//2;first.train(split,0);state=first.training_state()
    resumed=build_pinn(c);resumed.load_training_state(state);resumed.train(args.updates-split,0)
    exact=(whole.loss_history==resumed.loss_history and whole.inverse_history==resumed.inverse_history and all(torch.equal(value,resumed.network.state_dict()[key]) for key,value in whole.network.state_dict().items()))
    assert exact
    evidence=dict(method=c.method,scope='Short implementation/resume check; not paper accuracy reproduction.',device=str(whole.device),torch=torch.__version__,observations=2000,source='assets/training/inverse_burgers/source.json',network=c.network_type,layers=list(c.layers),seed=c.seed,learning_rate=c.lr,updates=args.updates,initial_guesses=deepcopy(c.inverse_settings),estimated_parameters=whole.estimated_parameters(),final_pre_update_losses=whole.loss_history[-1],exact_same_device_resume=exact)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))

if __name__=='__main__':main()
