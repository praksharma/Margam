"""Expert PINN workflow evidence on Burgers; no accuracy reproduction."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    parser.add_argument('--steps',type=int,default=100);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if not 2<=args.steps<=100000:parser.error('Use 2–100000 steps.')
    import torch
    from margam.templates.examples import example_config
    from margam.templates.poisson import build_pinn
    torch.set_num_threads(1);config=example_config('expert_burgers');config.device=args.device
    config.points=128;config.layers=(16,16,16);config.network_options.update(features=8)
    config.iterations=args.steps;config.expert_settings.update(balance_every=10,decay_steps=100)
    m=build_pinn(config);phase=args.steps//2;m.train(phase,display_every=0);state=m.training_state()
    resumed=build_pinn(config);resumed.load_training_state(state)
    m.train(args.steps-phase,display_every=0);resumed.train(args.steps-phase,display_every=0)
    identical=m.loss_history==resumed.loss_history and all(torch.equal(value,resumed.network.state_dict()[name]) for name,value in m.network.state_dict().items())
    if not identical:raise RuntimeError('Expert continuation changed its training trajectory.')
    report=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),torch=str(torch.__version__),device=str(m.device),
        gpu=torch.cuda.get_device_name(m.device) if m.device.type=='cuda' else None,
        scope='Independent Expert’s Guide v1 workflow: finite training, balancing/sampling/decay and exact same-device continuation; no accuracy claim.',
        paper='https://arxiv.org/abs/2308.08468',paper_version='2308.08468v1',configuration=asdict(config),
        steps=m.iter,checkpoint_resume_identical=identical,initial_losses=m.loss_history[0],final_losses=m.loss_history[-1],
        summary=m.expert_summary(),balance_history=m.balance_history,reference_solution_used=False)
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
