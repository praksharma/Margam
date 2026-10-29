"""CPU check of non-unit box loss measures and importance-weighted continuation."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args()
    import torch
    from margam.templates.poisson import PoissonConfig,build_pinn
    from margam.components.adaptive_sampling import sampling_defaults
    torch.set_num_threads(1)
    config=PoissonConfig(pde_points='interior',x_max=3.,y_max=2.,points=16,boundary_points=8,layers=(16,16),iterations=40)
    mean=build_pinn(config);mean_losses={key:value.item() for key,value in mean.compute_losses().items()}
    config.loss_formulation='integral';integral=build_pinn(config)
    losses={key:value.item() for key,value in integral.compute_losses().items()}
    for name,value in losses.items():
        torch.testing.assert_close(torch.tensor(value),torch.tensor(mean_losses[name]*integral.loss_measures[name]),rtol=1e-6,atol=1e-6)
    report=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),device='cpu',torch=str(torch.__version__),
                scope='Loss measure scaling and exact continuation; not an accuracy/performance comparison.',
                configuration=asdict(config),measures=integral.loss_measures,initial_mean_losses=mean_losses,initial_integral_losses=losses,
                analytic_scaling_verified=True,runs={})
    for mode in ('fixed','importance'):
        if mode=='importance':
            config.adaptive_sampling=sampling_defaults('importance');config.adaptive_sampling.update(interval=10,candidates=128,chunk=32)
        model=build_pinn(config);model.train(15,display_every=0);state=model.training_state()
        resumed=build_pinn(config);resumed.load_training_state(state)
        model.train(25,display_every=0);resumed.train(25,display_every=0)
        identical=model.loss_history==resumed.loss_history and all(torch.equal(value,resumed.network.state_dict()[key]) for key,value in model.network.state_dict().items())
        if not identical:raise RuntimeError('Integral loss checkpoint changed the trajectory.')
        report['runs'][mode]=dict(configuration=asdict(config),steps=model.iter,checkpoint_resume_identical=identical,
                                 sampling_events=model.sampling_controller.history if model.sampling_controller is not None else [],final_losses=model.loss_history[-1])
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
