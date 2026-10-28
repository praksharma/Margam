"""Small CPU workflow checks for both residual samplers; not an accuracy study."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Optional JSON evidence file; no output file by default.')
    args=parser.parse_args()
    import torch
    from margam.components.adaptive_sampling import sampling_defaults
    from margam.templates.poisson import PoissonConfig,build_pinn
    torch.set_num_threads(1)
    report=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),device='cpu',torch=str(torch.__version__),
                scope='Scheduling, point counts and exact checkpoint resume; no numerical accuracy comparison.',runs={})
    for sampler in ('rar','importance'):
        settings=sampling_defaults(sampler);settings.update(interval=20,candidates=256,chunk=64)
        if sampler=='rar':settings.update(batch=8,maximum=64)
        config=PoissonConfig(points=36,boundary_points=8,layers=(16,16),iterations=60,adaptive_sampling=settings)
        started=time.perf_counter();model=build_pinn(config);initial=len(model.collocation_point_sample)
        model.train(25,display_every=0);checkpoint=model.training_state()
        resumed=build_pinn(config);resumed.load_training_state(checkpoint)
        model.train(35,display_every=0);resumed.train(35,display_every=0)
        identical=model.loss_history==resumed.loss_history and all(torch.equal(value,resumed.network.state_dict()[key]) for key,value in model.network.state_dict().items())
        if not identical:raise RuntimeError('Resume changed the training trajectory.')
        run=dict(configuration=asdict(config),steps=model.iter,initial_points=initial,final_points=len(model.collocation_point_sample),
                 sampling_history=model.sampling_controller.history,checkpoint_resume_identical=identical,
                 final_losses=model.loss_history[-1],elapsed_seconds=time.perf_counter()-started)
        report['runs'][sampler]=run
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
