"""Record weighted imported-boundary losses and exact same-device continuation."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--output',type=Path,default=Path('build/boundary-weights.json'))
    args=parser.parse_args()
    import torch
    from margam.templates.prepared_triangle import prepared_triangle_config
    from margam.templates.poisson import build_pinn
    torch.set_num_threads(1)
    config=prepared_triangle_config();config.method='baseline';config.device=args.device
    config.pde_points='both';config.layers=(8,8);config.boundary_weight=3.
    config.bc_weights=dict(base=2.,upright=.5,slope=3.)
    full=build_pinn(config);full.train(4,0)
    part=build_pinn(config);part.train(2,0)
    resumed=build_pinn(config);resumed.load_training_state(part.training_state());resumed.train(2,0)
    assert full.loss_history==resumed.loss_history
    for name,value in full.network.state_dict().items():torch.testing.assert_close(value,resumed.network.state_dict()[name],rtol=0,atol=0)
    for row in full.loss_history:
        reconstructed=row['pde']*config.pde_weight+sum(row['weighted:'+name] for name in config.constraint_names)
        assert abs(reconstructed-row['total'])<=max(1e-6,abs(row['total'])*1e-6)
    report=dict(case='Prepared triangle / baseline / mixed boundary operators',device=str(full.device),torch=str(torch.__version__),
                updates=4,coefficients=config.bc_weights,effective_weights=config.loss_weights,
                pde_points=len(full.collocation_point_sample),
                boundary_counts={condition.name:len(points) for condition,points,target in full._constraints},
                final_losses=full.loss_history[-1],exact_same_device_resume=True,
                objective_reconciled=True,scope='Loss composition and continuation correctness, not solution accuracy.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
