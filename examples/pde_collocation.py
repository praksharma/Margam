"""Check explicit boundary collocation and same-device training continuation."""
import argparse,json
from pathlib import Path


def main():
    import torch
    from margam.templates.poisson import PoissonConfig,BoundarySampling,build_pinn
    from margam.templates.examples import example_config
    from margam.components.adaptive_sampling import sampling_defaults
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--output',type=Path,default=Path('build/pde-collocation.json'))
    args=parser.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA is unavailable; use --device cpu explicitly.')
    torch.set_num_threads(1)
    records=[]
    for method in ('baseline','importance','gpinn','rba','expert'):
        c=example_config('expert_burgers') if method=='expert' else PoissonConfig(shape='Interval',points=8,layers=(8,8),pde_points='both')
        c.device=args.device
        if method=='expert':
            c.points=16;c.layers=(8,8);c.network_type='fcnn';c.network_options={};c.expert_settings['chunks']=2
            c.boundary_sampling={side:BoundarySampling(2,'random',10) for side in c.sides}
        elif method=='importance':
            c.adaptive_sampling=sampling_defaults('importance');c.adaptive_sampling.update(interval=1,candidates=16,chunk=8)
        else:c.method=method
        full=build_pinn(c);full.train(4,0)
        part=build_pinn(c);part.train(2,0);resumed=build_pinn(c);resumed.load_training_state(part.training_state());resumed.train(2,0)
        assert full.loss_history==resumed.loss_history,method
        for key,value in full.network.state_dict().items():assert torch.equal(value,resumed.network.state_dict()[key]),(method,key)
        records.append(dict(method=method,pde_rows=len(full.collocation_point_sample),exact_resume=True,updates=4))
    record=dict(device=torch.cuda.get_device_name(0) if args.device=='cuda' else 'cpu',torch=str(torch.__version__),scope='Boundary-inclusive PDE loss and exact same-device 2+2 continuation; no accuracy claim.',runs=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
