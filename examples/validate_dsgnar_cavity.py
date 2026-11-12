"""Real-device cavity Jacobian parity; optional independent reference refinement."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from examples.dsgnar_cavity import CavitySiren,FunctionalCavityTrainer,make_problem,sample_points
from margam.methods.dsgnar import DSGNARPINN
from margam.components.dsgnar_config import dsgnar_defaults


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--coarse',type=Path);parser.add_argument('--fine',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(3);old=torch.get_default_dtype()
        try:torch.set_default_dtype(torch.float64);network=CavitySiren(60,4)
        finally:torch.set_default_dtype(old)
    points,wall=sample_points(5,9);settings=dsgnar_defaults();settings.update(batch_size=3,sketch_size=8,probes=3)
    generic=DSGNARPINN(make_problem(points,wall),network,settings=settings).compile(device=args.device)
    functional=FunctionalCavityTrainer(make_problem(points,wall),deepcopy(network),settings=settings).compile(device=args.device)
    checks=[]
    for a,b in zip(generic._conditions(),functional._conditions()):
        jmax=rmax=0.
        for (ja,ra),(jb,rb) in zip(generic._jacobian_batches(a),functional._jacobian_batches(b)):
            torch.testing.assert_close(ja,jb,rtol=2e-11,atol=2e-12);torch.testing.assert_close(ra,rb,rtol=2e-12,atol=2e-13)
            jmax=max(jmax,(ja-jb).abs().max().item());rmax=max(rmax,(ra-rb).abs().max().item())
        checks.append(dict(condition=a[0],jacobian_max_abs=jmax,residual_max_abs=rmax,status='passed'))
    from examples.validate_cuda import source_identity
    source=source_identity(Path(__file__).resolve().parents[1]);source['cavity_script_sha256']=hashlib.sha256(Path('examples/dsgnar_cavity.py').read_bytes()).hexdigest()
    report=dict(scope='Actual-device full 60x4 cavity network parameter Jacobians versus generic autograd; no accuracy or speed claim.',source=source,
        device=args.device,hardware=torch.cuda.get_device_name() if args.device=='cuda' else 'CPU',parameters=sum(p.numel() for p in network.parameters()),precision='float64',checks=checks)
    if args.coarse or args.fine:
        if not(args.coarse and args.fine):parser.error('Supply both reference resolutions.')
        with np.load(args.coarse,allow_pickle=False) as coarse,np.load(args.fine,allow_pickle=False) as fine:
            stride=(len(fine['x'])-1)//(len(coarse['x'])-1)
            for axis in ('x','y'):np.testing.assert_array_equal(coarse[axis],fine[axis][::stride])
            if not bool(coarse['converged']) or not bool(fine['converged']):raise ValueError('References must be converged.')
            a=np.stack((coarse['u'],coarse['v']),-1);b=np.stack((fine['u'][::stride,::stride],fine['v'][::stride,::stride]),-1)
            report['reference_refinement']=dict(coarse_grid=len(coarse['x']),fine_grid=len(fine['x']),relative_velocity_difference=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
                max_abs_velocity_difference=float(np.max(np.abs(a-b))),coarse_steady_residual=float(coarse['steady_residual']),fine_steady_residual=float(fine['steady_residual']),
                coarse_seconds=float(coarse['seconds']),fine_seconds=float(fine['seconds']),coarse_sha256=hashlib.sha256(args.coarse.read_bytes()).hexdigest(),fine_sha256=hashlib.sha256(args.fine.read_bytes()).hexdigest(),
                scope='Difference on coincident nodes; evidence of discretization error, not a rigorous error bound or Dedalus reproduction.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
