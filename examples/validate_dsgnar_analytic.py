"""Actual-device wave and multiscale parameter Jacobians versus generic autograd."""
import argparse
from copy import deepcopy
import hashlib,json
from pathlib import Path
import torch
from examples.dsgnar_analytic import build
from margam.methods.dsgnar import DSGNARPINN


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);checks=[]
    for case,network,width in [('wave','siren',16),('multiscale','fourier',24)]:
        m=build(case,network=network,width=width,points=7,boundary=5,batch=3,device=a.device)
        generic=DSGNARPINN(m.problem,deepcopy(m.network),settings=m.settings,weights=m.initial_weights,configuration=m.configuration).compile(device=a.device)
        for left,right in zip(m._conditions(),generic._conditions()):
            jmax=rmax=0.
            for (ja,ra),(jb,rb) in zip(m._jacobian_batches(left),generic._jacobian_batches(right)):
                torch.testing.assert_close(ja,jb,rtol=2e-11,atol=3e-10);torch.testing.assert_close(ra,rb,rtol=2e-12,atol=3e-11)
                jmax=max(jmax,(ja-jb).abs().max().item());rmax=max(rmax,(ra-rb).abs().max().item())
            checks.append(dict(case=case,network=network,parameters=sum(p.numel() for p in m.parameters),condition=left[0],jacobian_max_abs=jmax,residual_max_abs=rmax,status='passed'))
    from examples.validate_cuda import source_identity
    root=Path(__file__).resolve().parents[1];source=source_identity(root);source['experiment_script_sha256']=hashlib.sha256((root/'examples/dsgnar_analytic.py').read_bytes()).hexdigest()
    report=dict(scope='Actual-device full-network wave and multiscale Jacobian parity, including remainder batches; not paper accuracy or speed.',source=source,device=a.device,hardware=torch.cuda.get_device_name() if a.device=='cuda' else 'CPU',precision='float64',checks=checks)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
