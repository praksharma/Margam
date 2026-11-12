"""Independent DSGNAR smooth-lid cavity, paper Appendix C.7 (Re=100).

Five separately balanced scalar conditions; pressure offset is unconstrained.
Optional finite-difference reference is independent and resolution limited.
No upstream code or data is copied. Native UI cavity uses its existing pressure
anchor and grouped/side-specific conditions, so is labelled a variation.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from margam.research import DSGNARPINN,PINNProblem,PointConstraint,derivative
from margam.components.dsgnar_config import dsgnar_defaults


class CavitySiren(torch.nn.Module):
    """Paper B.4 sine MLP with positive trainable per-layer frequencies."""
    def __init__(self,width=60,depth=4):
        super().__init__();sizes=[2]+[width]*depth+[3]
        self.layers=torch.nn.ModuleList(torch.nn.Linear(a,b) for a,b in zip(sizes,sizes[1:]))
        self.log_frequencies=torch.nn.Parameter(torch.zeros(depth))
        for i,layer in enumerate(self.layers):
            limit=1/layer.in_features if i==0 else math.sqrt(6/layer.in_features) if i<depth else math.sqrt(3/layer.in_features)
            torch.nn.init.uniform_(layer.weight,-limit,limit)
            if i==0:torch.nn.init.uniform_(layer.bias,-1,1)
            elif i==depth:torch.nn.init.zeros_(layer.bias)
            else:torch.nn.init.normal_(layer.bias,std=.01)
    def forward(self,x):
        h=2*x-1
        for frequency,layer in zip(self.log_frequencies,self.layers[:-1]):h=torch.sin(frequency.exp()*layer(h))
        return self.layers[-1](h)


def residuals(x,values):
    u,v,p=(values[:,i:i+1] for i in range(3))
    d=lambda f,axis,order=1:derivative(f,x,axis,order)
    return (u*d(u,0)+v*d(u,1)+d(p,0)-.01*(d(u,0,2)+d(u,1,2)),
            u*d(v,0)+v*d(v,1)+d(p,1)-.01*(d(v,0,2)+d(v,1,2)),d(u,0)+d(v,1))


def lid_target(x):return torch.where(x[:,1:2]==1,4*x[:,0:1]*(1-x[:,0:1]),0.)


def sample_points(count,boundary,seed=0,*,dtype=torch.float64,sampling='uniform'):
    if min(count,boundary)<4:raise ValueError('At least four interior and boundary points required.')
    g=torch.Generator().manual_seed(seed)
    interior=torch.rand(count,2,generator=g,dtype=dtype)
    wall=torch.rand(boundary,2,generator=g,dtype=dtype)
    # Balanced wall coverage, including non-multiples of four, with fixed draws.
    for side in range(4):wall[side::4,side//2]=side%2
    if sampling=='corner_enriched':
        n=count//16;b=boundary//16
        if n:
            interior[-2*n:-n,1]=.9+.1*interior[-2*n:-n,1]
            half=n//2;interior[-n:]=.1*interior[-n:].square();interior[-n:,1]=1-interior[-n:,1]
            if half:interior[-half:,0]=1-interior[-half:,0]
        if b:
            wall[-b:,1]=1;wall[-b:,0]*=.1
            if b//2:wall[-(b//2):,0]=1-wall[-(b//2):,0]
    elif sampling!='uniform':raise ValueError('Unknown sampling policy.')
    return interior,wall


def make_problem(interior,wall):
    return PINNProblem(lambda x,y:residuals(x,y)[0],interior,[
        PointConstraint('momentum_y',interior,0.,lambda x,y:residuals(x,y)[1]),
        PointConstraint('continuity',interior,0.,lambda x,y:residuals(x,y)[2]),
        PointConstraint('boundary_u',wall,lid_target,lambda x,y:y[:,0:1]),
        PointConstraint('boundary_v',wall,0.,lambda x,y:y[:,1:2])],data_identity='smooth-lid-cavity-Re100-five-conditions-v1')


class FunctionalCavityTrainer(DSGNARPINN):
    """Validated pointwise derivatives; shared DSGNAR sketch/update/checkpoint."""
    def compile(self,*args,**kwargs):
        super().compile(*args,**kwargs)
        from torch.func import functional_call,jacrev,vmap
        def value(parameters,point):return functional_call(self.network,parameters,point[None,:])[0]
        gradient=jacrev(value,argnums=1)
        hessian=jacrev(gradient,argnums=1)
        def equations(parameters,point):
            fields=value(parameters,point);d=gradient(parameters,point);dd=hessian(parameters,point)
            return torch.stack((fields[0]*d[0,0]+fields[1]*d[0,1]+d[2,0]-.01*(dd[0,0,0]+dd[0,1,1]),
                fields[0]*d[1,0]+fields[1]*d[1,1]+d[2,1]-.01*(dd[1,0,0]+dd[1,1,1]),d[0,0]+d[1,1]))
        self.functional_values={};self.functional_blocks={}
        for name,index in [('pde',0),('momentum_y',1),('continuity',2),('boundary_u',0),('boundary_v',1)]:
            fn=equations if name in ('pde','momentum_y','continuity') else value
            def component(parameters,point,fn=fn,index=index):return fn(parameters,point)[index]
            def auxiliary(parameters,point,fn=component):
                result=fn(parameters,point);return result,result
            self.functional_values[name]=vmap(component,in_dims=(None,0))
            self.functional_blocks[name]=vmap(jacrev(auxiliary,argnums=0,has_aux=True),in_dims=(None,0))
        return self
    def _batches(self,condition,batch_size=None):
        name,points,target,_=condition;parameters=dict(self.network.named_parameters());batch_size=batch_size or self.settings['batch_size']
        for start in range(0,len(points),batch_size):
            batch=points[start:start+batch_size]
            with torch.no_grad():values=self.functional_values[name](parameters,batch)
            expected=target if target.ndim==0 else target[start:start+len(batch)].reshape(-1)
            yield (values-expected).detach()
    def _jacobian_batches(self,condition):
        name,points,target,_=condition;parameters=dict(self.network.named_parameters())
        for start in range(0,len(points),self.settings['batch_size']):
            batch=points[start:start+self.settings['batch_size']];gradients,values=self.functional_blocks[name](parameters,batch)
            jacobian=torch.cat([gradients[key].reshape(len(batch),-1) for key in parameters],dim=1)
            expected=target if target.ndim==0 else target[start:start+len(batch)].reshape(-1)
            yield jacobian.detach(),(values-expected).detach()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda');parser.add_argument('--width',type=int,default=60);parser.add_argument('--depth',type=int,default=4)
    parser.add_argument('--points',type=int,default=32768);parser.add_argument('--boundary',type=int,default=16384)
    parser.add_argument('--sketch',type=int,default=4000);parser.add_argument('--batch',type=int,default=64);parser.add_argument('--probes',type=int,default=24)
    parser.add_argument('--attempts',type=int,default=150);parser.add_argument('--seconds',type=float,default=3600);parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--sampling',choices=('uniform','corner_enriched'),default='uniform');parser.add_argument('--compile-functional',action='store_true')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--checkpoint',type=Path);parser.add_argument('--resume',type=Path)
    parser.add_argument('--reference',type=Path);parser.add_argument('--csv',type=Path)
    parser.add_argument('--evaluate-only',action='store_true',help='Reload and evaluate without a training attempt.')
    args=parser.parse_args()
    if args.evaluate_only and not args.resume:parser.error('--evaluate-only requires --resume.')
    if not 4<=args.width<=128 or not 1<=args.depth<=8 or not 1<=args.attempts<=10000 or not math.isfinite(args.seconds) or args.seconds<=0:parser.error('Invalid architecture or run budget.')
    from contextlib import nullcontext
    # Ten pointwise vmap wrappers share a Python code object. Full and remainder
    # batches legitimately exceed Dynamo's default eight cached specializations.
    # Keep a bounded, CLI-local allowance and restore it on exit.
    context=torch._dynamo.config.patch(recompile_limit=32) if args.compile_functional else nullcontext()
    with context:run(args)


def run(args):
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(args.seed)
    points,wall=sample_points(args.points,args.boundary,args.seed,sampling=args.sampling)
    settings=dsgnar_defaults();settings.update(sketch_size=args.sketch,batch_size=args.batch,probes=args.probes,stage_grace=80,min_radius=1e-8,seed=args.seed)
    configuration=dict(case='smooth-lid-cavity-C7-v1',reynolds=100.,width=args.width,depth=args.depth,seed=args.seed,sampling=args.sampling,
        derivative_backend='compiled-functional' if args.compile_functional else 'functional',torch_version=str(torch.__version__) if args.compile_functional else None)
    model=FunctionalCavityTrainer(make_problem(points,wall),CavitySiren(args.width,args.depth),weights=dict(pde=.01,momentum_y=.01,continuity=.01,boundary_u=1.,boundary_v=1.),settings=settings,configuration=configuration).compile(device=args.device)
    if args.resume:
        from margam.execution.codec import load_packet
        model.load_training_state(load_packet(args.resume))
    if args.compile_functional:
        model.functional_blocks={name:torch.compile(fn,fullgraph=True) for name,fn in model.functional_blocks.items()}
        model.functional_values={name:torch.compile(fn,fullgraph=True) for name,fn in model.functional_values.items()}
    reference=None;reference_meta=None
    if args.reference:
        with np.load(args.reference,allow_pickle=False) as archive:
            if not bool(archive['converged']) or float(archive['reynolds'])!=100:raise ValueError('Reference must be a converged Re=100 smooth-lid solution.')
            xx,yy=np.meshgrid(archive['x'],archive['y']);evaluation=np.column_stack((xx.ravel(),yy.ravel()));reference=np.column_stack((archive['u'].ravel(),archive['v'].ravel()))
            reference_meta=dict(method=str(archive['method']),grid=len(archive['x']),steady_residual=float(archive['steady_residual']),sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest())
    else:evaluation=np.array([(x,y) for y in np.linspace(0,1,65) for x in np.linspace(0,1,65)])
    from examples.validate_cuda import source_identity
    source=source_identity(Path(__file__).resolve().parents[1]);source['benchmark_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    history=[];started=time.perf_counter();initial=model._losses();starting_attempt=model.iter
    starting_weights=model._flat().clone();starting_rng=model.generator.get_state().clone()
    def metrics():
        prediction=model.predict(evaluation).detach().cpu().numpy()
        return prediction,dict(relative_velocity_l2=float(np.linalg.norm(prediction[:,:2]-reference)/np.linalg.norm(reference)) if reference is not None else None,
            velocity_min=prediction[:,:2].min(axis=0).tolist(),velocity_max=prediction[:,:2].max(axis=0).tolist())
    def save():
        prediction,measurement=metrics()
        report=dict(case='DSGNAR smooth-lid cavity; Re=100; paper Appendix C.7',paper='https://arxiv.org/abs/2607.02194',source=source,configuration=configuration,
            scope='Independent five-condition implementation and measured training; finite-difference reference differs from paper Dedalus protocol.',
            paper_target_velocity_l2=1.13e-4,paper_seconds=402.9,device=args.device,hardware=torch.cuda.get_device_name() if args.device=='cuda' else 'CPU',
            precision='float64',parameters=sum(p.numel() for p in model.parameters),pde_points=len(points),boundary_points=len(wall),settings=settings,
            starting_attempt=starting_attempt,attempts=model.iter,inference_only=args.evaluate_only,attempts_this_run=model.iter-starting_attempt,
            inference_preserved_weights_iteration_rng=True if args.evaluate_only else None,checkpoint_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None,seconds=time.perf_counter()-started,initial_losses=initial,final_losses=model._losses(),history=history,reference=reference_meta,**measurement)
        args.output.parent.mkdir(parents=True,exist_ok=True);tmp=args.output.with_suffix('.partial');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(args.output)
        if args.checkpoint:
            from margam.execution.codec import save_packet,cpu_tree
            args.checkpoint.parent.mkdir(parents=True,exist_ok=True);save_packet(args.checkpoint,cpu_tree(model.training_state()))
        if args.csv:
            args.csv.parent.mkdir(parents=True,exist_ok=True);np.savetxt(args.csv,np.column_stack((evaluation,prediction)),delimiter=',',header='x,y,u,v,p',comments='')
        return measurement
    print(json.dumps(dict(start=True,parameters=sum(p.numel() for p in model.parameters),initial_losses=initial)),flush=True)
    for _ in range(0 if args.evaluate_only else args.attempts):
        model.train(1,0);row=dict(attempt=model.iter,seconds=time.perf_counter()-started,losses=model._losses(),radius=model.radius,accepted=model.diagnostics[-1]['accepted']);history.append(row)
        if model.iter%5==0 or model.converged:row.update(save())
        print(json.dumps(row),flush=True)
        if model.converged or time.perf_counter()-started>args.seconds:break
    if args.evaluate_only:
        assert model.iter==starting_attempt and torch.equal(model._flat(),starting_weights) and torch.equal(model.generator.get_state(),starting_rng)
    print(json.dumps(dict(done=True,attempts=model.iter,**save())),flush=True)


if __name__=='__main__':main()
