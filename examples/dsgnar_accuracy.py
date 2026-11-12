"""DSGNAR accuracy experiments with an independently generated Burgers reference.

Paper case: periodic x in [-1,1], t in [0,1], nu=.01, u(x,0)=-sin(pi*x).
This is distinct from Margam's original-PINN Dirichlet Burgers example.
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from margam.research import DSGNARPINN,PINNProblem,PointConstraint,derivative
from margam.components.dsgnar_config import dsgnar_defaults


class PeriodicMLP(torch.nn.Module):
    """Paper B.3 tanh network with one learned scale per hidden layer."""
    def __init__(self,width=16,depth=6,bias_noise=.01):
        super().__init__();self.layers=torch.nn.ModuleList(torch.nn.Linear(a,b) for a,b in zip([3]+[width]*depth,[width]*depth+[1]))
        self.scales=torch.nn.Parameter(torch.ones(depth))
        for layer in self.layers[:-1]:torch.nn.init.orthogonal_(layer.weight,gain=5/3);torch.nn.init.normal_(layer.bias,std=bias_noise) if bias_noise else torch.nn.init.zeros_(layer.bias)
        torch.nn.init.normal_(self.layers[-1].weight,std=1/math.sqrt(width));torch.nn.init.normal_(self.layers[-1].bias,std=bias_noise) if bias_noise else torch.nn.init.zeros_(self.layers[-1].bias)
    def forward(self,points):
        x=points[:,0:1];h=torch.cat((torch.cos(math.pi*x),torch.sin(math.pi*x),points[:,1:2]),dim=1)
        for scale,layer in zip(self.scales,self.layers[:-1]):h=torch.tanh(scale*layer(h))
        return self.layers[-1](h)


class FunctionalBurgersTrainer(DSGNARPINN):
    """Concrete per-point functional derivatives; shares every DSGNAR update."""
    def compile(self,*args,**kwargs):
        super().compile(*args,**kwargs)
        from torch.func import functional_call,jacrev,vmap
        def value(parameters,point):return functional_call(self.network,parameters,point[None,:]).reshape(())
        spatial=jacrev(value,argnums=1)
        second=jacrev(spatial,argnums=1)
        def pde(parameters,point):
            u=value(parameters,point);gradient=spatial(parameters,point)
            return gradient[1]+u*gradient[0]-.01*second(parameters,point)[0,0]
        def jacobian(function):
            def with_value(parameters,point):
                result=function(parameters,point);return result,result
            return vmap(jacrev(with_value,argnums=0,has_aux=True),in_dims=(None,0))
        self.functional_blocks={'pde':jacobian(pde),'initial':jacobian(value)}
        self.functional_values={'pde':vmap(pde,in_dims=(None,0)),'initial':vmap(value,in_dims=(None,0))}
        return self

    def _batches(self,condition,batch_size=None):
        name,points,target,operator=condition
        if operator is not None or name not in self.functional_values:raise ValueError('Functional benchmark conditions must be PDE or initial values.')
        batch_size=self.settings['batch_size'] if batch_size is None else batch_size
        parameters=dict(self.network.named_parameters())
        for start in range(0,len(points),batch_size):
            batch=points[start:start+batch_size]
            with torch.no_grad():values=self.functional_values[name](parameters,batch)
            batch_target=target if target.ndim==0 else target[start:start+len(batch)].reshape(-1)
            yield (values-batch_target).detach()

    def _jacobian_batches(self,condition):
        name,points,target,operator=condition
        if operator is not None or name not in self.functional_blocks:raise ValueError('Functional benchmark path supports its concrete PDE and initial condition only.')
        parameters=dict(self.network.named_parameters())
        for start in range(0,len(points),self.settings['batch_size']):
            batch=points[start:start+self.settings['batch_size']]
            gradients,values=self.functional_blocks[name](parameters,batch)
            jacobian=torch.cat([gradients[key].reshape(len(batch),-1) for key in parameters],dim=1)
            batch_target=target if target.ndim==0 else target[start:start+len(batch)].reshape(-1)
            yield jacobian.detach(),(values-batch_target).detach()


def burgers_reference(points,order=256):
    """Cole–Hopf heat convolution, independently evaluated by Gauss–Hermite.

    Log shifting keeps the periodic heat expectation well scaled. No upstream
    data is loaded. Evaluate in chunks to bound reference workspace.
    """
    from scipy.special import roots_hermite
    nodes,weights=roots_hermite(order);weights=weights/np.sqrt(np.pi)
    result=[]
    for start in range(0,len(points),512):
        rows=np.asarray(points[start:start+512],dtype=np.float64)
        y=rows[:,0,None]+np.sqrt(4*.01*rows[:,1,None])*nodes[None,:]
        exponent=-np.cos(np.pi*y)/(2*np.pi*.01);exponent-=exponent.max(axis=1,keepdims=True)
        mass=np.exp(exponent)*weights
        result.extend((-np.sum(np.sin(np.pi*y)*mass,axis=1)/np.sum(mass,axis=1)).tolist())
    return np.array(result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--precision',choices=('float32','float64'),default='float32')
    parser.add_argument('--points',type=int,default=32768);parser.add_argument('--initial',type=int,default=16384)
    parser.add_argument('--width',type=int,default=16);parser.add_argument('--depth',type=int,default=6);parser.add_argument('--bias-noise',type=float,default=.01)
    parser.add_argument('--sketch',type=int,default=700);parser.add_argument('--batch',type=int,default=128)
    parser.add_argument('--probes',type=int,default=24);parser.add_argument('--attempts',type=int,default=300)
    parser.add_argument('--seconds',type=float,default=1800);parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--resume',type=Path);parser.add_argument('--profile',action='store_true')
    parser.add_argument('--functional',action='store_true',help='Use validated per-point functional Burgers Jacobians.')
    parser.add_argument('--compile-functional',action='store_true');parser.add_argument('--warm-start',type=Path)
    args=parser.parse_args();torch.set_num_threads(1);torch.set_default_dtype(getattr(torch,args.precision));torch.manual_seed(args.seed)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable.')
    generator=torch.Generator().manual_seed(args.seed)
    points=torch.rand((args.points,2),generator=generator);points[:,0]=2*points[:,0]-1
    initial=torch.column_stack((torch.linspace(-1,1,args.initial),torch.zeros(args.initial)))
    def residual(x,u):return derivative(u,x,1)+u*derivative(u,x,0)-.01*derivative(u,x,0,2)
    problem=PINNProblem(residual,points,[PointConstraint('initial',initial,lambda x:-torch.sin(math.pi*x[:,0:1]))])
    settings=dsgnar_defaults();settings.update(sketch_size=args.sketch,batch_size=args.batch,probes=args.probes,precision=args.precision,seed=args.seed,min_radius=1e-4 if args.precision=='float32' else 1e-8)
    model=(FunctionalBurgersTrainer if args.functional else DSGNARPINN)(problem,PeriodicMLP(args.width,args.depth,args.bias_noise),weights=dict(pde=.01,initial=1.),settings=settings,
        configuration=dict(case='periodic-burgers-2607.02194-C1-v4',viscosity=.01,width=args.width,depth=args.depth,seed=args.seed,bias_noise=args.bias_noise,
            derivative_backend='compiled-functional' if args.compile_functional else 'functional' if args.functional else 'generic',
            compiler_torch_version=str(torch.__version__) if args.compile_functional else None)).compile(device=args.device)
    if args.resume:
        from margam.execution.codec import load_packet
        model.load_training_state(load_packet(args.resume))
    if args.warm_start:
        from margam.execution.codec import load_packet
        model.network.load_state_dict(load_packet(args.warm_start)['network'])
    if args.compile_functional:
        if not args.functional:raise ValueError('--compile-functional requires --functional.')
        model.functional_blocks={name:torch.compile(function,fullgraph=True) for name,function in model.functional_blocks.items()}
        model.functional_values={name:torch.compile(function,fullgraph=True) for name,function in model.functional_values.items()}
    starting_attempt=model.iter;phase_seconds={}
    if args.profile:
        def timed(function,name):
            def measured(*a,**kw):
                if args.device=='cuda':torch.cuda.synchronize()
                before=time.perf_counter();result=function(*a,**kw)
                if args.device=='cuda':torch.cuda.synchronize()
                phase_seconds[name]=phase_seconds.get(name,0.)+time.perf_counter()-before
                return result
            return measured
        model._sketch=timed(model._sketch,'sketch')
        model._losses=timed(model._losses,'probe_residuals')
        torch.linalg.svd=timed(torch.linalg.svd,'svd')
    evaluation=np.array([(x,t) for t in np.linspace(0,1,65) for x in np.linspace(-1,1,257)],dtype=np.float64)
    reference=burgers_reference(evaluation,256);checked=burgers_reference(evaluation[::37],512)
    reference_difference=float(np.max(np.abs(checked-reference[::37])))
    def error():
        prediction=model.predict(evaluation).detach().cpu().numpy().reshape(-1)
        return float(np.linalg.norm(prediction-reference)/np.linalg.norm(reference))
    history=[];started=time.perf_counter();initial_error=error()
    def save():
        report=dict(case='Periodic Burgers, paper Appendix C.1',scope='Accuracy attempt; reproduction only if settings and measured error match the stated target.',
            paper='https://arxiv.org/abs/2607.02194',target_relative_l2=4.75e-7,
            target_context='Paper single-precision 16x6 MLP, 1447 parameters, sketch 700, 32768 PDE / 16384 IC rows, H100; 9.8 seconds.',
            warm_start=bool(args.warm_start),resumed=bool(args.resume),
            matches_reported_precision_and_counts=(args.precision=='float32' and args.points==32768 and args.initial==16384 and args.width==16 and args.depth==6 and args.sketch==700 and not args.warm_start),
            device=args.device,hardware=torch.cuda.get_device_name() if args.device=='cuda' else 'CPU',torch_version=str(torch.__version__),
            precision=args.precision,jacobian_backend='compiled-functional-per-point' if args.compile_functional else 'functional-per-point' if args.functional else 'generic-batched-reverse',pde_points=args.points,initial_points=args.initial,parameters=sum(p.numel() for p in model.parameters),
            width=args.width,depth=args.depth,bias_noise=args.bias_noise,settings=settings,attempts=model.iter,starting_attempt=starting_attempt,seconds=time.perf_counter()-started,phase_seconds=phase_seconds,
            initial_relative_l2=initial_error,final_relative_l2=error(),reference='Independent Cole-Hopf Gauss-Hermite, 257 x 65 uniform evaluation grid',
            reference_256_vs_512_max_abs=reference_difference,history=history,summary=model.dsgnar_summary())
        report['target_met']=report['final_relative_l2']<=report['target_relative_l2']
        args.output.parent.mkdir(parents=True,exist_ok=True);temporary=args.output.with_suffix('.partial');temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temporary.replace(args.output)
        if args.checkpoint:
            from margam.execution.codec import save_packet,cpu_tree
            args.checkpoint.parent.mkdir(parents=True,exist_ok=True);save_packet(args.checkpoint,cpu_tree(model.training_state()))
        return report
    print(json.dumps(dict(start=True,parameters=sum(p.numel() for p in model.parameters),initial_error=initial_error,reference_check=reference_difference)),flush=True)
    for _ in range(args.attempts):
        model.train(1,0)
        if model.iter%1==0 or model.converged:
            row=dict(attempt=model.iter,relative_l2=error(),seconds=time.perf_counter()-started,accepted=sum(r['accepted'] for r in model.diagnostics),radius=model.radius)
            history.append(row);print(json.dumps(row),flush=True);save()
        if model.converged or time.perf_counter()-started>args.seconds:break
    report=save();print(json.dumps(dict(done=True,attempts=model.iter,error=report['final_relative_l2'],target_met=report['target_met'])),flush=True)


if __name__=='__main__':main()
