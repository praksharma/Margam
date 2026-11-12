"""Independent wave/multiscale Poisson DSGNAR experiments with exact references.

Paper Appendix C.3/C.5 equations; portable single-window/smaller-budget runs.
Uses Margam's existing SIREN/Fourier networks and shared DSGNAR update/checkpoint code.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from margam.research import DSGNARPINN,PINNProblem,PointConstraint,derivative
from margam.components.dsgnar_config import dsgnar_defaults


def reference(case,x):
    a,b=x[...,0:1],x[...,1:2]
    sin,cos=(torch.sin,torch.cos) if isinstance(x,torch.Tensor) else (np.sin,np.cos)
    if case=='wave':return sin(math.pi*a)*cos(2*math.pi*b)+.5*sin(4*math.pi*a)*cos(8*math.pi*b)
    return sum(sin(w*math.pi*a)*sin(w*math.pi*b) for w in (4,8,12,16))/4


def forcing(x):return sum(.5*(w*math.pi)**2*torch.sin(w*math.pi*x[:,0:1])*torch.sin(w*math.pi*x[:,1:2]) for w in (4,8,12,16))


def make_problem(case,points=2048,boundary=512,seed=0):
    from examples.dsgnar_cavity import sample_points
    interior,wall=sample_points(points,boundary,seed,dtype=torch.float64)
    if case=='multiscale':
        return PINNProblem(lambda x,u:-derivative(u,x,0,2)-derivative(u,x,1,2),interior,
            [PointConstraint('boundary',wall)],residual_target=forcing,data_identity='multiscale-Poisson-C5-modes-4-8-12-16-v1')
    if case!='wave':raise ValueError('Unknown analytical case.')
    initial=torch.column_stack((torch.linspace(0,1,boundary,dtype=torch.float64),torch.zeros(boundary,dtype=torch.float64)))
    # Two endpoint traces only; no final-time condition.
    generator=torch.Generator().manual_seed(seed+1);wall=torch.rand(boundary,2,generator=generator,dtype=torch.float64);wall[:,0]=torch.arange(boundary)%2
    return PINNProblem(lambda x,u:derivative(u,x,1,2)-4*derivative(u,x,0,2),interior,[
        PointConstraint('initial',initial,lambda x:torch.sin(math.pi*x[:,0:1])+.5*torch.sin(4*math.pi*x[:,0:1])),
        PointConstraint('initial_velocity',initial,0.,lambda x,u:derivative(u,x,1)),
        PointConstraint('boundary',wall)],data_identity='standing-wave-C3-c2-two-modes-v1')


class FunctionalAnalyticTrainer(DSGNARPINN):
    def compile(self,*args,**kwargs):
        super().compile(*args,**kwargs)
        from torch.func import functional_call,jacrev,vmap
        case=self.configuration['case']
        def value(parameters,point):return functional_call(self.network,parameters,point[None,:]).reshape(())
        gradient=jacrev(value,argnums=1);hessian=jacrev(gradient,argnums=1)
        def pde(parameters,point):
            h=hessian(parameters,point)
            return h[1,1]-4*h[0,0] if case=='wave' else -h[0,0]-h[1,1]
        def velocity(parameters,point):return gradient(parameters,point)[1]
        self.functional_values={};self.functional_blocks={}
        for name in self.names:
            function=pde if name=='pde' else velocity if name=='initial_velocity' else value
            def auxiliary(parameters,point,fn=function):
                result=fn(parameters,point);return result,result
            self.functional_values[name]=vmap(function,in_dims=(None,0))
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


def build(case,*,width=24,depth=3,points=2048,boundary=512,seed=0,sketch=700,batch=64,probes=16,reweight=.05,network='fourier',device='cpu'):
    from margam.templates.examples import example_config
    from margam.nn.factory import build_network
    c=example_config('dsgnar_'+case);c.layers=tuple([width]*depth);c.seed=c.network_seed=seed;c.device=device
    if network=='siren':c.network_type='siren';c.network_options={'first_omega':8.,'omega':1.}
    elif network=='fourier':
        c.network_type='fourier';c.network_options={'frequencies':(.5,1.,2.,4.) if case=='wave' else (2.,4.,6.,8.)}
    else:raise ValueError('Unknown experiment architecture.')
    settings=dsgnar_defaults();settings.update(sketch_size=sketch,batch_size=batch,probes=probes,seed=seed,stage_grace=80,reweight_exponent=reweight)
    network=build_network(c)
    problem=make_problem(case,points,boundary,seed)
    weights={'pde':.01,**{condition.name:1. for condition in problem.constraints}}
    configuration=dict(case=case,version=1,width=width,depth=depth,seed=seed,network_spec=network._margam_spec)
    return FunctionalAnalyticTrainer(problem,network,settings=settings,weights=weights,configuration=configuration).compile(device=device)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('case',choices=('wave','multiscale'))
    p.add_argument('--network',choices=('fourier','siren'))
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--width',type=int);p.add_argument('--depth',type=int,default=3)
    p.add_argument('--points',type=int,default=2048);p.add_argument('--boundary',type=int,default=512);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--sketch',type=int,default=700);p.add_argument('--batch',type=int,default=64);p.add_argument('--probes',type=int,default=16)
    p.add_argument('--reweight-exponent',type=float)
    p.add_argument('--attempts',type=int,default=180);p.add_argument('--seconds',type=float,default=1800)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--resume',type=Path);p.add_argument('--evaluate-only',action='store_true');p.add_argument('--csv',type=Path)
    a=p.parse_args()
    a.network=a.network or ('siren' if a.case=='wave' else 'fourier')
    if a.width is None:a.width=16 if a.case=='wave' else 24
    if a.reweight_exponent is None:a.reweight_exponent=0. if a.case=='wave' else .05
    if a.evaluate_only and not a.resume:p.error('--evaluate-only requires --resume.')
    if not 2<=a.width<=128 or not 1<=a.depth<=8 or not 4<=a.points<=100000 or not 4<=a.boundary<=32768 or not 1<=a.attempts<=10000 or not math.isfinite(a.seconds) or a.seconds<=0:p.error('Invalid architecture, points or run budget.')
    torch.set_num_threads(1)
    model=build(a.case,width=a.width,depth=a.depth,points=a.points,boundary=a.boundary,seed=a.seed,sketch=a.sketch,batch=a.batch,probes=a.probes,reweight=a.reweight_exponent,network=a.network,device=a.device)
    if a.resume:
        from margam.execution.codec import load_packet
        model.load_training_state(load_packet(a.resume))
    from examples.validate_cuda import source_identity
    source=source_identity(Path(__file__).resolve().parents[1]);source['benchmark_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    grid=np.linspace(0,1,129);xx,yy=np.meshgrid(grid,grid);evaluation=np.column_stack((xx.ravel(),yy.ravel()));exact=reference(a.case,evaluation).ravel()
    initial_weights=model._flat().clone();initial_rng=model.generator.get_state().clone();starting=model.iter;history=[];started=time.perf_counter();initial=model._losses()
    def save():
        prediction=model.predict(evaluation).detach().cpu().numpy().ravel();error=float(np.linalg.norm(prediction-exact)/np.linalg.norm(exact))
        record=dict(case=a.case,paper='https://arxiv.org/abs/2607.02194',source=source,scope='Exact paper equations; smaller pointwise network, fixed samples and single wave time window. Not a full paper protocol or speed reproduction.',
            precision='float64',device=a.device,hardware=torch.cuda.get_device_name() if a.device=='cuda' else 'CPU',parameters=sum(p.numel() for p in model.parameters),
            configuration=model.configuration,points=a.points,boundary=a.boundary,settings=model.settings,attempts=model.iter,starting_attempt=starting,seconds=time.perf_counter()-started,
            initial_losses=initial,final_losses=model._losses(),final_condition_weights=model.weights,relative_l2=error,evaluation_grid=[129,129],reference='Analytical standing wave' if a.case=='wave' else 'Analytical four-mode Poisson',
            inference_only=a.evaluate_only,inference_preserved_weights_iteration_rng=True if a.evaluate_only else None,checkpoint_sha256=hashlib.sha256(a.resume.read_bytes()).hexdigest() if a.resume else None,history=history)
        a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.partial');tmp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');tmp.replace(a.output)
        if a.checkpoint:
            from margam.execution.codec import save_packet,cpu_tree
            a.checkpoint.parent.mkdir(parents=True,exist_ok=True);save_packet(a.checkpoint,cpu_tree(model.training_state()))
        if a.csv:
            a.csv.parent.mkdir(parents=True,exist_ok=True);np.savetxt(a.csv,np.column_stack((evaluation,prediction,exact)),delimiter=',',header='x,t,u,reference' if a.case=='wave' else 'x,y,u,reference',comments='')
        return error
    print(json.dumps(dict(start=True,case=a.case,parameters=sum(p.numel() for p in model.parameters),initial_losses=initial)),flush=True)
    for _ in range(0 if a.evaluate_only else a.attempts):
        model.train(1,0);row=dict(attempt=model.iter,seconds=time.perf_counter()-started,losses=model._losses(),radius=model.radius,accepted=model.diagnostics[-1]['accepted']);history.append(row)
        if model.iter%5==0 or model.converged:row['relative_l2']=save()
        print(json.dumps(row),flush=True)
        if model.converged or time.perf_counter()-started>a.seconds:break
    if a.evaluate_only:assert starting==model.iter and torch.equal(initial_weights,model._flat()) and torch.equal(initial_rng,model.generator.get_state())
    print(json.dumps(dict(done=True,attempts=model.iter,relative_l2=save())),flush=True)


if __name__=='__main__':main()
