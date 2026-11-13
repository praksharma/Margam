"""Paper-sized SCALE cavity reproduction attempt, independent PyTorch code.

Uses local upstream CFD coordinates/BC labels and reference velocities. Never
executes upstream code or uses interior velocity labels in the training objective.
Research checkpoints are separate from native GUI model/run archives.
"""
import argparse
from copy import deepcopy
import hashlib,json,math,os,platform,time
from pathlib import Path
import numpy as np
import torch
from margam.research.scale_cavity import ScaleCavityNetwork,cavity_terms


def protocol(reynolds,seed=None):
    if reynolds not in (3200,7500):raise ValueError('Exact local notebook protocols are available for Re=3200 and Re=7500.')
    return dict(reynolds=reynolds,width=32 if reynolds==3200 else 64,batch=400 if reynolds==3200 else 1000,boundary_draws=50,
        tau_sc=.06 if reynolds==3200 else .095,tau_alpha=1. if reynolds==3200 else .5,bc_weight=15. if reynolds==3200 else 10.,
        lr=.0005,end_lr=1e-10,cosine_exponent=.8 if reynolds==3200 else 1.2,schedule_steps=50000,seed=(50 if reynolds==3200 else 20) if seed is None else seed)


def load_reference(directory,reynolds):
    name='LDC_RE3200_100x100_subsample200x200.csv' if reynolds==3200 else 'LDC_RE7500_150x150_CELL_sub256x256.csv'
    path=Path(directory)/name;table=np.genfromtxt(path,delimiter=',',names=True)
    if table.dtype.names!=('x','y','u','v','p'):raise ValueError('Unexpected cavity reference columns.')
    data=np.column_stack([table[k] for k in table.dtype.names]);points=data[:,:2]
    if data.ndim!=2 or not np.isfinite(data).all() or len(np.unique(points,axis=0))!=len(points):raise ValueError('Reference needs finite unique coordinates/fields.')
    if not np.array_equal(points.min(0),[0.,0.]) or not np.array_equal(points.max(0),[1.,1.]):raise ValueError('Expected a unit-square CFD reference.')
    wall=((points==0)|(points==1)).any(1);corner=((points==0)|(points==1)).all(1)
    if (wall&~corner).sum()<50:raise ValueError('Insufficient boundary points.')
    return data,wall,corner,dict(file=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(data))


def learning_rate(step,settings):
    fraction=min(step/settings['schedule_steps'],1.)
    return settings['end_lr']+(settings['lr']-settings['end_lr'])*(.5*(1+math.cos(math.pi*fraction)))**settings['cosine_exponent']


def evaluate(network,data,wall,corner):
    outputs=[];dtype=next(network.parameters()).dtype;device=next(network.parameters()).device
    with torch.no_grad():
        for start in range(0,len(data),2048):outputs.append(network(torch.as_tensor(data[start:start+2048,:2],dtype=dtype,device=device)).cpu().numpy())
    prediction=np.concatenate(outputs);target=data[:,2:4]
    if not np.isfinite(prediction).all():raise FloatingPointError('Nonfinite predictions; no valid checkpoint published.')
    def metric(mask,values=prediction[:,:2]):
        error=values[mask]-target[mask];den=np.linalg.norm(target[mask])
        return dict(rows=int(mask.sum()),velocity_mse=float(np.mean(error**2)),velocity_relative_l2=float(np.linalg.norm(error)/den),velocity_max_absolute_error=float(abs(error).max()))
    replaced=prediction[:,:2].copy();replaced[wall]=target[wall]
    return dict(raw_without_corners=metric(~corner),raw_interior=metric(~wall),raw_complete=metric(np.ones(len(data),dtype=bool)),boundary_replaced_for_comparison_only=metric(~corner,replaced)),prediction


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-directory',type=Path,default=Path.cwd().parent/'SCALE-PINN'/'ns-ldc')
    p.add_argument('--reynolds',type=int,choices=(3200,7500),default=3200);p.add_argument('--seed',type=int)
    p.add_argument('--updates',type=int,default=50000);p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--eager',action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path);p.add_argument('--report-every',type=int,default=1000)
    a=p.parse_args();settings=protocol(a.reynolds,a.seed)
    if not 1<=a.updates<=settings['schedule_steps'] or a.report_every<1:p.error('Use 1–50000 updates and a positive report interval.')
    if a.output.exists() and not a.resume:p.error('Choose a fresh output directory, or explicitly resume.')
    if a.device=='cpu' and not a.eager:p.error('CPU mode currently requires --eager; compiled measurements use CUDA.')
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','1');torch.set_num_threads(1);torch.set_default_dtype(torch.float32)
    torch.manual_seed(settings['seed']);data,wall,corner,reference=load_reference(a.reference_directory,a.reynolds)
    a.output.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    device=torch.device(a.device);network=ScaleCavityNetwork(settings['width']).to(device);previous=deepcopy(network).requires_grad_(False)
    lr=torch.tensor(settings['lr'],device=device);optimizer=torch.optim.Adam(network.parameters(),lr=lr,fused=a.device=='cuda')
    sample=data[~corner];pool_wall=wall[~corner];xy=torch.as_tensor(sample[:,:2],dtype=torch.float32,device=device)
    flags=torch.as_tensor(pool_wall,device=device);target=torch.as_tensor(np.where(pool_wall[:,None],sample[:,2:4],0.),dtype=torch.float32,device=device)
    # All-domain draws intentionally include some wall points, matching the local
    # notebook. Those rows join BC loss and are masked out of PDE losses.
    all_count=len(xy);all_points=torch.cat((xy,xy[flags]));all_flags=torch.cat((flags,flags[flags]));all_targets=torch.cat((target,target[flags]))
    generator=torch.Generator(device=device).manual_seed(settings['seed']+104729)
    signature=dict(protocol=settings,reference=reference,precision='float32',derivatives='explicit-coordinate-jets',optimizer='torch-adam',compiled=not a.eager)
    step=0;history=[];training_seconds=0.;compilation_seconds=0.
    from margam.execution.codec import cpu_tree
    from examples.validate_cuda import source_identity
    root=Path(__file__).resolve().parents[1];source=source_identity(root);script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    signature.update(torch=str(torch.__version__),device=a.device,network_sha256=hashlib.sha256((root/'margam/research/scale_cavity.py').read_bytes()).hexdigest())
    if a.resume:
        if a.resume.stat().st_size>64*1024*1024:raise ValueError('Research checkpoint exceeds 64 MiB.')
        saved=torch.load(a.resume,map_location='cpu',weights_only=True)
        if saved.get('format')!='margam-scale-cavity-research' or saved.get('version')!=2 or saved.get('signature')!=signature:raise ValueError('Checkpoint protocol/data/backend differs.')
        network.load_state_dict(saved['network']);previous.load_state_dict(saved['previous']);optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['sampler_rng']);step=saved['step'];history=saved['history'];training_seconds=saved['training_seconds']
        lr=optimizer.param_groups[0]['lr'].to(device=device);optimizer.param_groups[0]['lr']=lr
        if len(history)!=step or not 0<=step<a.updates:raise ValueError('Checkpoint step/history does not match the requested continuation.')
    boundary_count=int(flags.sum())
    current_parameters=list(network.parameters());old_parameters=list(previous.parameters())
    def sample_batch():
        indices=torch.cat((torch.randint(all_count,(settings['batch']-50,),generator=generator,device=device),all_count+torch.randint(boundary_count,(50,),generator=generator,device=device)))
        return all_points[indices],all_flags[indices],all_targets[indices]
    def objective(points,boundary,targets):
        corrected,u_bc,v_bc,raw,correction=cavity_terms(network,previous,points,boundary,targets,reynolds=a.reynolds,tau_sc=settings['tau_sc'],tau_alpha=settings['tau_alpha'])
        loss=corrected+settings['bc_weight']*(u_bc+v_bc)
        return loss,torch.stack((loss,corrected,raw,u_bc,v_bc,correction))
    function=objective if a.eager else torch.compile(objective,fullgraph=True)
    # Compile forward/backward without taking an optimizer update or consuming
    # the training sampler's RNG. Compilation is timed separately and included
    # in wall total; no hidden pretraining or reference-guided warm start.
    rng=generator.get_state();batch=sample_batch();cold=time.perf_counter();loss,_=function(*batch);loss.backward();optimizer.zero_grad(set_to_none=True);generator.set_state(rng)
    if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    compilation_seconds=time.perf_counter()-cold
    print(json.dumps(dict(event='ready',settings=settings,parameters=sum(v.numel() for v in current_parameters),compile_and_first_backward_seconds=compilation_seconds)),flush=True)
    block=[];block_start=time.perf_counter();last_metrics=None
    while step<a.updates:
        optimizer.zero_grad(set_to_none=True);lr.fill_(learning_rate(step,settings));loss,stats=function(*sample_batch());loss.backward()
        with torch.no_grad():torch._foreach_copy_(old_parameters,current_parameters)
        optimizer.step();step+=1;block.append(stats.detach())
        if step%a.report_every==0 or step==a.updates:
            if device.type=='cuda':torch.cuda.synchronize()
            training_seconds+=time.perf_counter()-block_start
            rows=torch.stack(block).cpu().double().numpy();block.clear()
            if not np.isfinite(rows).all():raise FloatingPointError('Nonfinite benchmark losses; no valid checkpoint published.')
            history.extend(rows.tolist());last_metrics,prediction=evaluate(network,data,wall,corner)
            state=dict(format='margam-scale-cavity-research',version=2,signature=signature,step=step,network=cpu_tree(network.state_dict()),previous=cpu_tree(previous.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),sampler_rng=generator.get_state(),history=history,training_seconds=training_seconds)
            def finite(item):
                if isinstance(item,torch.Tensor):return bool(torch.isfinite(item).all())
                if isinstance(item,dict):return all(finite(v) for v in item.values())
                if isinstance(item,(list,tuple)):return all(finite(v) for v in item)
                return True
            if not finite((state['network'],state['previous'],state['optimizer'])):raise FloatingPointError('Nonfinite optimizer/model state; checkpoint not published.')
            current_source=source_identity(root)
            if current_source['runtime_and_test_sha256']!=source['runtime_and_test_sha256'] or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=script_hash:
                raise ValueError('Benchmark source changed during training; checkpoint not published.')
            temporary=a.output/'checkpoint.tmp';torch.save(state,temporary);temporary.replace(a.output/'checkpoint.pt')
            report=dict(scope='Independent PyTorch Scale-PINN cavity reproduction attempt; research protocol, not a native GUI checkpoint.',settings=settings,reference=reference,steps=step,parameters=sum(v.numel() for v in current_parameters),metrics=last_metrics,
                training_seconds=training_seconds,compile_and_first_backward_seconds=compilation_seconds,wall_seconds=time.perf_counter()-started,python=platform.python_version(),torch=torch.__version__,hardware=torch.cuda.get_device_name() if device.type=='cuda' else 'CPU',device=a.device,
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,notes=['Boundary velocity labels only; interior CFD values are evaluation-only.','No pressure gauge, matching the research notebook; velocity metrics are pressure-gauge independent.','Different PyTorch/JAX RNG streams; no bitwise upstream initialization claim.','PDE uses interior rows only for this research protocol; native arrow rules are unchanged.','Times are synchronized local measurements; evaluation/checkpoint I/O excluded from training time.'])
            report['source']=source;report['benchmark_sha256']=script_hash
            (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            np.savetxt(a.output/'predictions.csv',np.column_stack((data[:,:2],prediction)),delimiter=',',header='x,y,u,v,p',comments='')
            print(json.dumps(dict(event='progress',step=step,training_seconds=training_seconds,loss=rows[-1,0],**last_metrics['raw_without_corners'])),flush=True)
            block_start=time.perf_counter()
    return 0


if __name__=='__main__':raise SystemExit(main())
