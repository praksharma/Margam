"""Opt-in isolated DSGNAR phase/batch profiling, with continuation parity checks.

No GUI model, saved run or dependency is changed. Each case/batch has a fresh
process, one warm-up attempt and bounded measured attempts. Timings synchronize
CUDA; overlapping Jacobian/sketch durations must not be summed twice.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import torch
from examples.dsgnar_analytic import build


class PhaseProfiler:
    """Scoped instrumentation for a standalone worker, never a live GUI thread."""
    def __init__(self,model,svd_memory=False):
        self.model=model;self.phases={};self.stack=ExitStack();self.svd_memory=svd_memory
        self.svd_memory_calls=[];self.overall_allocated_peak=0;self.overall_reserved_peak=0
    def synchronize(self):
        if self.model.device.type=='cuda':torch.cuda.synchronize(self.model.device)
    def add(self,name,elapsed):
        row=self.phases.setdefault(name,dict(seconds=0.,calls=0));row['seconds']+=elapsed;row['calls']+=1
    def measured(self,name,function):
        def call(*args,**kwargs):
            self.synchronize();started=time.perf_counter()
            try:return function(*args,**kwargs)
            finally:self.synchronize();self.add(name,time.perf_counter()-started)
        return call
    def measured_svd(self,function):
        timed=self.measured('svd',function)
        def call(*args,**kwargs):
            if not self.svd_memory or self.model.device.type!='cuda':return timed(*args,**kwargs)
            self.synchronize();device=self.model.device
            self.overall_allocated_peak=max(self.overall_allocated_peak,torch.cuda.max_memory_allocated(device))
            self.overall_reserved_peak=max(self.overall_reserved_peak,torch.cuda.max_memory_reserved(device))
            before=torch.cuda.memory_allocated(device);torch.cuda.reset_peak_memory_stats(device)
            try:return timed(*args,**kwargs)
            finally:
                self.synchronize();peak=torch.cuda.max_memory_allocated(device);reserved=torch.cuda.max_memory_reserved(device)
                self.svd_memory_calls.append(dict(solver_device=str(args[0].device),allocated_before_bytes=before,peak_allocated_bytes=peak,
                    allocated_growth_bytes=peak-before,peak_reserved_bytes=reserved))
                self.overall_allocated_peak=max(self.overall_allocated_peak,peak)
                self.overall_reserved_peak=max(self.overall_reserved_peak,reserved)
        return call
    def __enter__(self):
        from margam.methods import dsgnar
        original=self.model._jacobian_batches
        def blocks(*args,**kwargs):
            iterator=iter(original(*args,**kwargs))
            while True:
                self.synchronize();started=time.perf_counter()
                try:result=next(iterator)
                except StopIteration:return
                finally:self.synchronize()
                self.add('jacobian_blocks',time.perf_counter()-started)
                yield result
        try:
            for name,function in [('_sketch',self.measured('jacobian_and_sketch',self.model._sketch)),
                                  ('_losses',self.measured('full_objective',self.model._losses)),('_jacobian_batches',blocks)]:
                self.stack.enter_context(patch.object(self.model,name,function))
            for owner,name,phase in [(torch.linalg,'svd','svd'),(dsgnar,'lm_step','regularized_steps'),(dsgnar,'choose_radius','radius_interpolation')]:
                self.stack.enter_context(patch.object(owner,name,self.measured_svd(getattr(owner,name)) if name=='svd' else self.measured(phase,getattr(owner,name))))
            return self
        except BaseException:self.stack.close();raise
    def __exit__(self,*exception):return self.stack.__exit__(*exception)


def assert_same_state(left,right):
    """Exact trained state; existing last-bit regularization diagnostic tolerance."""
    a,b=deepcopy(left),deepcopy(right)
    torch.testing.assert_close(torch.tensor(a['dsgnar'].pop('lambda_history'),dtype=torch.float64),torch.tensor(b['dsgnar'].pop('lambda_history'),dtype=torch.float64),rtol=5e-14,atol=0.)
    for x,y in zip(a['dsgnar']['diagnostics'],b['dsgnar']['diagnostics']):
        torch.testing.assert_close(x.pop('regularization'),y.pop('regularization'),rtol=5e-14,atol=0.)
    def equal(x,y):
        if isinstance(x,torch.Tensor):torch.testing.assert_close(x,y,rtol=0,atol=0)
        elif isinstance(x,dict):
            if x.keys()!=y.keys():raise AssertionError('Checkpoint fields differ.')
            for key in x:equal(x[key],y[key])
        elif isinstance(x,(list,tuple)):
            if type(x) is not type(y) or len(x)!=len(y):raise AssertionError('Checkpoint sequences differ.')
            for p,q in zip(x,y):equal(p,q)
        elif x!=y:raise AssertionError('Checkpoint values differ.')
    equal(a,b)


def process_rss():
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
    except ImportError:return None


def worker(case,device,batch,attempts=2,points=2048,boundary=512,*,width=None,depth=3,sketch=700,precision='float64',svd_memory=False):
    from margam.execution.codec import cpu_tree
    network='siren' if case=='wave' else 'fourier';width=width if width is not None else 16 if case=='wave' else 24
    options=dict(network=network,width=width,depth=depth,sketch=sketch,precision=precision,points=points,boundary=boundary,batch=batch,device=device,reweight=0. if case=='wave' else .05)
    torch.set_num_threads(1);model=build(case,**options);model.train(1,0)
    snapshot=cpu_tree(model.training_state());starting=model.iter
    cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state(model.device).clone() if model.device.type=='cuda' else None
    memory={}
    if model.device.type=='cuda':
        torch.cuda.synchronize(model.device);torch.cuda.reset_peak_memory_stats(model.device)
        memory['resident_allocated_bytes_before']=torch.cuda.memory_allocated(model.device)
    with PhaseProfiler(model,svd_memory=svd_memory) as profiler:
        profiler.synchronize();started=time.perf_counter();model.train(attempts,0);profiler.synchronize();seconds=time.perf_counter()-started
    if model.iter-starting!=attempts:raise RuntimeError('Radius stop prevented completion of the requested profiling attempts.')
    if not torch.equal(cpu_rng,torch.get_rng_state()):raise AssertionError('Profiling changed global CPU RNG.')
    if gpu_rng is not None:
        if not torch.equal(gpu_rng,torch.cuda.get_rng_state(model.device)):raise AssertionError('Profiling changed global CUDA RNG.')
        memory.update(peak_allocated_bytes=max(profiler.overall_allocated_peak,torch.cuda.max_memory_allocated(model.device)),peak_reserved_bytes=max(profiler.overall_reserved_peak,torch.cuda.max_memory_reserved(model.device)))
        if svd_memory:memory['svd_calls']=profiler.svd_memory_calls
    memory['process_peak_rss_bytes_at_end']=process_rss()
    # Reference model is constructed only after memory/timing capture. Its
    # continuation starts at the identical post-warmup data/weights/private RNG.
    after=cpu_tree(model.training_state());control=build(case,**options);control.load_training_state(snapshot)
    profiler.synchronize();control_started=time.perf_counter();control.train(attempts,0);profiler.synchronize();control_seconds=time.perf_counter()-control_started
    assert_same_state(after,cpu_tree(control.training_state()))
    size=min(model.settings['sketch_size'],sum(p.numel() for p in model.parameters))
    rows=sum(values.numel() for c in model._conditions() for values in model._batches(c))
    return dict(case=case,network=network,parameters=sum(p.numel() for p in model.parameters),precision=precision,device=device,width=width,depth=depth,requested_sketch_size=sketch,
        batch_size=batch,points=points,boundary=boundary,scalar_residual_rows=rows,sketch_size=size,warmup_attempts=1,measured_attempts=attempts,
        seconds=seconds,seconds_per_attempt=seconds/attempts,control_seconds_per_attempt=control_seconds/attempts,
        instrumentation_time_ratio=seconds/control_seconds,phases=profiler.phases,memory=memory,
        hypothetical_dense_jacobian_bytes=rows*sum(p.numel() for p in model.parameters)*torch.tensor([],dtype=getattr(torch,precision)).element_size(),sketch_matrix_bytes=size*size*torch.tensor([],dtype=getattr(torch,precision)).element_size(),
        checkpoint_parity=True,global_rng_preserved=True,regularization_diagnostic_rtol=5e-14,
        accepted_measured_attempts=sum(d['accepted'] for d in model.diagnostics[starting:]),final_raw_losses=model._losses(),
        hardware=torch.cuda.get_device_name(model.device) if model.device.type=='cuda' else 'CPU')


def profile_source(root):
    from examples.validate_cuda import source_identity
    source=source_identity(root)
    source['profiler_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source['experiment_sha256']=hashlib.sha256((root/'examples/dsgnar_analytic.py').read_bytes()).hexdigest()
    return source


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--case',choices=('wave','multiscale','both'),default='both');p.add_argument('--batches',type=int,nargs='+',default=[16,64,128,256])
    p.add_argument('--attempts',type=int,default=2);p.add_argument('--points',type=int,default=2048);p.add_argument('--boundary',type=int,default=512)
    p.add_argument('--width',type=int);p.add_argument('--depth',type=int,default=3);p.add_argument('--sketch',type=int,default=700)
    p.add_argument('--precision',choices=('float32','float64'),default='float64');p.add_argument('--svd-memory',action='store_true')
    p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--output',type=Path);p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    if not a.batches or len(a.batches)>8 or len(set(a.batches))!=len(a.batches) or any(not 1<=n<=256 for n in a.batches):p.error('Use 1–8 distinct batch sizes between 1 and 256.')
    if not 1<=a.attempts<=10 or not 4<=a.points<=32768 or not 4<=a.boundary<=16384:p.error('Invalid bounded profiling budget.')
    if a.width is not None and not 2<=a.width<=128 or not 1<=a.depth<=6 or not 1<=a.sketch<=4096 or not 30<=a.timeout<=7200:p.error('Invalid architecture, sketch or timeout.')
    if a.worker:
        if a.case=='both' or len(a.batches)!=1:p.error('Worker requires one case and batch.')
        print(json.dumps(worker(a.case,a.device,a.batches[0],a.attempts,a.points,a.boundary,width=a.width,depth=a.depth,sketch=a.sketch,precision=a.precision,svd_memory=a.svd_memory),allow_nan=False));return
    if a.output is None:p.error('--output is required.')
    rows=[];root=Path(__file__).resolve().parents[1];source=profile_source(root)
    for case in (('wave','multiscale') if a.case=='both' else (a.case,)):
        for batch in a.batches:
            args=[sys.executable,'-m','examples.profile_dsgnar','--worker','--device',a.device,'--case',case,'--batches',str(batch),'--attempts',str(a.attempts),'--points',str(a.points),'--boundary',str(a.boundary)]
            args+=['--depth',str(a.depth),'--sketch',str(a.sketch),'--precision',a.precision]
            if a.width is not None:args+=['--width',str(a.width)]
            if a.svd_memory:args.append('--svd-memory')
            result=subprocess.run(args,cwd=root,capture_output=True,text=True,timeout=a.timeout)
            if result.returncode:raise RuntimeError('Profiling worker failed; no successful report published.\n'+result.stderr)
            rows.append(json.loads(result.stdout));print(f'{case}: batch {batch}, {rows[-1]["seconds_per_attempt"]:.3f} s/attempt',flush=True)
    if profile_source(root)!=source:raise RuntimeError('Source changed during profiling; no successful report published.')
    report=dict(format='margam-dsgnar-profile',version=2,source=source,python=sys.version.split()[0],torch_version=str(torch.__version__),compiled_cuda=torch.version.cuda,fresh_process_per_configuration=True,cases=rows,
        scope='Synchronized instrumented complete attempts after one warm-up; fixed concrete cases with explicitly recorded architecture/sketch/point counts. Not H100/JAX throughput, a hardware quota, or a paper accuracy claim.',
        limitations=['Jacobian-block seconds are included in jacobian-and-sketch seconds; do not sum them twice.',
                    'Synchronization/instrumentation overhead is included; reference total timing is also recorded. Neither is steady-state compiler throughput.',
                    'Batch sizes change the realized random sketch and floating-point grouping; trajectories are not compared across batches.',
                    'CUDA allocated/reserved peaks are allocator measurements after warm-up, not all driver/OS allocations or a future-fit guarantee.',
                    'Optional SVD resets capture phase peaks; an explicit running maximum preserves the complete-attempt peak.',
                    'Process RSS is the lifetime high-water mark through timing, including imports/warm-up; it is not an update-only delta.',
                    'Reference continuation is checked separately for each batch setting; the control model is created after memory capture.'])
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.partial');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(a.output)


if __name__=='__main__':main()
