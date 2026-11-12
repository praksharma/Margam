"""Paired old/new CountSketch measurements and full-continuation equivalence.

The dense reference is Margam's implementation before this optimization, retained
locally for comparison. It is not upstream code. Each case/batch runs in a fresh
process; both branches start from the same post-warmup checkpoint.
"""
import argparse
import gc
import hashlib,json,math
from pathlib import Path
import subprocess,sys,time
from unittest.mock import patch
import torch
from examples.dsgnar_analytic import build
from examples.profile_dsgnar import assert_same_state


def dense_count_add(matrix,residual,jacobian,values,buckets,signs):
    """Frozen pre-optimization reference: all buckets, including empty ones."""
    for column in range(buckets.shape[0]):
        order=torch.argsort(buckets[column],stable=True)
        counts=torch.bincount(buckets[column],minlength=len(residual))
        factor=signs[column,order]/math.sqrt(buckets.shape[0])
        matrix+=torch.segment_reduce(jacobian[order]*factor[:,None],'sum',lengths=counts)
        residual+=torch.segment_reduce(values[order]*factor,'sum',lengths=counts)


def paired(case,device,batch,attempts=2,order=('dense','candidate'),experimental=False):
    from margam.methods import dsgnar
    from margam.execution.codec import cpu_tree
    torch.set_num_threads(1);options=dict(network='siren' if case=='wave' else 'fourier',width=16 if case=='wave' else 24,
        reweight=0. if case=='wave' else .05,points=2048,boundary=512,batch=batch,device=device)
    optimized=dsgnar._count_add_touched if experimental else dsgnar.count_add
    def sync():
        if device=='cuda':torch.cuda.synchronize()
    def clear():
        gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    warm=build(case,**options)
    with patch.object(dsgnar,'count_add',dense_count_add):warm.train(1,0)
    snapshot=cpu_tree(warm.training_state());del warm;clear();states={};rows={}
    for name in order:
        model=build(case,**options);model.load_training_state(snapshot)
        sync()
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        with patch.object(dsgnar,'count_add',dense_count_add if name=='dense' else optimized):
            started=time.perf_counter();model.train(attempts,0);sync();elapsed=time.perf_counter()-started
        if model.iter!=snapshot['iter']+attempts:raise RuntimeError('Incomplete comparison attempts.')
        memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()) if device=='cuda' else {}
        rows[name]=dict(seconds_per_attempt=elapsed/attempts,memory=memory)
        states[name]=cpu_tree(model.training_state());del model;clear()
    assert_same_state(states['dense'],states['candidate'])
    return dict(case=case,device=device,batch_size=batch,precision='float64',measured_attempts=attempts,order=list(order),
        branches=rows,reference_over_candidate_time_ratio=rows['dense']['seconds_per_attempt']/rows['candidate']['seconds_per_attempt'],
        full_checkpoint_parity=True,candidate_mode='experimental-touched' if experimental else 'production-dispatch',regularization_diagnostic_rtol=5e-14,
        hardware=torch.cuda.get_device_name() if device=='cuda' else 'CPU')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--batches',nargs='+',type=int,default=[16,64,128,256]);p.add_argument('--attempts',type=int,default=2)
    p.add_argument('--output',type=Path);p.add_argument('--worker',choices=('wave','multiscale'),help=argparse.SUPPRESS)
    p.add_argument('--experimental-touched',action='store_true',help='Measure occupied-row CUDA prototype instead of the production dispatch.')
    p.add_argument('--reverse',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    if not 1<=len(a.batches)<=8 or len(a.batches)!=len(set(a.batches)) or any(not 1<=b<=256 for b in a.batches) or not 1<=a.attempts<=10:p.error('Invalid bounded comparison budget.')
    if a.worker:
        if len(a.batches)!=1:p.error('Worker requires one batch.')
        print(json.dumps(paired(a.worker,a.device,a.batches[0],a.attempts,order=('candidate','dense') if a.reverse else ('dense','candidate'),experimental=a.experimental_touched),allow_nan=False));return
    if a.output is None:p.error('--output is required.')
    rows=[];root=Path(__file__).resolve().parents[1]
    for case in ('wave','multiscale'):
        for index,batch in enumerate(a.batches):
            args=[sys.executable,'-m','examples.compare_dsgnar_sketch','--worker',case,'--device',a.device,'--batches',str(batch),'--attempts',str(a.attempts)]
            if index%2:args.append('--reverse')
            if a.experimental_touched:args.append('--experimental-touched')
            result=subprocess.run(args,cwd=root,capture_output=True,text=True,timeout=300)
            if result.returncode:raise RuntimeError('Sketch comparison failed; no successful report published.\n'+result.stderr)
            rows.append(json.loads(result.stdout));print(f'{case} batch {batch}: {rows[-1]["reference_over_candidate_time_ratio"]:.3f}×',flush=True)
    from examples.validate_cuda import source_identity
    source=source_identity(root);source['comparison_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report=dict(format='margam-countsketch-comparison',version=1,source=source,cases=rows,
        scope='Short paired full-attempt comparison of the specified candidate on fixed cases; not paper accuracy or production throughput.',
        method='One common dense warm-up; both branches reload its checkpoint, garbage-collect previous models and release allocator caches. Branch order alternates across batch settings.',
        limitations=['Only two measured attempts per branch by default; elapsed ratios are noisy.',
                    'CUDA peaks exclude driver allocations and are not a fit guarantee.',
                    'Full state parity is checked within each setting; changing batch still changes the realized sketch.'])
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.partial');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(a.output)


if __name__=='__main__':main()
