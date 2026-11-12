"""CountSketch accumulation-only benchmark, including 4,000-row sketches.

This never factors the sketch or trains a network. Both output matrices remain
resident during peak measurement; the shared baseline is recorded explicitly.
"""
import argparse
import hashlib,json,statistics,time
from pathlib import Path
import torch
from margam.methods.dsgnar import _count_add_touched as count_add
from examples.compare_dsgnar_sketch import dense_count_add


def measure(device,size,batch,dtype,repeats=3):
    g=torch.Generator().manual_seed(7);torch_dtype=getattr(torch,dtype)
    jac=torch.randn(batch,size,generator=g,dtype=torch_dtype).to(device);values=torch.randn(batch,generator=g,dtype=torch_dtype).to(device)
    buckets=torch.randint(size,(4,batch),generator=g).to(device);buckets[:,:batch//2]=0
    signs=(torch.randint(2,(4,batch),generator=g)*2-1).to(device=device,dtype=torch_dtype)
    a=torch.zeros(size,size,dtype=torch_dtype,device=device);b=torch.zeros_like(a);x=a.new_zeros(size);y=x.clone()
    def sync():
        if device=='cuda':torch.cuda.synchronize()
    rows={name:dict(seconds=[],peak_allocated_bytes=[]) for name in ('dense','touched')}
    # Warm both paths before measuring either.
    dense_count_add(a,x,jac,values,buckets,signs);count_add(b,y,jac,values,buckets,signs);sync()
    torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(x,y,rtol=0,atol=0)
    for repeat in range(repeats):
        for name in (('dense','touched') if repeat%2==0 else ('touched','dense')):
            matrix,vector,function=(a,x,dense_count_add) if name=='dense' else (b,y,count_add)
            matrix.zero_();vector.zero_();sync()
            if device=='cuda':torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter();function(matrix,vector,jac,values,buckets,signs);sync();rows[name]['seconds'].append(time.perf_counter()-start)
            if device=='cuda':rows[name]['peak_allocated_bytes'].append(torch.cuda.max_memory_allocated())
        torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(x,y,rtol=0,atol=0)
    for r in rows.values():r['median_seconds']=statistics.median(r['seconds'])
    return dict(device=device,dtype=dtype,sketch_size=size,batch_size=batch,hashes=4,repeats=repeats,collision_pattern='Half of each hash maps to bucket zero; remainder uniform',exact_matrix_and_residual_parity=True,branches=rows,
        median_time_ratio=rows['dense']['median_seconds']/rows['touched']['median_seconds'],
        shared_resident_output_bytes=(2*size*size+2*size)*torch.tensor([],dtype=torch_dtype).element_size(),
        scope='Accumulation only; two matrices and inputs resident. Matrix zeroing, SRCT, Jacobians and SVD are excluded.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--size',type=int,default=4000);p.add_argument('--batches',nargs='+',type=int,default=[16,256]);p.add_argument('--repeats',type=int,default=3);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not 1<=a.size<=4096 or not 1<=len(a.batches)<=4 or len(a.batches)!=len(set(a.batches)) or any(not 1<=n<=256 for n in a.batches) or not 1<=a.repeats<=10:p.error('Invalid bounded accumulation benchmark.')
    torch.set_num_threads(1);old=torch.get_deterministic_debug_mode()
    try:
        torch.use_deterministic_algorithms(True)
        rows=[measure(a.device,a.size,batch,dtype,a.repeats) for dtype in ('float32','float64') for batch in a.batches]
    finally:torch.set_deterministic_debug_mode(old)
    from examples.validate_cuda import source_identity
    root=Path(__file__).resolve().parents[1];source=source_identity(root);source['microbenchmark_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report=dict(format='margam-countsketch-microbenchmark',version=1,source=source,cases=rows,
        hardware=torch.cuda.get_device_name() if a.device=='cuda' else 'CPU',scope='Accumulation only: occupied rows are production CPU and experimental CUDA; not a paper-sized SVD or end-to-end solve.')
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.partial');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(a.output)
    for row in rows:print(row['dtype'],row['batch_size'],row['median_time_ratio'],flush=True)


if __name__=='__main__':main()
