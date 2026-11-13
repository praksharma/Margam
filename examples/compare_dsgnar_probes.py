"""Frozen uncached radius reference and exact CPU probe-selection timing."""
import argparse
import hashlib,json,math,statistics,time
from pathlib import Path
import torch
from margam.methods.dsgnar import pchip_value,choose_radius


def uncached_choose_radius(radii,ratios,target):
    """Margam's pre-cache calculation; every bisection refits unchanged data."""
    safe=torch.cummin(torch.as_tensor(ratios,dtype=torch.float64).clamp_min(0),0).values
    if safe[-1]>=target:return radii[-1]
    if safe[0]<=target:return radii[0]
    knots=torch.tensor([math.log(radius) for radius in radii],dtype=torch.float64)
    low=knots[0].item();high=knots[-1].item()
    for _ in range(60):
        middle=(low+high)/2
        if pchip_value(knots,safe,middle)>=target:low=middle
        else:high=middle
    return math.exp((low+high)/2)


def cases():
    rows=[]
    for n in (3,8,16,24,32):
        rows.append((torch.logspace(-4,3,n,dtype=torch.float64).tolist(),torch.linspace(.95,-.2,n,dtype=torch.float64).tolist(),.075))
    rows += [([1.,2.,4.,8.],[.9,.5,.5,.1],.7),([1.,2.,4.,8.],[.9,.3,-1.,-2.],.075),([1.,2.,4.],[.9,.9,.9],.075),([1.,2.,4.],[0.,0.,0.],.075)]
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repeats',type=int,default=5);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not 1<=a.repeats<=100:p.error('Use 1–100 repetitions.')
    torch.set_num_threads(1);rows=[]
    for radii,ratios,target in cases():
        old=uncached_choose_radius(radii,ratios,target);new=choose_radius(radii,ratios,target)
        if old!=new:raise AssertionError('Radius changed.')
        times={name:[] for name in ('uncached','cached')}
        for repeat in range(a.repeats):
            for name,fn in ([('uncached',uncached_choose_radius),('cached',choose_radius)] if repeat%2==0 else [('cached',choose_radius),('uncached',uncached_choose_radius)]):
                start=time.perf_counter();actual=fn(radii,ratios,target);times[name].append(time.perf_counter()-start)
                if actual!=old:raise AssertionError('Repeated radius changed.')
        rows.append(dict(probes=len(radii),ratios=ratios,target=target,radius=old,exact_radius=True,median_seconds={k:statistics.median(v) for k,v in times.items()},samples=times))
    from examples.validate_cuda import source_identity
    root=Path(__file__).resolve().parents[1];source=source_identity(root);source['probe_benchmark_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report=dict(format='margam-probe-interpolation-comparison',version=1,source=source,cases=rows,scope='CPU radius interpolation only, not SVD, trial residual evaluation or whole-attempt throughput.')
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.partial');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(a.output)


if __name__=='__main__':main()
