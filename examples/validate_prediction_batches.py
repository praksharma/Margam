"""Measure real pointwise inference memory; no paper-accuracy claim."""
import argparse
from copy import deepcopy
import gc,hashlib,json,time,platform
from pathlib import Path
import torch
from margam.components.prediction import prediction_model,predict_in_batches,NATIVE_PREDICTION_BATCH
from margam.templates.poisson import PoissonConfig,build_pinn


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--rows',type=int,default=50000);p.add_argument('--batch',type=int,default=NATIVE_PREDICTION_BATCH);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not 2049<=a.rows<=100000 or not 1<=a.batch<=2048:p.error('Use 2049–100000 rows and batch 1–2048.')
    if a.output.exists():p.error('Choose a fresh output path; existing evidence is not overwritten.')
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    c=PoissonConfig(device=a.device,points=4,boundary_points=2,layers=(128,)*6,network_seed=7)
    trained=build_pinn(c);trained.train(1,0)
    from margam.execution.codec import cpu_tree
    checkpoint=cpu_tree(trained.training_state());del trained;gc.collect()
    model=prediction_model(c,checkpoint)
    g=torch.Generator().manual_seed(19);points=torch.rand(a.rows,2,generator=g,dtype=torch.float64)
    before=deepcopy(model.training_state());cpu_rng=torch.get_rng_state().clone()
    gpu_rng=torch.cuda.get_rng_state().clone() if a.device=='cuda' else None
    def sync():
        if a.device=='cuda':torch.cuda.synchronize()
    def reset():
        gc.collect();sync()
        if a.device=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    def memory():
        return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()) if a.device=='cuda' else {}
    model.predict(points[:a.batch]);sync();reset()
    start=time.perf_counter();full=model.predict(points);sync();one_seconds=time.perf_counter()-start;one_memory=memory()
    reference=full.detach().cpu();del full;reset();seen=[]
    hook=model.network.register_forward_pre_hook(lambda net,args:seen.append(len(args[0])))
    try:
        start=time.perf_counter();batched=predict_in_batches(model,points,a.batch);sync();batch_seconds=time.perf_counter()-start;batch_memory=memory()
    finally:hook.remove()
    torch.testing.assert_close(batched,reference,rtol=1e-11,atol=1e-12)
    assert max(seen)<=a.batch and sum(seen)==a.rows and len(seen)==(a.rows+a.batch-1)//a.batch
    for k,v in before['network'].items():torch.testing.assert_close(v,model.training_state()['network'][k],rtol=0,atol=0)
    assert model.iter==before['iter'] and torch.equal(cpu_rng,torch.get_rng_state())
    if gpu_rng is not None:assert torch.equal(gpu_rng,torch.cuda.get_rng_state())
    if a.device=='cuda':assert batch_memory['peak_allocated_bytes']<one_memory['peak_allocated_bytes']
    from examples.validate_cuda import source_identity
    root=Path(__file__).resolve().parents[1];source=source_identity(root);source['validator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report=dict(scope='Known pointwise network, 50000-row-scale inference; not training memory, a fit guarantee or paper accuracy.',
        source=source,python=platform.python_version(),torch=torch.__version__,compiled_cuda=torch.version.cuda,device=a.device,hardware=torch.cuda.get_device_name() if a.device=='cuda' else 'CPU',rows=a.rows,batch=a.batch,precision='float64',layers=[128]*6,
        parameters=sum(t.numel() for t in model.network.parameters()),one_shot=dict(seconds=one_seconds,**one_memory),batched=dict(seconds=batch_seconds,**batch_memory),
        forward_calls=len(seen),maximum_forward_rows=max(seen),cpu_output=True,checkpoint_and_rng_preserved=True,max_absolute_difference=(batched-reference).abs().max().item(),
        limits='Allocator peaks exclude driver memory. Both time and peak figures are short measurements. CPU assembly/output/IPC memory still scales with total rows. Different GEMM batch shapes allow floating-point roundoff.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
