"""Curate every completed cavity attempt with independent inference verification."""
import argparse,hashlib,json,platform
from pathlib import Path
import numpy as np
import torch
from examples.plot_scale_cavity import inspect_run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--reference-directory',type=Path,default=Path.cwd().parent/'SCALE-PINN'/'ns-ldc');p.add_argument('--include-best',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Choose a fresh evidence filename.')
    torch.set_num_threads(1);records=[];times=[];names=set()
    for run in a.runs:
        if run.name in names:raise ValueError('Run basenames must be unique.')
        names.add(run.name)
        for best in (False,True) if a.include_best and (run/'best-report.json').exists() else (False,):
            network,data,report,fresh,prediction=inspect_run(run,a.reference_directory,best=best)
            prefix='best-' if best else ''
            state=torch.load(run/(prefix+'checkpoint.pt'),map_location='cpu',weights_only=True)
            initial=report.get('initialization',dict(kind='he_uniform',source_updates=0,source_training_seconds=0.))
            record=dict(run=run.name,checkpoint_kind='reference_selected' if best else 'final',report_sha256=hashlib.sha256((run/(prefix+'report.json')).read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256((run/(prefix+'checkpoint.pt')).read_bytes()).hexdigest(),checkpoint_version=state['version'],
                settings=report['settings'],steps=report['steps'],total_updates=report.get('total_updates',report['steps']),parameters=report['parameters'],reference=report['reference'],metrics=report['metrics'],fresh_cpu_metrics=fresh,
                training_seconds=report['training_seconds'],total_training_seconds=report.get('total_training_seconds',report['training_seconds']),compile_and_first_backward_seconds=report['compile_and_first_backward_seconds'],wall_seconds=report['wall_seconds'],hardware=report['hardware'],torch=report['torch'],initialization=initial,source=report['source'],benchmark_sha256=report['benchmark_sha256'],selection=report.get('selection'))
            candidates=[run/'sources'/(report['benchmark_sha256']+'.py'),run.parent/'sources'/(report['benchmark_sha256']+'.py')]
            snapshot=next((path for path in candidates if path.is_file()),None)
            record['driver_snapshot_verified']=snapshot is not None and hashlib.sha256(snapshot.read_bytes()).hexdigest()==report['benchmark_sha256']
            if best:
                final=json.loads((run/'report.json').read_text());record['selection_search_training_seconds']=final.get('total_training_seconds',final['training_seconds'])
            else:times.append(report['training_seconds'])
            records.append(record)
    summaries=[]
    for re in (3200,7500):
        matching=[r for r in records if r['settings']['reynolds']==re and r['initialization']['kind']=='he_uniform' and r['checkpoint_kind']=='final']
        if matching:
            errors=[r['metrics']['raw_without_corners']['velocity_relative_l2'] for r in matching]
            summaries.append(dict(reynolds=re,runs=len(matching),seeds=[r['settings']['seed'] for r in matching],minimum_relative_l2=min(errors),maximum_relative_l2=max(errors),mean_relative_l2=float(np.mean(errors)),best_final_run=matching[int(np.argmin(errors))]['run']))
    result=dict(format='margam-scale-cavity-attempts',version=1,scope='Independent PyTorch reproduction attempts. Raw velocity metrics on local CFD rows; all attempts retained. A met error threshold alone is not a full paper-protocol or hardware reproduction.',paper=dict(arxiv='2602.19475v1',repository_revision='5ff8219d2604de8a57a5b7a9c7db9b435c5c3257',reported={"3200":dict(relative_l2=.0173,updates=50000,training_seconds=90),"7500":dict(relative_l2=.0297,updates=50000,training_seconds=150)},hardware='RTX 3090, JAX'),records=records,original_protocol_seed_summary=summaries,measured_study_training_seconds=sum(times),verification=dict(python=platform.python_version(),torch=str(torch.__version__),inference='Fresh CPU network reconstruction; metric checked against stored GPU report with rtol=1e-4, atol=1e-8.'),
        limitations=['Random streams differ from upstream JAX; identical numeric seeds are not identical weights.','Training timing excludes evaluation and checkpoint I/O; compiler and wall time are separate.','Warm restarts change Adam/sampler state, rate and possibly batch size; cumulative costs are explicit.','Reference-selected checkpoints are separate from final fixed-budget checkpoints; selection search cost is reported.','Reference fields and upstream code are not redistributed.','These research checkpoints are separate from native GUI model/run archives.'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(records=len(records),study_training_seconds=sum(times),seed_summary=summaries),indent=2))


if __name__=='__main__':main()
