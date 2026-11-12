"""Headless numerical child. Only fixed job operations and data declarations load."""
import json
import os
from pathlib import Path
import sys
from threading import Event,Thread
from .codec import load_packet,save_packet,decode_config,decode_request,cpu_tree


def main():
    directory=Path(sys.argv[1]);job=load_packet(directory/'input.pt')
    if job.get('format')!='margam-job' or job.get('version')!=1 or job.get('operation') not in ('train','evaluate','predict'):
        raise ValueError('Invalid numerical job.')
    identifier=job['id'];cancelled=Event();finished=Event();sequence=0
    def emit(kind,**values):
        nonlocal sequence
        sequence+=1
        text=json.dumps(dict(job=identifier,sequence=sequence,kind=kind,**values),allow_nan=False,separators=(',',':'))
        if len(text.encode())>1024*1024:raise ValueError('Numerical event exceeds 1 MiB.')
        print(text,flush=True)
    def control():
        pending=b''
        while True:
            # A daemon must not hold sys.stdin's buffered lock at interpreter exit.
            raw=os.read(0,4096)
            if not raw:
                cancelled.set()
                if not finished.wait(5):os._exit(72)
                return
            pending+=raw
            if len(pending)>4096:cancelled.set();pending=b'';continue
            while b'\n' in pending:
                line,pending=pending.split(b'\n',1)
                try:
                    command=json.loads(line)
                    if command==dict(job=identifier,kind='cancel'):cancelled.set()
                except (ValueError,UnicodeError):cancelled.set()
    Thread(target=control,daemon=True).start();stage='model construction'
    try:
        import torch
        from ..templates.poisson import build_pinn
        from ..components.prediction import prediction_model
        from ..components.outputs import evaluate_outputs
        dtype=job['dtype']
        if dtype not in ('float16','bfloat16','float32','float64'):raise ValueError('Unsupported job scalar dtype.')
        torch.set_default_dtype(getattr(torch,dtype));torch.set_num_threads(1)
        config=decode_config(job['config']);operation=job['operation']
        requests=[decode_request(item,config) for item in job['requests']]
        emit('ready',pid=os.getpid())
        if operation=='train':
            model=build_pinn(config)
            if job['checkpoint'] is not None:
                stage='checkpoint restore';model.load_training_state(job['checkpoint'])
            stage='training update'
            while model.iter<config.iterations and not cancelled.is_set() and not getattr(model,'converged',False):
                previous=model.iter;controller=getattr(model,'sampling_controller',None)
                events=len(controller.history) if controller is not None else 0
                model.train(min(25,config.iterations-model.iter),display_every=0)
                emit('progress',step=model.iter,records=model.loss_history[previous:])
                for name,event in (('scale_summary','scale'),('expert_summary','expert')):
                    if hasattr(model,name):emit(event,summary=getattr(model,name)())
                if controller is not None:
                    for event in controller.history[events:]:emit('sampling',event=event)
                if getattr(model,'converged',False):break
            stage='checkpoint capture';checkpoint=cpu_tree(model.training_state())
        else:
            stage='checkpoint restore';checkpoint=job['checkpoint'];model=prediction_model(config,checkpoint)
        if operation=='predict':
            stage='output evaluation';result=dict(prediction=model.predict(job['points']).detach().cpu())
        else:
            stage='output evaluation'
            outputs=evaluate_outputs(model,config,requests) if getattr(model,'outputs_ready',True) else {}
            outputs={key:{name:value for name,value in data.items() if name!='config'} for key,data in outputs.items()}
            result=dict(outputs=outputs,steps=model.iter,cancelled=cancelled.is_set(),device=str(model.device))
        metadata=job.get('metadata') or {}
        if operation=='train':
            metadata={}
            if hasattr(model,'collocation_point_sample'):
                metadata['points']=model.collocation_point_sample
                metadata['constraints']=[dict(name=c.name,points=p,target=t) for c,p,t in model._constraints]
            for name in ('adaptive_summary','balancing_summary','scale_summary','expert_summary','attention_summary','dsgnar_summary'):
                if hasattr(model,name):metadata[name]=getattr(model,name)()
            if hasattr(model.network,'estimated_parameters'):metadata['estimated_parameters']=model.network.estimated_parameters()
        if any(name=='PySide6' or name.startswith('PySide6.') for name in sys.modules):raise RuntimeError('Numerical child unexpectedly imported Qt.')
        metadata=cpu_tree(metadata)
        stage='result transfer'
        save_packet(directory/'output.pt',dict(format='margam-job-result',version=1,id=identifier,pid=os.getpid(),
            checkpoint=checkpoint,result=result,metadata=metadata,headless=True))
        emit('complete')
        return 0
    except Exception as error:
        from ..components.resources import worker_failure
        emit('failed',message=worker_failure(error,stage,job['config'].get('device','unknown')))
        return 1
    finally:finished.set()


if __name__=='__main__':raise SystemExit(main())
