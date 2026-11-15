"""Serial fresh-study execution using the same isolated numerical jobs as desktop."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import stat
from tempfile import NamedTemporaryFile,TemporaryDirectory
import time
from .. import study_document as sd
from ..studies import resolve_axes
from .graph import DocumentGraph,prepare_graph
from .transport import JobTransport,JobFailed,validate_wall_time


def timestamp():return datetime.now(timezone.utc).isoformat()


def atomic_json(path,value):
    path=Path(path);temporary=None
    try:
        with NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,prefix='.record-',delete=False) as stream:
            temporary=Path(stream.name);json.dump(value,stream,allow_nan=False,sort_keys=True,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def source_identity():
    root=Path(__file__).resolve().parents[1]
    files={'margam/'+path.relative_to(root).as_posix():sd.digest(path.read_bytes()) for path in sorted(root.rglob('*.py'))}
    return dict(sha256=sd.digest(sd.canonical(files)),files=files)


def source_matches(identity):
    try:return source_identity()==identity
    except OSError:return False


def file_digest(path):
    hasher=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(1024*1024):hasher.update(chunk)
    return hasher.hexdigest()


def snapshot(source,destination):
    """Freeze the exact input archive before parsing/copying its identity."""
    limit=sd.TOTAL_LIMIT+sd.MANIFEST_LIMIT+4*1024*1024
    fd=os.open(source,os.O_RDONLY|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        before=os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size>limit:raise ValueError('Choose a regular study file within the archive byte budget.')
        size=0;hasher=hashlib.sha256()
        with Path(destination).open('xb') as output:
            while chunk:=stream.read(1024*1024):
                size+=len(chunk)
                if size>limit:raise ValueError('Study changed or exceeded the archive byte budget.')
                output.write(chunk);hasher.update(chunk)
        after=os.fstat(stream.fileno())
        if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('Study changed while reading; retry from a stable file.')
    return hasher.hexdigest()


def inspect(path):
    study,files=sd.read(path);_,total=resolve_axes(study['model'],study['axes'])
    return dict(study_id=study['study_id'],trials=total,inputs=len(files),input_bytes=sum(map(len,files.values())),execution=study['execution'])


def runtime(device):
    import torch
    import numpy as np
    available=torch.cuda.is_available();effective=('cuda' if available else 'cpu') if device=='auto' else device
    reason='CUDA was explicitly requested but is unavailable; no CPU fallback was used.' if effective=='cuda' and not available else None
    try:version=importlib.metadata.version('margam')
    except importlib.metadata.PackageNotFoundError:version='source checkout'
    return effective,reason,dict(python=platform.python_version(),margam=version,torch=str(torch.__version__),numpy=str(np.__version__),
        cuda_runtime=torch.version.cuda,system=platform.system(),machine=platform.machine(),gpu=torch.cuda.get_device_name(0) if available and effective=='cuda' else None)


def run_study(path,output,*,device=None,wall_time_seconds=None,exports=False):
    """Run every finite-grid trial fresh; never overwrite/resume an existing folder.

    Ctrl-C discards/reaps the active isolated job, retains completed trials and
    records the rest as not_started. Grid/checkpoint resume is separate future work.
    """
    if device is not None and device not in ('auto','cpu','cuda'):raise ValueError('Choose auto, cpu or cuda.')
    if type(exports) is not bool:raise ValueError('Exports must be explicitly enabled or disabled.')
    if wall_time_seconds is not None:validate_wall_time(wall_time_seconds)
    output=Path(output).absolute()
    if output.exists() or output.is_symlink():raise FileExistsError('Choose a fresh output directory. Existing study results are never overwritten.')
    with TemporaryDirectory(prefix='margam-study-source-') as directory:
        frozen=Path(directory)/'study.margam-study';archive_sha=snapshot(path,frozen)
        study,files=sd.read(frozen);_,total=resolve_axes(study['model'],study['axes'])
        intent=study['execution'];requested=device or intent['device'];limit=intent['wall_time_seconds'] if wall_time_seconds is None else wall_time_seconds
        effective,incompatible,versions=runtime(requested);identity=source_identity()
        output.mkdir(mode=0o700,parents=True,exist_ok=False);shutil.copyfile(frozen,output/'study.margam-study')
        rows=[dict(index=i,trial_id=f"DP{i+1:04d}-{study['study_id']}",directory=f'DP{i+1:04d}',status='not_started',stage=None,reason=None,steps=None,metrics=None) for i in range(total)]
        report=dict(format='margam-study-results',version=1,study_id=study['study_id'],study_archive_sha256=archive_sha,
            started_utc=timestamp(),finished_utc=None,state='running',headless=True,
            saved_execution=intent,effective_execution=dict(requested_device=requested,device=effective,wall_time_seconds=limit,exports=exports),
            runtime=versions,source=identity,trials=rows,counts={})
        def publish():
            report['counts']=dict(Counter(row['status'] for row in rows));atomic_json(output/'summary.json',report)
        publish();stop=False;interrupted=False
        try:
            for row in rows:
                if stop:break
                started=time.perf_counter();folder=output/row['directory'];folder.mkdir(mode=0o700)
                stage=folder/'.artifacts';workspace=None;transport=None
                row.update(status='preparing',stage='preflight',started_utc=timestamp());publish()
                detail=dict(**row,choices=[],model_sha256=None,config=None,artifacts=[],exports=[],timing={})
                try:
                    if not source_matches(identity):raise RuntimeError('Runtime source changed since this study started; remaining trials were not launched.')
                    identifier,model=sd.trial_model(study,row['index']);assert identifier==row['trial_id']
                    detail['model_sha256']=sd.digest(sd.canonical(model))
                    by_id={node['id']:node for node in model['nodes']}
                    detail['choices']=[dict(node_id=axis['node_id'],parameter=axis['parameter'],value=deepcopy(by_id[axis['node_id']]['settings']['parameters'][axis['parameter']])) for axis in study['axes']]
                    atomic_json(folder/'model.margam.json',model)
                    workspace=sd.restore(files);graph=DocumentGraph(model,workspace,effective)
                    config,requests=prepare_graph(graph)
                    detail['config']=workspace.portable_document(asdict(config));detail['timing']['preflight_seconds']=time.perf_counter()-started
                    model['config']=detail['config'];detail['model_sha256']=sd.digest(sd.canonical(model))
                    atomic_json(folder/'model.margam.json',model);atomic_json(folder/'config.json',detail['config'])
                    if incompatible:
                        row.update(status='incompatible',reason=incompatible);continue
                    if not source_matches(identity):raise RuntimeError('Runtime source changed before numerical launch; remaining trials were not launched.')
                    from .codec import encode_config,encode_request
                    transport=JobTransport(wall_time_seconds=limit)
                    row.update(status='running',stage='numerical');publish();job_started=time.perf_counter()
                    with (folder/'events.jsonl').open('x',encoding='utf-8',buffering=1) as log:
                        def event(value):
                            log.write(json.dumps(value,allow_nan=False)+'\n')
                            if value['kind']=='ready':detail['child_pid']=value['pid']
                        packet=transport.run(dict(operation='train',config=encode_config(config),requests=[encode_request(r) for r in requests],
                            checkpoint=None,metadata={},dtype='float32'),event)
                    detail['timing']['numerical_job_seconds']=time.perf_counter()-job_started
                    if not source_matches(identity):raise RuntimeError('Runtime source changed during numerical work; this trial was not published and remaining trials were not launched.')
                    if not packet.get('headless') or packet['result']['steps']!=packet['checkpoint']['iter']:raise ValueError('Numerical checkpoint/result mismatch.')
                    if packet['result']['cancelled']:raise InterruptedError('Numerical job was cancelled; no completed trial was published.')
                    row['stage']='artifacts';stage.mkdir(mode=0o700);artifact_started=time.perf_counter()
                    from ..ui.run_document import save_run
                    model['config']=detail['config']
                    nodes={node.identifier:node for node in graph.nodes()}
                    monitors={request.identifier:list(enumerate(packet['checkpoint']['loss_history'],1))[-nodes[request.identifier].settings['history_limit']:] for request in requests if request.kind=='monitor'}
                    archive=dict(format='margam-run',version=4,model=model,result=packet['result'],monitors=monitors,checkpoint=packet['checkpoint'],owned_files=files)
                    save_run(stage/'run.margam-run.pt',archive)
                    metrics={}
                    from ..components.error_norms import error_norms
                    from ..components.output_export import export_data,extension
                    for number,(identifier,data) in enumerate(packet['result']['outputs'].items(),1):
                        if data.get('reference') is not None:
                            metrics[identifier]=error_norms(data['values'],data['reference'],fields=data.get('fields',['u']))
                        if exports and nodes[identifier].kind in ('export','paraview'):
                            data=dict(data,config=config);name=f'output-{number:04d}-'+sd.digest(identifier.encode())[:16]+extension(data,nodes[identifier].kind)
                            export_data(stage/name,data,nodes[identifier].kind)
                            detail['exports'].append(dict(node_id=identifier,block=nodes[identifier].title,kind=nodes[identifier].kind,path=name))
                    artifact_map=[dict(path=target.relative_to(stage).as_posix(),size=target.stat().st_size,sha256=file_digest(target)) for target in sorted(stage.rglob('*')) if target.is_file()]
                    if not source_matches(identity):raise RuntimeError('Runtime source changed during artifact preparation; this trial was not published and remaining trials were not launched.')
                    stage.rename(folder/'artifacts');detail['artifacts']=artifact_map
                    detail['timing']['artifact_seconds']=time.perf_counter()-artifact_started
                    detail['last_loss']=deepcopy(packet['checkpoint']['loss_history'][-1]) if packet['checkpoint']['loss_history'] else None
                    detail['dtype']=str(next(iter(packet['checkpoint']['network'].values())).dtype)
                    row.update(status='completed',stage='complete',steps=packet['result']['steps'],metrics=metrics)
                except KeyboardInterrupt:
                    row.update(status='interrupted',reason='Interrupted by the user. The active job was discarded; already published artifacts remain available.');stop=True;interrupted=True
                except InterruptedError as error:
                    row.update(status='interrupted',reason=str(error));stop=True;interrupted=True
                except Exception as error:
                    row.update(status='invalid' if row['stage']=='preflight' else 'failed',reason=str(error))
                    if transport is not None and transport.deadline_exceeded:detail['failure_kind']='job_time_limit'
                    if not source_matches(identity):
                        detail['failure_kind']='source_changed';row['status']='failed';stop=True
                finally:
                    if workspace is not None:workspace.close()
                    if stage.exists():shutil.rmtree(stage)
                    detail.update(row);detail['timing']['wall_seconds']=time.perf_counter()-started;detail['finished_utc']=timestamp()
                    detail['artifacts_published']=(folder/'artifacts').exists()
                    atomic_json(folder/'trial.json',detail);publish()
        except KeyboardInterrupt:
            stop=True;interrupted=True
            for row in rows:
                if row['status'] in ('preparing','running'):
                    row.update(status='interrupted',reason='Interrupted while recording this trial; no further trial was started.')
                    folder=output/row['directory']
                    if (folder/'.artifacts').exists():shutil.rmtree(folder/'.artifacts')
                    atomic_json(folder/'trial.json',row)
            if workspace is not None:workspace.close()
        report['finished_utc']=timestamp()
        report['state']='interrupted' if interrupted or any(row['status']=='interrupted' for row in rows) else 'aborted' if stop else 'completed' if all(row['status']=='completed' for row in rows) else 'finished_with_errors'
        publish()
        if any(name=='PySide6' or name.startswith('PySide6.') for name in __import__('sys').modules):
            # Library callers may be the GUI; the CLI separately enforces no Qt.
            report['headless']=False;publish()
        return report
