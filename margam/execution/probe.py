"""Explicit disposable native training probes; no capacity prediction or checkpoint."""
import time


def validate_probe(config,updates):
    if type(updates) is not int or not 1<=updates<=1000:raise ValueError('Probe updates/attempts must be an integer from 1 to 1000.')
    if updates>config.iterations:raise ValueError('Probe budget cannot exceed the configured training budget.')
    if config.method=='expert' and config.expert_settings['windows']>1:raise ValueError('Resource probes currently require a single Expert time window; teacher-transfer peaks are not qualified.')


def begin_measurement(config):
    import torch
    cuda=config.device=='cuda' or config.device=='auto' and torch.cuda.is_available()
    if cuda:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    return dict(cuda=cuda,started=time.perf_counter())


def finish_measurement(measurement,model,updates,initial_rows):
    import torch,platform
    cuda=measurement['cuda']
    if cuda:torch.cuda.synchronize(model.device)
    elapsed=time.perf_counter()-measurement['started']
    rss=None
    try:
        import resource,sys
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    except (ImportError,OSError):pass
    controller=getattr(model,'sampling_controller',None)
    return dict(format='margam-resource-probe',version=1,method=model.__class__.__name__,device=str(model.device),dtype=str(next(model.network.parameters()).dtype).removeprefix('torch.'),
        requested_updates=updates,completed_updates=model.iter,converged=bool(getattr(model,'converged',False)),
        build_and_train_seconds=elapsed,process_peak_rss_bytes=rss,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(model.device) if cuda else None,
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(model.device) if cuda else None,
        initial_pde_rows=initial_rows,final_pde_rows=len(model.collocation_point_sample),observed_sampling_events=len(controller.history) if controller is not None else 0,
        scope='Fresh disposable model; complete native updates/attempts at the original schedule horizon. CPU RSS is the whole child-process high-water mark, including imports; CUDA peaks cover model construction and training allocator use after context initialization.',
        excluded=['output evaluation and exports','checkpoint capture and transfer','later adaptive growth or schedules not reached','other processes and CUDA driver/context memory'],
        runtime=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda),capacity_prediction=False,checkpoint_retained=False)


def probe_study(path,*,trial=1,updates=1,device=None,wall_time_seconds=60):
    """Probe one saved design point, preserving the study and parent numerical state."""
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from .. import study_document as sd
    from .graph import DocumentGraph,prepare_graph
    from .study_runner import snapshot,source_identity,source_matches
    from .codec import encode_config
    from .transport import JobTransport
    if type(trial) is not int or trial<1:raise ValueError('Choose a positive one-based trial number.')
    if device is not None and device not in ('auto','cpu','cuda'):raise ValueError('Choose Automatic, CPU or CUDA.')
    transport=JobTransport(wall_time_seconds=wall_time_seconds);identity=source_identity()
    with TemporaryDirectory(prefix='margam-probe-source-') as directory:
        frozen=Path(directory)/'study.margam-study';archive_sha=snapshot(path,frozen);study,files=sd.read(frozen)
        trial_id,model=sd.trial_model(study,trial-1);workspace=sd.restore(files)
        try:
            config,requests=prepare_graph(DocumentGraph(model,workspace,device or study['execution']['device']))
            validate_probe(config,updates)
            if not source_matches(identity):raise RuntimeError('Runtime source changed before the probe; no measurement published.')
            started=time.perf_counter()
            packet=transport.run(dict(operation='probe',config=encode_config(config),requests=[],checkpoint=None,metadata={},dtype='float32',probe_updates=updates))
            elapsed=time.perf_counter()-started
            if not source_matches(identity):raise RuntimeError('Runtime source changed during the probe; no measurement published.')
            if packet['checkpoint'] is not None or packet['metadata'] or not packet['headless']:raise ValueError('Unexpected retained probe state.')
            result=packet['result'];result.update(study_id=study['study_id'],trial_id=trial_id,study_archive_sha256=archive_sha,source_sha256=identity['sha256'],numerical_job_seconds=elapsed,requested_output_count=len(requests),requested_device=device or study['execution']['device'],wall_time_seconds=transport.wall_time_seconds,configured_updates=config.iterations,configured_method=config.method,network_type=config.network_type,layers=list(config.layers),adaptive_sampling=config.adaptive_sampling)
            return result
        finally:workspace.close()
