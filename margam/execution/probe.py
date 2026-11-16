"""Explicit disposable native training probes; no capacity prediction or checkpoint."""
import time
from contextlib import contextmanager


def is_windowed(config):return config.method=='expert' and config.expert_settings['windows']>1


def probe_limit(config):return min(config.iterations,100000 if is_windowed(config) else 1000)


def validate_probe(config,updates):
    maximum=probe_limit(config)
    if type(updates) is not int or not 1<=updates<=maximum:raise ValueError(f'Probe updates/attempts must be an integer from 1 to {maximum}, within the original training budget.')


def process_peak_rss():
    try:
        import resource,sys
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    except (ImportError,OSError):return None


class WindowMeasurement:
    """Concrete Expert window/teacher-transfer phases, never a solver checkpoint."""
    def __init__(self,config,cuda):
        self.cuda=cuda;self.device='cuda' if cuda else 'cpu';self.phases=[];self.current=None
        self.count=config.expert_settings['windows'];self.steps=config.expert_settings['window_steps']
        low,high=config.bounds;self.edges=[low[-1]+(high[-1]-low[-1])*i/self.count for i in range(self.count+1)]
        self.begin('construction',1,0)
    def sync(self):
        if self.cuda:
            import torch
            torch.cuda.synchronize(self.device)
    def begin(self,kind,window,step):
        import torch
        self.sync()
        if self.cuda:torch.cuda.reset_peak_memory_stats(self.device)
        self.current=dict(kind=kind,window=window,start_update=step,time_interval=self.edges[window-1:window+1],
            cuda_start_allocated_bytes=torch.cuda.memory_allocated(self.device) if self.cuda else None,
            cuda_start_reserved_bytes=torch.cuda.memory_reserved(self.device) if self.cuda else None)
        self.started=time.perf_counter()
    def end(self,model):
        import torch
        self.sync();phase=self.current
        if phase is None:raise RuntimeError('No active Expert probe phase.')
        phase.update(end_update=model.iter,duration_seconds=time.perf_counter()-self.started,
            process_peak_rss_bytes=process_peak_rss(),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if self.cuda else None,
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device) if self.cuda else None,
            cuda_end_allocated_bytes=torch.cuda.memory_allocated(self.device) if self.cuda else None,
            cuda_end_reserved_bytes=torch.cuda.memory_reserved(self.device) if self.cuda else None,
            completed_window_snapshots=len(model.completed),teacher_count=len(model.teachers),
            cached_teacher_copies=sum(len(teacher.cache) for teacher in model.teachers),pde_rows=len(model.collocation_point_sample))
        self.phases.append(phase);self.current=None
        if len(self.phases)>64:raise RuntimeError('Expert probe phase budget exceeded.')
    def failure_stage(self,default):
        if self.current is None:return default
        p=self.current;return f"Expert {p['kind'].replace('_',' ')} (window {p['window']}, starting at update {p['start_update']})"
    def attach(self,model):
        if model.iter or model.completed:raise ValueError('Expert resource measurement requires a fresh model.')
        if model._transition_observer is not None:raise ValueError('An Expert transition observer is already active.')
        self.end(model);self.begin('window_training',1,model.iter);model._transition_observer=self.transition
    @contextmanager
    def transition(self,model):
        self.end(model);self.begin('teacher_transfer',len(model.completed)+2,model.iter)
        yield
        self.end(model);self.begin('window_training',len(model.completed)+1,model.iter)
    def finish(self,model):
        self.end(model);model._transition_observer=None
        if self.cuda:
            allocated=max(phase['cuda_peak_allocated_bytes'] for phase in self.phases)
            reserved=max(phase['cuda_peak_reserved_bytes'] for phase in self.phases)
        else:allocated=reserved=None
        return dict(version=2,configured_windows=self.count,updates_per_window=self.steps,
            observed_windows=len(model.completed)+1,observed_transfers=len(model.completed),
            next_transfer_pending=len(model.completed)<self.count-1 and model.active.iter>=self.steps,
            teacher_count=len(model.teachers),cached_teacher_copies=sum(len(t.cache) for t in model.teachers),
            internal_window_snapshots_included=True,phases=self.phases,
            cuda_peak_allocated_bytes=allocated,cuda_peak_reserved_bytes=reserved,
            peak_aggregation='CUDA overall peaks are maxima of sequential phase peaks, not sums. Phase allocator peaks include live state from earlier phases; CPU RSS is a cumulative process high-water mark, never a phase-local increment.',
            measurement_overhead_included=True)


def begin_measurement(config):
    import torch
    cuda=config.device=='cuda' or config.device=='auto' and torch.cuda.is_available()
    if cuda:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    result=dict(cuda=cuda,started=time.perf_counter())
    if is_windowed(config):result['windows']=WindowMeasurement(config,cuda)
    return result


def finish_measurement(measurement,model,updates,initial_rows):
    import torch,platform
    cuda=measurement['cuda']
    if cuda:torch.cuda.synchronize(model.device)
    windowed=measurement['windows'].finish(model) if 'windows' in measurement else None
    elapsed=time.perf_counter()-measurement['started'];rss=process_peak_rss()
    controller=getattr(model,'sampling_controller',None)
    result=dict(format='margam-resource-probe',version=1,method=model.__class__.__name__,device=str(model.device),dtype=str(next(model.network.parameters()).dtype).removeprefix('torch.'),
        requested_updates=updates,completed_updates=model.iter,converged=bool(getattr(model,'converged',False)),
        build_and_train_seconds=elapsed,process_peak_rss_bytes=rss,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(model.device) if cuda else None,
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(model.device) if cuda else None,
        initial_pde_rows=initial_rows,final_pde_rows=len(model.collocation_point_sample),observed_sampling_events=len(controller.history) if controller is not None else 0,
        scope='Fresh disposable model; complete native updates/attempts at the original schedule horizon. CPU RSS is the whole child-process high-water mark, including imports; CUDA peaks cover model construction and training allocator use after context initialization.',
        excluded=['output evaluation and exports','checkpoint capture and transfer','later adaptive growth or schedules not reached','other processes and CUDA driver/context memory'],
        runtime=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda),capacity_prediction=False,checkpoint_retained=False)
    if windowed:
        result.update(windowed)
        result['scope']='Fresh disposable Expert time marching; complete native updates at the original horizon, including every teacher transfer actually reached. CUDA allocator phases include construction, window training and handoff; context initialization is excluded. Internal completed-window snapshots and frozen teachers are included. Profiling synchronization/bookkeeping overhead is included and can alter overlap/allocator reuse relative to uninstrumented execution.'
        result['runtime'].update(machine=platform.machine(),device_name=torch.cuda.get_device_name(model.device) if cuda else platform.machine())
        result['excluded']=['output evaluation and exports','final run checkpoint capture and transfer','later windows or schedules not reached','other processes and CUDA driver/context memory']
        from .probe_report import validate_window_report
        validate_window_report(result)
    return result


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
            result=packet['result']
            if is_windowed(config):
                from .probe_report import validate_window_report
                validate_window_report(result)
                if result['configured_windows']!=config.expert_settings['windows'] or result['updates_per_window']!=config.expert_settings['window_steps']:raise ValueError('Probe window schedule differs from the selected trial.')
            elif result.get('version')!=1:raise ValueError('Unexpected single-window probe version.')
            result.update(study_id=study['study_id'],trial_id=trial_id,study_archive_sha256=archive_sha,source_sha256=identity['sha256'],numerical_job_seconds=elapsed,requested_output_count=len(requests),requested_device=device or study['execution']['device'],wall_time_seconds=transport.wall_time_seconds,configured_updates=config.iterations,configured_method=config.method,network_type=config.network_type,layers=list(config.layers),adaptive_sampling=config.adaptive_sampling)
            return result
        finally:workspace.close()
