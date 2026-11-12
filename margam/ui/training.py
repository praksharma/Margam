"""Background baseline execution. Only plain numeric results cross into the UI."""
from threading import Event
from PySide6.QtCore import QThread, Signal


class TrainingWorker(QThread):
    progress = Signal(int, object)
    expert_progress = Signal(object)
    scale_progress = Signal(object)
    sampling_updated = Signal(object)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, config, requests=(), parent=None, checkpoint=None):
        super().__init__(parent)
        self.checkpoint = checkpoint
        self.config = config
        self.requests = tuple(requests)
        self.model = None
        self.cancel_event = Event()

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        stage='model construction'
        try:
            import torch
            from ..templates.poisson import build_pinn
            from ..components.outputs import evaluate_outputs
            # The application runs one training job at a time. Keep tiny full-batch
            # examples from spending their time scheduling large CPU thread pools.
            torch.set_num_threads(1)
            config = self.config
            model = self.model = build_pinn(config)
            if self.checkpoint is not None:
                stage='checkpoint restore';model.load_training_state(self.checkpoint)
            stage='training update'
            while model.iter < config.iterations and not self.cancel_event.is_set() and not getattr(model,'converged',False):
                previous = model.iter
                events=len(model.sampling_controller.history) if model.sampling_controller is not None else 0
                model.train(min(25, config.iterations-model.iter),display_every=0)
                self.progress.emit(model.iter, model.loss_history[previous:])
                if hasattr(model,'scale_summary'):self.scale_progress.emit(model.scale_summary())
                if hasattr(model,'expert_summary'):self.expert_progress.emit(model.expert_summary())
                if model.sampling_controller is not None:
                    for event in model.sampling_controller.history[events:]:self.sampling_updated.emit(dict(event))
                if getattr(model,'converged',False):break
            stage='output evaluation'
            outputs = evaluate_outputs(model, config, self.requests) if getattr(model,'outputs_ready',True) else {}
            self.completed.emit({'outputs': outputs, 'steps': model.iter,
                                 'cancelled': self.cancel_event.is_set(),
                                 'device': str(model.device), 'config': config})
        except Exception as error:
            from ..components.resources import worker_failure
            self.model=None;self.checkpoint=None
            self.failed.emit(worker_failure(error,stage,self.config.device))


class EvaluationWorker(QThread):
    completed=Signal(object)
    failed=Signal(str)
    evaluation_only=True

    def __init__(self,config,requests,*,model=None,checkpoint=None,parent=None):
        super().__init__(parent);self.config=config;self.requests=tuple(requests);self.model=model;self.checkpoint=checkpoint;self.cancel_event=Event()

    def cancel(self):self.cancel_event.set()

    def run(self):
        stage='checkpoint restore'
        try:
            from ..components.prediction import prediction_model
            from ..components.outputs import evaluate_outputs
            if self.model is None:self.model=prediction_model(self.config,self.checkpoint)
            stage='output evaluation'
            outputs=evaluate_outputs(self.model,self.config,self.requests)
            self.completed.emit(dict(outputs=outputs,steps=self.model.iter,cancelled=self.cancel_event.is_set(),device=str(self.model.device),config=self.config))
        except Exception as error:
            from ..components.resources import worker_failure
            self.model=None;self.checkpoint=None
            self.failed.emit(worker_failure(error,stage,self.config.device))


class IsolatedJob:
    """A supervisor thread emits Qt signals; all numerical work is a child job."""
    isolated=True
    def _setup_transport(self):
        from ..execution.transport import JobTransport
        self.transport=JobTransport();self.child_pid=None
    def cancel(self):
        self.cancel_event.set();self.transport.cancel()
    def force_stop(self):self.transport.terminate()
    def _run_job(self,operation):
        stage='job preparation'
        try:
            import torch
            from ..execution.codec import encode_config,encode_request
            from ..execution.retained import RetainedModel
            previous=self.model
            checkpoint=self.checkpoint if self.checkpoint is not None else previous.training_state() if previous is not None else None
            metadata=previous.metadata if isinstance(previous,RetainedModel) else {}
            dtype=next(iter(checkpoint['network'].values())).dtype if checkpoint is not None else torch.get_default_dtype()
            payload=dict(operation=operation,config=encode_config(self.config),requests=[encode_request(r) for r in self.requests],
                checkpoint=checkpoint,metadata=metadata,dtype=str(dtype).removeprefix('torch.'))
            def event(value):
                kind=value['kind']
                if kind=='ready':self.child_pid=value['pid']
                elif kind=='progress':self.progress.emit(value['step'],value['records'])
                elif kind=='expert':self.expert_progress.emit(value['summary'])
                elif kind=='scale':self.scale_progress.emit(value['summary'])
                elif kind=='sampling':self.sampling_updated.emit(value['event'])
            stage='isolated numerical job'
            packet=self.transport.run(payload,event)
            result=packet['result'];checkpoint=packet['checkpoint']
            if result['steps']!=checkpoint['iter']:raise ValueError('Numerical result/checkpoint phase mismatch.')
            self.child_pid=packet['pid'];self.headless=packet['headless']
            self.model=previous if isinstance(previous,RetainedModel) and str(previous.device)==result['device'] else RetainedModel(self.config,checkpoint,packet['metadata'],result['device'])
            result['config']=self.config
            for output in result['outputs'].values():output['config']=self.config
            self.checkpoint=None;self.completed.emit(result)
        except Exception as error:
            from ..components.resources import worker_failure
            self.model=None;self.checkpoint=None
            from ..execution.transport import JobFailed
            self.failed.emit(str(error) if isinstance(error,JobFailed) else worker_failure(error,stage,self.config.device))


class IsolatedTrainingWorker(IsolatedJob,TrainingWorker):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self._setup_transport()
    def run(self):self._run_job('train')


class IsolatedEvaluationWorker(IsolatedJob,EvaluationWorker):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self._setup_transport()
    def run(self):self._run_job('evaluate')
