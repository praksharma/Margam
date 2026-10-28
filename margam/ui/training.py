"""Background baseline execution. Only plain numeric results cross into the UI."""
from threading import Event
from PySide6.QtCore import QThread, Signal


class TrainingWorker(QThread):
    progress = Signal(int, object)
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
        try:
            import torch
            from ..templates.poisson import build_pinn
            from ..components.outputs import evaluate_outputs
            # The application runs one training job at a time. Keep tiny full-batch
            # examples from spending their time scheduling large CPU thread pools.
            torch.set_num_threads(1)
            config = self.config
            model = self.model = build_pinn(config)
            if self.checkpoint is not None:model.load_training_state(self.checkpoint)
            while model.iter < config.iterations and not self.cancel_event.is_set():
                previous = model.iter
                events=len(model.sampling_controller.history) if model.sampling_controller is not None else 0
                model.train(min(25, config.iterations-model.iter),display_every=0)
                self.progress.emit(model.iter, model.loss_history[previous:])
                if model.sampling_controller is not None:
                    for event in model.sampling_controller.history[events:]:self.sampling_updated.emit(dict(event))
            outputs = evaluate_outputs(model, config, self.requests)
            self.completed.emit({'outputs': outputs, 'steps': model.iter,
                                 'cancelled': self.cancel_event.is_set(),
                                 'device': str(model.device), 'config': config})
        except Exception as error:
            self.failed.emit(f'{type(error).__name__}: {error}')
