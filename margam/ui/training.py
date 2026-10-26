"""Background baseline execution. Only plain numeric results cross into the UI."""
import math
from threading import Event
from PySide6.QtCore import QThread, Signal


class TrainingWorker(QThread):
    progress = Signal(int, object)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.cancel_event = Event()

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        try:
            import torch
            from ..templates.poisson import build_pinn, interval_reference
            # The application runs one training job at a time. Keep tiny full-batch
            # examples from spending their time scheduling large CPU thread pools.
            torch.set_num_threads(1)
            config = self.config
            model = build_pinn(config)
            while model.iter < config.iterations and not self.cancel_event.is_set():
                previous = model.iter
                model.train(min(25, config.iterations-model.iter),display_every=0)
                self.progress.emit(model.iter, model.loss_history[previous:])
            lower,upper = config.bounds
            size = 101 if config.shape == 'Interval' else 41
            xs = [lower[0]+i*(upper[0]-lower[0])/(size-1) for i in range(size)]
            if config.shape == 'Interval':
                points = [[x] for x in xs]
            else:
                ys = [lower[1]+i*(upper[1]-lower[1])/(size-1) for i in range(size)]
                points = [[x,y] for y in ys for x in xs]
            values = model.predict(torch.tensor(points)).detach().cpu().reshape(-1).tolist()
            reference = interval_reference(config,points)
            rmse = math.sqrt(sum((a-b)**2 for a,b in zip(values,reference))/len(values)) if reference is not None else None
            self.completed.emit({'points':points,'values':values,'reference':reference,
                                 'rmse':rmse,'size':size,'steps':model.iter,
                                 'cancelled':self.cancel_event.is_set(), 'device':str(model.device),
                                 'config':config})
        except Exception as error:
            self.failed.emit(f'{type(error).__name__}: {error}')
