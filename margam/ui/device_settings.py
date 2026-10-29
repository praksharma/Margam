"""One desktop execution preference, independent of model physics and block settings."""
from dataclasses import asdict
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import QLabel


def same_physics(left,right):
    if left is None or right is None:return False
    left,right=asdict(left),asdict(right)
    left.pop('device',None);right.pop('device',None)
    return left==right


class DeviceSettings:
    def setup_device_menu(self):
        import torch
        self._cuda_available=torch.cuda.is_available()
        self._gpu_name=torch.cuda.get_device_name(0) if self._cuda_available else 'Unavailable'
        self.device_menu=self.menuBar().addMenu('Device')
        group=QActionGroup(self);group.setExclusive(True);self.device_actions={}
        for key,label in [('auto','Automatic · GPU when available'),('cpu','CPU'),('cuda','GPU · '+self._gpu_name)]:
            action=self.device_menu.addAction(label);action.setCheckable(True);group.addAction(action)
            action.setEnabled(key!='cuda' or self._cuda_available)
            action.triggered.connect(lambda checked=False,value=key:self.set_application_device(value))
            self.device_actions[key]=action
        self.device_menu.addSeparator();self.device_menu.addAction('Device information…').triggered.connect(self.show_device_information)
        self.device_status=QLabel();self.statusBar().addPermanentWidget(self.device_status)
        self.refresh_device_status()

    def execution_device(self):
        return ('cuda' if self._cuda_available else 'cpu') if self.application_device=='auto' else self.application_device

    def refresh_device_status(self):
        if not hasattr(self,'device_actions'):return
        for key,action in self.device_actions.items():action.setChecked(key==self.application_device)
        self.device_status.setText('Device: '+('Auto → ' if self.application_device=='auto' else '')+('GPU' if self.execution_device()=='cuda' else 'CPU'))
        self.device_status.setToolTip('Application-wide target for training, resumed training and prediction. Data preparation and export stay on the CPU.')

    def set_application_device(self,value,*,remember=True):
        if value not in ('auto','cpu','cuda') or self.is_running() or self.scene.locked:return False
        if value=='cuda' and not self._cuda_available:
            self.statusBar().showMessage('CUDA GPU is unavailable. Choose Automatic or CPU.');return False
        self.application_device=value;self.config.device=self.execution_device()
        state=getattr(self,'desktop_state',None)
        if remember and state is not None:state.settings.setValue('execution_device',value)
        self.refresh_device_status();self.selection_changed()
        self.statusBar().showMessage('Training and prediction device: '+('GPU · '+self._gpu_name if self.execution_device()=='cuda' else 'CPU'))
        return True

    def show_device_information(self):
        import torch
        from .setting_help import show_information
        return show_information(self,'Execution device',
            'One application preference controls new training, checkpoint continuation and inference for all trainers. Model files may contain older per-block device values; the application preference takes priority.\n\n'
            'Automatic uses the CUDA GPU when available, otherwise the CPU. GPU uses the first CUDA device; CPU runs numerical work on the processor. Data loading, geometry preparation, tables and file export remain on the CPU. Switching device preserves trained weights; the next prediction reconstructs them on the chosen target without retraining. Device changes are locked while a job or external editor is active.\n\n'
            f'Preference: {self.application_device}\nEffective device: {self.execution_device()}\nGPU: {self._gpu_name}\nPyTorch: {torch.__version__}\nCompiled CUDA: {torch.version.cuda or "none"}')
