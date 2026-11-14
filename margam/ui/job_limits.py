"""Local execution wall-time policy, independent of numerical model settings."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog,QVBoxLayout,QLabel,QFormLayout,QSpinBox,QDialogButtonBox
from ..execution.transport import MAX_WALL_TIME_SECONDS
from .icons import icon

DESCRIPTION=(
    'Maximum time per numerical job, in seconds (0 = no limit).\n\n'
    'Includes starting the calculation, training/continuation or output evaluation, and preparing its results. '
    'Timed-out jobs are stopped and discarded without a resumable partial result. Saved run files are preserved.\n\n'
    'Preparing inputs and reading results in the main window, explicit file exports and geometry editing are outside this limit. '
    'This setting does not limit memory or change samples, precision or optimizer settings.'
)


class JobLimitDialog(QDialog):
    def __init__(self,seconds,parent=None):
        super().__init__(parent);self.setWindowTitle('Numerical job time limit');self.resize(620,250)
        layout=QVBoxLayout(self)
        description=QLabel(DESCRIPTION);description.setTextFormat(Qt.TextFormat.PlainText);description.setWordWrap(True);layout.addWidget(description)
        form=QFormLayout();layout.addLayout(form)
        self.seconds=QSpinBox();self.seconds.setObjectName('job_time_limit_seconds')
        self.seconds.setRange(0,MAX_WALL_TIME_SECONDS);self.seconds.setSpecialValueText('No limit');self.seconds.setSingleStep(60);self.seconds.setKeyboardTracking(False);self.seconds.setValue(seconds)
        self.seconds.setToolTip('0 means unlimited. The timer starts when the numerical child process launches. A deadline discards the running job.')
        form.addRow('Maximum job time (seconds)',self.seconds)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)


class JobLimits:
    def setup_job_limit_menu(self):
        self.job_time_limit_action=self.run_menu.addAction('Job time limit…')
        self.job_time_limit_action.setIcon(icon('time_limit'))
        self.job_time_limit_action.setToolTip('Maximum time for each numerical job; 0 means no limit. Expired jobs are discarded.')
        self.job_time_limit_action.triggered.connect(self.request_job_time_limit)

    def set_job_time_limit(self,seconds,*,remember=True):
        if type(seconds) is not int or not 0<=seconds<=MAX_WALL_TIME_SECONDS or self.is_running() or self.scene.locked:return False
        self.job_time_limit_seconds=seconds
        state=getattr(self,'desktop_state',None)
        if remember and state is not None:state.settings.setValue('job_time_limit_seconds',seconds)
        self.job_time_limit_action.setToolTip('Numerical job time limit: '+(str(seconds)+' seconds; expired jobs are discarded.' if seconds else 'No limit.'))
        self.statusBar().showMessage('Numerical job time limit: '+(str(seconds)+' seconds' if seconds else 'No limit'))
        return True

    def request_job_time_limit(self):
        if self.is_running() or self.scene.locked:return False
        dialog=JobLimitDialog(self.job_time_limit_seconds,self)
        return self.set_job_time_limit(dialog.seconds.value()) if dialog.exec()==QDialog.DialogCode.Accepted else False
