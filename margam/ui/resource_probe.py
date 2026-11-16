"""Reviewed desktop resource probes through the existing owned CLI process."""
import json
import os
from PySide6.QtCore import Qt,QTimer
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QFormLayout,QHBoxLayout,QLabel,QLineEdit,
    QPushButton,QSpinBox,QPlainTextEdit,QFileDialog,QProgressBar)
from .study_results import Reader
from .study_execution import StudyProcess,prepare_review
from .icons import icon


def prepare_probe(source,trial,updates,device,limit,check):
    from .. import study_document as sd
    from ..execution.graph import DocumentGraph,prepare_graph
    from ..execution.probe import validate_probe
    value=prepare_review('run',source,None,check);workspace=None
    try:
        study,files=sd.read(value['path']);identifier,model=sd.trial_model(study,trial-1)
        workspace=sd.restore(files);config,requests=prepare_graph(DocumentGraph(model,workspace,device));validate_probe(config,updates)
        if check():raise InterruptedError('Review cancelled.')
        value['intent']=(source,trial,updates,device,limit)
        value['trial_id']=identifier
        value['description']=dict(method=config.method,physics=config.pde_preset or config.physics,network=config.network_type,layers=list(config.layers),configured_updates=config.iterations,requested_outputs=len(requests))
        return value
    except BaseException:value['temporary'].cleanup();raise
    finally:
        if workspace is not None:workspace.close()


def report_text(value):
    def memory(number):return 'not measured' if number is None else f'{number/1024**2:.2f} MiB ({number} bytes)'
    return '\n'.join([f"{value['trial_id']}\n{value['completed_updates']} complete updates/attempts observed; original horizon {value['configured_updates']}",
        f"{value['configured_method']} · {value['network_type']} · {value['device']} · {value['dtype']}",
        'Child process peak RAM: '+memory(value['process_peak_rss_bytes']),
        'CUDA allocated peak: '+memory(value['cuda_peak_allocated_bytes']),
        'CUDA reserved peak: '+memory(value['cuda_peak_reserved_bytes']),
        f"Construction + updates: {value['build_and_train_seconds']:.3f} s; transport/startup wall: {value['numerical_job_seconds']:.3f} s",
        f"PDE rows: {value['initial_pde_rows']} → {value['final_pde_rows']}; residual sampler events observed: {value['observed_sampling_events']}",
        f"Output requests not evaluated: {value['requested_output_count']}",'',value['scope'],'Excluded: '+', '.join(value['excluded']),
        '', 'Measured scope only; no full-run fit prediction. No weights, checkpoint or numerical outputs retained.','',json.dumps(value,indent=2,allow_nan=False)])


class ResourceProbe(QDialog):
    def __init__(self,owner):
        super().__init__(owner);self.owner=owner;self.reader=None;self.worker=None;self.prepared=None;self.active=False;self.result=None
        self.setWindowTitle('Resource probe');self.resize(880,760);layout=QVBoxLayout(self)
        note=QLabel('Measure disposable complete native updates of a saved study trial. The canvas and its trained results stay unchanged. Review freezes the saved input; no checkpoint or output is retained. Hiding this window leaves the probe running.')
        note.setWordWrap(True);layout.addWidget(note);form=QFormLayout();layout.addLayout(form)
        self.source=QLineEdit();self.source.setPlaceholderText('Saved .margam-study file');self.browse=QPushButton('Browse…');self.browse.setIcon(icon('open'));self.browse.clicked.connect(self.choose_source)
        row=QHBoxLayout();row.addWidget(self.source);row.addWidget(self.browse);form.addRow('Saved study',row)
        self.trial=QSpinBox();self.trial.setRange(1,512);self.trial.setToolTip('One-based design-point number in the saved Cartesian or paired study.');form.addRow('Design point',self.trial)
        self.updates=QSpinBox();self.updates.setRange(1,1000);self.updates.setToolTip('Complete updates/attempts, within the original horizon. Later balancing or sampling is measured only if this budget reaches it.');form.addRow('Probe updates / attempts',self.updates)
        self.limit=QSpinBox();self.limit.setRange(0,604800);self.limit.setValue(60);self.limit.setSpecialValueText('Unlimited');self.limit.setToolTip('Explicit probe child launch-to-exit seconds. Default 60; 0 unlimited. This does not change the application job-limit preference.');form.addRow('Probe time limit (seconds)',self.limit)
        self.device=QLabel();form.addRow('Application device',self.device)
        self.review_text=QPlainTextEdit();self.review_text.setReadOnly(True);self.review_text.setMaximumHeight(140);layout.addWidget(self.review_text)
        row=QHBoxLayout();layout.addLayout(row)
        self.review_button=QPushButton('Review');self.review_button.clicked.connect(self.review);row.addWidget(self.review_button)
        self.start_button=QPushButton('Run disposable probe');self.start_button.setIcon(icon('play'));self.start_button.clicked.connect(self.start);row.addWidget(self.start_button)
        self.stop_button=QPushButton('Stop / discard');self.stop_button.clicked.connect(self.stop);row.addWidget(self.stop_button)
        self.force_button=QPushButton('Force stop / discard');self.force_button.clicked.connect(lambda:self.stop(force=True));row.addWidget(self.force_button)
        self.progress=QProgressBar();self.progress.setRange(0,1);layout.addWidget(self.progress)
        self.status=QLabel('Choose a saved study and Review. CPU RAM includes imports; CUDA allocator peaks exclude driver/context memory. Outputs and later growth are not predicted.');self.status.setTextFormat(Qt.TextFormat.PlainText);self.status.setWordWrap(True);layout.addWidget(self.status)
        self.text=QPlainTextEdit();self.text.setReadOnly(True);self.text.setMaximumBlockCount(1000);layout.addWidget(self.text)
        close=QPushButton('Close');close.clicked.connect(self.close);layout.addWidget(close)
        for control in (self.source,self.trial,self.updates,self.limit):
            (control.textChanged if isinstance(control,QLineEdit) else control.valueChanged).connect(self.invalidate)
        self.buttons()

    def intent(self):return (self.source.text().strip(),self.trial.value(),self.updates.value(),self.owner.application_device,self.limit.value())
    def release_review(self):
        if self.prepared is not None:self.prepared['temporary'].cleanup()
        self.prepared=None
    def invalidate(self,*_):
        if self.active or self.reader is not None:return
        self.release_review();self.result=None;self.review_text.clear();self.text.clear();self.buttons()
    def buttons(self):
        busy=self.active or self.reader is not None;other=self.owner.is_running() and not self.active
        for control in (self.source,self.trial,self.updates,self.limit,self.browse):control.setEnabled(not busy)
        self.device.setText(self.owner.application_device+' · uses the application preference; changing it requires a new review')
        self.review_button.setEnabled(not busy and not other);self.start_button.setEnabled(not busy and not other and self.prepared is not None)
        self.stop_button.setEnabled(self.active and self.worker is not None and not self.worker.stop_requested);self.force_button.setEnabled(self.active)
    def choose_source(self):
        path,_=QFileDialog.getOpenFileName(self,'Choose saved study','','Margam study (*.margam-study)')
        if path:self.source.setText(path)
    def review(self):
        if self.active or self.reader is not None:return False
        self.invalidate()
        if os.name!='posix':self.status.setText('Desktop resource probes currently require POSIX process groups.');return False
        if self.owner.is_running() or self.owner.scene.locked or self.owner.project_workspace.busy:self.status.setText('Finish the current job or geometry edit first.');return False
        intent=self.intent()
        if not intent[0]:self.status.setText('Choose a saved study first.');return False
        source,trial,updates,device,limit=intent
        self.reader=Reader(lambda check:prepare_probe(source,trial,updates,device,limit,check),self);self.reader.finished.connect(self.review_finished)
        self.status.setText('Freezing saved inputs and validating the selected trial; no network allocated…');self.buttons()
        try:self.reader.start()
        except Exception as error:
            self.reader.deleteLater();self.reader=None;self.status.setText('Review could not start: '+str(error));self.buttons();return False
        return True
    def review_finished(self):
        reader=self.sender()
        if reader is None or reader is not self.reader:return
        self.reader=None
        if reader.error:self.status.setText('Review failed: '+reader.error)
        elif reader.abort_requested or reader.isInterruptionRequested() or reader.value['intent']!=self.intent():
            if reader.value is not None:reader.value['temporary'].cleanup()
            self.status.setText('Review cancelled or settings changed. Review again.')
        else:
            self.prepared=reader.value;description=self.prepared['description']
            self.review_text.setPlainText('Trial: '+self.prepared['trial_id']+'\n'+json.dumps(description,sort_keys=True)+'\nProbe budget: '+str(self.updates.value())+' · device: '+self.owner.application_device+' · limit: '+str(self.limit.value())+' s (0 unlimited)\nFrozen saved inputs; current canvas is not used. Original training horizon stays unchanged.')
            self.status.setText('Reviewed. Run executes this disposable snapshot. No full-run memory-fit prediction.')
        reader.value=None;reader.deleteLater();self.buttons()
    def start(self):
        if self.active or self.reader is not None or self.prepared is None:return False
        if self.owner.is_running() or self.owner.scene.locked or self.owner.project_workspace.busy:self.status.setText('Finish the current job or geometry edit first.');return False
        if self.prepared['intent']!=self.intent():self.invalidate();self.status.setText('Settings or application device changed. Review again.');return False
        _,trial,updates,device,limit=self.prepared['intent']
        args=['study','probe',str(self.prepared['path']),'--trial',str(trial),'--updates',str(updates),'--device',device,'--wall-time',str(limit)]
        self.active=True;self.result=None;self.worker=StudyProcess(args,None,self);self.worker.finished.connect(self.execution_finished);self.text.clear();self.progress.setRange(0,0)
        self.status.setText('Disposable probe running. Stop discards the measurement.');self.owner.set_busy(True);self.buttons()
        try:self.worker.start()
        except Exception as error:
            self.worker.deleteLater();self.worker=None;self.active=False;self.progress.setRange(0,1);self.release_review();self.owner.set_busy(False);self.status.setText('Probe could not start: '+str(error));self.buttons();return False
        return True
    def execution_finished(self):
        worker=self.sender()
        if worker is None or worker is not self.worker:return
        self.worker=None;self.active=False;self.result=None
        try:
            if worker.stop_requested:raise ValueError('Probe stopped; measurement discarded.')
            if worker.error or worker.returncode!=0:raise ValueError(worker.error or worker.tail.decode('utf-8',errors='replace') or 'Probe process failed.')
            value=json.loads(worker.tail.decode('utf-8'))
            if not isinstance(value,dict) or value.get('format')!='margam-resource-probe' or type(value.get('version')) is not int or value.get('version')!=1 or value.get('trial_id')!=self.prepared['trial_id'] or value.get('checkpoint_retained') is not False or value.get('capacity_prediction') is not False:raise ValueError('Invalid probe report; no measurement adopted.')
            intent=self.prepared['intent']
            if value['study_archive_sha256']!=self.prepared['archive_sha256'] or value['requested_updates']!=intent[2] or value['requested_device']!=intent[3] or type(value['completed_updates']) is not int or not 0<=value['completed_updates']<=intent[2]:raise ValueError('Probe report differs from the reviewed inputs/budget.')
            text=report_text(value);self.result=value;self.text.setPlainText(text);self.status.setText('Measured disposable work only. Canvas results preserved; no weights or numerical outputs retained.')
        except (ValueError,TypeError,KeyError,UnicodeError) as error:self.text.setPlainText(str(error)[-65536:]);self.status.setText('No measurement retained. See diagnostic below.')
        self.progress.setRange(0,1);self.progress.setValue(1 if self.result is not None else 0);worker.deleteLater();self.release_review();self.owner.set_busy(False);self.buttons()
        if self.owner._closing:QTimer.singleShot(0,self.owner.close)
    def stop(self,checked=False,*,force=False):
        if self.worker is not None:self.worker.request_stop(force);self.owner.cancel_button.setEnabled(False);self.status.setText('Stopping owned probe processes; measurement will be discarded.');self.buttons()
    def shutdown_review(self):
        if self.active:return
        if self.reader is not None:
            reader=self.reader;reader.abort_requested=True;reader.requestInterruption();reader.wait();reader.finished.disconnect(self.review_finished)
            if reader.value is not None:reader.value['temporary'].cleanup()
            reader.value=None;reader.deleteLater();self.reader=None
        self.release_review();self.buttons()
    def closeEvent(self,event):
        if not self.active:self.shutdown_review()
        super().closeEvent(event)
    def showEvent(self,event):super().showEvent(event);self.buttons()
