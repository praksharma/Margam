"""Local serial study control through the existing headless CLI, on POSIX systems."""
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Lock
import time
from PySide6.QtCore import QThread,Signal,QTimer,Qt
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,QPushButton,
    QLineEdit,QComboBox,QCheckBox,QSpinBox,QProgressBar,QPlainTextEdit,QFileDialog)
from .study_results import Reader
from .icons import icon
from ..execution.study_state import read_json,file_identity


def prepare_review(mode,source,output,check):
    """Review immutable inputs without constructing networks or changing the model."""
    if check():raise InterruptedError('Review cancelled.')
    if mode=='resume':
        from ..study_results import collect
        identity=file_identity(output,'execution.json')['sha256'];value=collect(output,check=check)
        if file_identity(output,'execution.json')['sha256']!=identity:raise ValueError('Execution settings changed during review.')
        return dict(temporary=None,metadata=value['report'],execution_sha256=identity)
    from ..execution.study_runner import snapshot,inspect
    temporary=TemporaryDirectory(prefix='margam-study-review-')
    try:
        frozen=Path(temporary.name)/'study.margam-study';digest=snapshot(source,frozen);metadata=inspect(frozen)
        if check():raise InterruptedError('Review cancelled.')
        return dict(temporary=temporary,path=frozen,metadata=metadata,archive_sha256=digest)
    except BaseException:temporary.cleanup();raise


def reported_progress(output):
    value=read_json(output,'summary.json')
    if not isinstance(value,dict):raise ValueError('Invalid study progress record.')
    if value.get('format')!='margam-study-results' or value.get('version')!=2 or not isinstance(value.get('trials'),list) or not 1<=len(value['trials'])<=512:raise ValueError('Invalid study progress record.')
    counts=value['counts']
    if not isinstance(counts,dict) or any(type(number) is not int or number<0 for number in counts.values()) or sum(counts.values())!=len(value['trials']):raise ValueError('Invalid study progress counts.')
    active=[f"DP{row['index']+1:04d}: {row['status']} · {row.get('stage') or 'waiting'}" for row in value['trials'] if row['status'] in ('preparing','running')]
    return dict(state=value['state'],counts=counts,total=len(value['trials']),active='; '.join(active),study_id=value['study_id'])


class StudyProcess(QThread):
    progress=Signal(object)
    def __init__(self,args,output,parent=None):
        super().__init__(parent);self.args=args;self.output=output;self.process=None;self.returncode=None;self.error=None;self.tail=b'';self.stop_requested=False;self.force_requested=False;self.interrupt_sent=False;self.guard=Lock();self.temp_root=None
    def request_stop(self,force=False):
        with self.guard:
            self.stop_requested=True;self.force_requested|=force
            process=self.process
            if process is not None and process.poll() is None:
                try:
                    if self.force_requested:os.killpg(process.pid,signal.SIGKILL)
                    elif not self.interrupt_sent:process.send_signal(signal.SIGINT);self.interrupt_sent=True
                except ProcessLookupError:pass
    def report(self):
        if self.output is None:return
        try:self.progress.emit(reported_progress(self.output))
        except (OSError,ValueError,KeyError,TypeError):pass
    def run(self):
        process=None;temporary=None
        try:
            temporary=TemporaryDirectory(prefix='margam-local-study-');self.temp_root=temporary.name
            environment=dict(os.environ,TMPDIR=temporary.name)
            process=subprocess.Popen([sys.executable,'-m','margam.cli',*self.args,'--watch-owner-stdin'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,env=environment)
            with self.guard:self.process=process;stopping=self.stop_requested;force=self.force_requested
            if stopping:self.request_stop(force)
            deadline=0.
            while True:
                ready,_,_=select.select([process.stdout],[],[],.2)
                if ready:
                    chunk=os.read(process.stdout.fileno(),8192)
                    if chunk:self.tail=(self.tail+chunk)[-65536:]
                    elif process.poll() is not None:break
                if time.monotonic()>=deadline:self.report();deadline=time.monotonic()+1.
                if process.poll() is not None and not ready:break
            self.returncode=process.wait();self.report()
        except Exception as error:self.error=str(error)
        finally:
            if process is not None:
                if process.poll() is None:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                process.wait()
                if process.stdout is not None:process.stdout.close()
                if process.stdin is not None:process.stdin.close()
            if temporary is not None:
                try:temporary.cleanup()
                except OSError as error:self.error=(self.error or '')+' Temporary cleanup failed: '+str(error)


class StudyExecution(QDialog):
    def __init__(self,owner):
        super().__init__(owner);self.owner=owner;self.worker=None;self.reader=None;self.prepared=None;self.active=False
        self.setWindowTitle('Run a saved study');self.resize(900,640);layout=QVBoxLayout(self)
        note=QLabel('Run a saved portable study serially in isolated processes. The current canvas and retained results stay unchanged. Review before starting. Stop discards the active trial; completed trials remain available.')
        note.setWordWrap(True);layout.addWidget(note)
        form=QFormLayout();layout.addLayout(form)
        self.mode=QComboBox();self.mode.addItem('Fresh execution','run');self.mode.addItem('Resume existing results','resume');form.addRow('Operation',self.mode)
        self.source=QLineEdit();self.source.setPlaceholderText('Saved .margam-study file');self.source_button=QPushButton('Browse study…');self.source_button.clicked.connect(self.choose_source)
        row=QHBoxLayout();row.addWidget(self.source);row.addWidget(self.source_button);form.addRow('Saved study',row)
        self.output=QLineEdit();self.output.setPlaceholderText('New results folder, or existing folder for resume');self.output_button=QPushButton('Browse folder…');self.output_button.clicked.connect(self.choose_output)
        row=QHBoxLayout();row.addWidget(self.output);row.addWidget(self.output_button);form.addRow('Results folder',row)
        self.device=QComboBox()
        for name,key in [('Use saved intent',None),('Automatic','auto'),('CPU','cpu'),('GPU / CUDA','cuda')]:self.device.addItem(name,key)
        form.addRow('Device override',self.device)
        self.limit_override=QCheckBox('Override saved per-job limit');self.limit=QSpinBox();self.limit.setRange(0,604800);self.limit.setSpecialValueText('Unlimited')
        row=QHBoxLayout();row.addWidget(self.limit_override);row.addWidget(self.limit);form.addRow('Seconds',row)
        self.exports=QCheckBox('Write enabled CSV / ParaView Export blocks');form.addRow(self.exports)
        self.retry=QCheckBox('Also retry failed trials (resume only)');form.addRow(self.retry)
        note=QLabel('Fresh execution uses the saved study device/limit unless overridden above; application preferences are unchanged. Resume keeps the original execution settings. Full preflight runs per trial; review is not a memory-fit or accuracy prediction.')
        note.setWordWrap(True);layout.addWidget(note)
        self.review_text=QPlainTextEdit();self.review_text.setReadOnly(True);self.review_text.setMaximumHeight(140);layout.addWidget(self.review_text)
        row=QHBoxLayout();layout.addLayout(row)
        self.review_button=QPushButton('Review');self.review_button.clicked.connect(self.review);row.addWidget(self.review_button)
        self.start_button=QPushButton('Start study');self.start_button.setIcon(icon('play'));self.start_button.clicked.connect(self.start);row.addWidget(self.start_button)
        self.stop_button=QPushButton('Stop study');self.stop_button.clicked.connect(self.stop);row.addWidget(self.stop_button)
        self.force_button=QPushButton('Force stop / discard');self.force_button.clicked.connect(lambda:self.stop(force=True));row.addWidget(self.force_button)
        self.progress=QProgressBar();layout.addWidget(self.progress)
        self.status=QLabel('Choose the saved study and a fresh results folder, then Review.');self.status.setTextFormat(Qt.TextFormat.PlainText);self.status.setWordWrap(True);layout.addWidget(self.status)
        self.log=QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumBlockCount(600);layout.addWidget(self.log)
        row=QHBoxLayout();layout.addLayout(row)
        self.results_button=QPushButton('Open results browser');self.results_button.clicked.connect(self.open_results);row.addWidget(self.results_button);row.addStretch()
        close=QPushButton('Close');close.clicked.connect(self.close);row.addWidget(close)
        self.inputs=[self.mode,self.source,self.output,self.device,self.limit_override,self.limit,self.exports,self.retry]
        for control in self.inputs:
            event=control.currentIndexChanged if isinstance(control,QComboBox) else control.textChanged if isinstance(control,QLineEdit) else control.valueChanged if isinstance(control,QSpinBox) else control.toggled
            event.connect(self.invalidate)
        self.buttons()

    def release_review(self):
        if self.prepared is not None and self.prepared['temporary'] is not None:self.prepared['temporary'].cleanup()
        self.prepared=None
    def invalidate(self,*_):
        if self.active or self.reader is not None:return
        self.release_review();self.review_text.clear();self.buttons()
    def buttons(self):
        busy=self.active or self.reader is not None;resume=self.mode.currentData()=='resume'
        for control in self.inputs:control.setEnabled(not busy)
        self.source.setEnabled(not busy and not resume);self.source_button.setEnabled(not busy and not resume);self.output_button.setEnabled(not busy)
        for control in (self.device,self.limit_override,self.exports):control.setEnabled(not busy and not resume)
        self.limit.setEnabled(not busy and not resume and self.limit_override.isChecked());self.retry.setEnabled(not busy and resume)
        self.review_button.setEnabled(not busy);self.start_button.setEnabled(not busy and self.prepared is not None)
        self.stop_button.setEnabled(self.active and self.worker is not None and not self.worker.stop_requested);self.force_button.setEnabled(self.active)
        self.results_button.setEnabled(not busy and bool(self.output.text().strip()) and (Path(self.output.text())/'summary.json').is_file())
    def choose_source(self):
        path,_=QFileDialog.getOpenFileName(self,'Choose a saved portable study','','Margam study (*.margam-study)')
        if path:self.source.setText(path)
    def choose_output(self):
        path=QFileDialog.getExistingDirectory(self,'Choose existing results' if self.mode.currentData()=='resume' else 'Choose parent for a NEW results folder')
        if path:self.output.setText(path if self.mode.currentData()=='resume' else str(Path(path)/'study-results'))
    def review(self):
        if self.active or self.reader is not None:return False
        self.invalidate()
        if os.name!='posix':self.status.setText('Local study control currently requires POSIX process groups.');return False
        output=self.output.text().strip();source=self.source.text().strip();mode=self.mode.currentData()
        if not output or mode=='run' and not source:self.status.setText('Choose the saved study and results folder.');return False
        if mode=='run' and (Path(output).exists() or Path(output).is_symlink()):self.status.setText('Fresh execution requires a new results folder.');return False
        self.reader=Reader(lambda check:prepare_review(mode,source,output,check),self);self.reader.finished.connect(self.review_finished)
        self.status.setText('Reviewing frozen study metadata…');self.buttons();self.reader.start();return True
    def review_finished(self):
        reader=self.sender()
        if reader is None or reader is not self.reader:return
        self.reader=None
        if reader.error:self.status.setText('Review failed: '+reader.error)
        elif reader.abort_requested:
            if reader.value is not None and reader.value['temporary'] is not None:reader.value['temporary'].cleanup()
        else:
            self.prepared=reader.value;metadata=self.prepared['metadata']
            if self.mode.currentData()=='run':
                self.review_text.setPlainText(f"Study: {metadata['study_id']}\nDesign points: {metadata['trials']}\nSaved device: {metadata['execution']['device']} · saved per-job limit: {metadata['execution']['wall_time_seconds']} s\nSelected device override: {self.device.currentText()} · explicit exports: {self.exports.isChecked()}\nFrozen input snapshots: {metadata['inputs']} ({metadata['input_bytes']} bytes)")
            else:self.review_text.setPlainText(f"Study: {metadata['study_id']}\nReported state: {metadata['state']}\nCounts: {json.dumps(metadata['counts'])}\nRecorded execution: {json.dumps(metadata['effective_execution'])}\nUnfinished trials restart fresh. Retry failed: {self.retry.isChecked()}")
            self.status.setText('Reviewed. Start executes this saved snapshot; the canvas is not used.')
        reader.value=None;reader.deleteLater();self.buttons()
    def start(self):
        if self.active or self.reader is not None or self.prepared is None:return False
        if self.owner.is_running() or self.owner.scene.locked or self.owner.project_workspace.busy:self.status.setText('Finish the current job or geometry edit first.');return False
        output=Path(self.output.text().strip()).absolute()
        if self.mode.currentData()=='run':
            if output.exists() or output.is_symlink():self.status.setText('Fresh execution requires a new results folder.');return False
            args=['study','run',str(self.prepared['path']),'--output',str(output)]
            if self.device.currentData() is not None:args+=['--device',self.device.currentData()]
            if self.limit_override.isChecked():args+=['--wall-time',str(self.limit.value())]
            if self.exports.isChecked():args+=['--export']
        else:
            args=['study','resume',str(output),'--expected-execution-sha256',self.prepared['execution_sha256']]
            if self.retry.isChecked():args+=['--retry-failed']
        self.active=True;self.worker=StudyProcess(args,output,self);self.worker.progress.connect(self.update_progress);self.worker.finished.connect(self.execution_finished)
        self.log.clear();self.progress.setRange(0,0);self.status.setText('Starting the headless serial runner…');self.owner.statusBar().showMessage('Study running · Run → Run saved study reopens its controls.');self.owner.set_busy(True);self.buttons();self.worker.start();return True
    def update_progress(self,value):
        if self.sender() is None or self.sender() is not self.worker:return
        self.progress.setRange(0,value['total']);self.progress.setValue(sum(value['counts'].get(key,0) for key in ('completed','failed','invalid','incompatible')))
        self.status.setText(value['state']+' · '+', '.join(f'{number} {key}' for key,number in sorted(value['counts'].items()))+(' · '+value['active'] if value['active'] else ''))
    def execution_finished(self):
        worker=self.sender()
        if worker is None or worker is not self.worker:return
        self.worker=None;self.active=False
        self.log.setPlainText(worker.tail.decode('utf-8',errors='replace')+(('\n'+worker.error) if worker.error else ''))
        messages={0:'Study completed. Open the results browser to inspect trials.',1:'Study finished with trial issues. Open results for individual reasons.',130:'Study interrupted. Resume restarts unfinished trials fresh.',2:'Study could not start or verify its inputs. See the execution log.'}
        self.status.setText('Study stopped forcefully; verify/restart unfinished trials with Resume.' if worker.force_requested else messages.get(worker.returncode,'Study process exited unexpectedly. See the execution log.'))
        self.owner.statusBar().showMessage(self.status.text()+' Canvas results preserved.');self.release_review()
        if self.progress.maximum()==0:self.progress.setRange(0,1);self.progress.setValue(0)
        worker.deleteLater();self.owner.set_busy(False);self.buttons()
        if self.owner._closing:QTimer.singleShot(0,self.owner.close)
    def stop(self,checked=False,*,force=False):
        if self.worker is not None:
            self.worker.request_stop(force);self.owner.cancel_button.setEnabled(False);self.buttons()
            self.status.setText('Force-stopping the owned study process group…' if force else 'Stopping the active trial and recording grid progress…')
    def open_results(self):
        if self.active or self.reader is not None:return False
        return self.owner.show_study_results().open_folder(self.output.text().strip())
    def shutdown_review(self):
        if self.reader is not None:
            reader=self.reader;reader.abort_requested=True;reader.requestInterruption();reader.wait();reader.finished.disconnect(self.review_finished)
            if reader.value is not None and reader.value['temporary'] is not None:reader.value['temporary'].cleanup()
            reader.value=None;reader.deleteLater();self.reader=None
        self.release_review();self.buttons()
    def closeEvent(self,event):
        if not self.active:self.shutdown_review()
        super().closeEvent(event)
