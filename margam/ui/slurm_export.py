"""Desktop export of saved studies through the data-only Slurm bundle writer."""
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QFormLayout,QHBoxLayout,QLabel,
    QLineEdit,QPushButton,QSpinBox,QCheckBox,QFileDialog)
from ..slurm import export_job


class SlurmExport(QDialog):
    def __init__(self,window):
        super().__init__(window);self.setWindowTitle('Export study for Slurm');self.resize(700,480)
        layout=QVBoxLayout(self)
        note=QLabel('Export a saved study, not the current canvas or unsaved parameter draft. No training, installation, cluster connection or submission occurs. The generated script runs trials serially on one node.')
        note.setWordWrap(True);layout.addWidget(note);form=QFormLayout();layout.addLayout(form)
        self.source=QLineEdit();self.output=QLineEdit()
        for label,field,callback in [('Saved study',self.source,self.browse_source),('New bundle folder',self.output,self.browse_output)]:
            row=QHBoxLayout();row.addWidget(field);button=QPushButton('Browse…');button.clicked.connect(callback);row.addWidget(button);form.addRow(label,row)
        self.cpus=QSpinBox();self.cpus.setRange(0,1024)
        self.memory=QSpinBox();self.memory.setRange(0,16777216)
        self.minutes=QSpinBox();self.minutes.setRange(0,10080)
        for label,widget,tip in [('CPUs per task',self.cpus,'Required: requested CPU count; sets OMP/MKL thread budgets.'),('Memory (MiB)',self.memory,'Required: total requested node memory, not a measured memory estimate.'),('Batch time (minutes)',self.minutes,'Required: total batch allocation time, distinct from each saved numerical job limit.')]:
            widget.setSpecialValueText('Required');widget.setToolTip(tip);form.addRow(label,widget)
        self.gpu=QCheckBox('Request one GPU and execute on CUDA');self.gpu.setToolTip('Unchecked forces CPU. Checked requires CUDA at execution; no CPU fallback.');form.addRow('Execution',self.gpu)
        self.partition=QLineEdit();self.account=QLineEdit()
        for label,widget in [('Partition (optional)',self.partition),('Account (optional)',self.account)]:
            widget.setMaxLength(64);widget.setToolTip('One site-specific token: letters, digits, dot, underscore or hyphen; start with a letter or digit.');form.addRow(label,widget)
        self.exports=QCheckBox('Write files requested by enabled, connected export blocks');form.addRow('Output files',self.exports)
        note=QLabel('Choose a folder that does not exist. Review job.sbatch and job.json before submitting on your cluster. Budgets are requests, not fit predictions or enforced local quotas. Copy the complete results folder back for File → Open study results.')
        note.setWordWrap(True);layout.addWidget(note)
        self.status=QLabel('Enter the three resource budgets explicitly.');self.status.setTextFormat(Qt.TextFormat.PlainText);self.status.setWordWrap(True);layout.addWidget(self.status)
        row=QHBoxLayout();layout.addLayout(row);button=QPushButton('Export bundle');button.clicked.connect(self.export_bundle);row.addWidget(button)
        button=QPushButton('Close');button.clicked.connect(self.close);row.addWidget(button)

    def browse_source(self):
        path,_=QFileDialog.getOpenFileName(self,'Choose saved study',self.source.text(),'Margam study (*.margam-study)')
        if path:self.source.setText(path)

    def browse_output(self):
        path,_=QFileDialog.getSaveFileName(self,'Choose a new bundle folder',self.output.text(),'All names (*)',options=QFileDialog.Option.DontConfirmOverwrite)
        if path:self.output.setText(path)

    def export_bundle(self):
        if not self.source.text().strip() or not self.output.text().strip():
            self.status.setText('Choose a saved study and a new bundle folder.');return False
        try:
            job=export_job(self.source.text(),self.output.text(),cpus=self.cpus.value(),memory_mb=self.memory.value(),minutes=self.minutes.value(),gpu=self.gpu.isChecked(),partition=self.partition.text().strip() or None,account=self.account.text().strip() or None,exports=self.exports.isChecked())
        except (OSError,ValueError,TypeError,KeyError,RecursionError) as error:
            self.status.setText('Export failed: '+str(error));return False
        self.status.setText('Exported '+Path(self.output.text()).name+' · study '+job['study_id'][:16]+'. Nothing submitted. Review job.sbatch and job.json; transfer the whole folder to your cluster.')
        return True
