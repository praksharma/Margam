"""Capture the real desktop exporter with a saved study and explicit resources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from margam.ui.model_document import capture_model
    from margam.studies import catalogue
    from margam.study_document import capture,write
    from margam.slurm import read_job
    from margam.execution.study_runner import source_identity
    from examples.documentation import validate_public_text
    import torch
    root=Path(__file__).resolve().parents[1];image=root/'assets/screenshots/manual-slurm-export.png';manifest=image.with_suffix('.json')
    if not args.overwrite and (image.exists() or manifest.exists()):parser.error('Capture exists; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');window=Window(state_directory=False);old=Path.cwd()
    try:
        window.set_application_device('cpu');before=capture_model(window);undo=window.undo_stack.index();rng=torch.get_rng_state().clone()
        axis=next(item for item in catalogue(before) if item['parameter']=='iterations')
        study,files=capture(before,[dict(node_id=axis['node_id'],parameter='iterations',mode='candidates',text='[2,3]')],window.project_workspace,execution=dict(device='cpu',wall_time_seconds=0),scene=window.scene)
        with TemporaryDirectory(prefix='margam-slurm-capture-') as directory:
            os.chdir(directory);write('poisson-grid.margam-study',study,files)
            dialog=window.show_slurm_export();dialog.source.setText('poisson-grid.margam-study');dialog.output.setText('poisson-job')
            dialog.cpus.setValue(4);dialog.memory.setValue(8192);dialog.minutes.setValue(60)
            assert dialog.export_bundle(),dialog.status.text();assert read_job('poisson-job')['study_id']==study['study_id']
            assert before==capture_model(window) and undo==window.undo_stack.index() and torch.equal(rng,torch.get_rng_state())
            for _ in range(4):app.processEvents()
            for cls in (QLabel,QLineEdit):
                for widget in dialog.findChildren(cls):validate_public_text(widget.text(),'Slurm export capture')
            assert dialog.grab().save(str(image))
        os.chdir(old)
        record=dict(generator='examples.capture_slurm_export',scope='Real desktop data-only export; no training or cluster submission.',source_sha256=source_identity()['sha256'],capture_tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),file=image.name,sha256=hashlib.sha256(image.read_bytes()).hexdigest(),current_model_preserved=True,parent_rng_preserved=True)
        text=json.dumps(record,indent=2)+'\n';validate_public_text(text,'Slurm export evidence');manifest.write_text(text)
        print('Captured desktop Slurm export; saved bundle and model/RNG preservation verified.')
    finally:os.chdir(old);window.undo_stack.setClean();window.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
