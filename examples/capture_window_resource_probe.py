"""Capture real Expert window/teacher phases through reviewed desktop controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--overwrite',action='store_true');p.add_argument('--device',choices=('cpu','cuda'),default='cpu');a=p.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit,QTableWidget
    from margam.ui.app import Window
    from margam.ui.theme import configure_theme
    from margam.ui.model_document import capture_model
    from margam.studies import catalogue
    from margam.study_document import capture,write
    from margam.execution.study_runner import source_identity
    from margam.execution.probe_report import validate_window_report
    from examples.documentation import validate_public_text
    import torch
    root=Path(__file__).resolve().parents[1];image=root/'assets/screenshots/manual-window-resource-probe.png';manifest=image.with_suffix('.json')
    if not a.overwrite and (image.exists() or manifest.exists()):p.error('Capture exists; use --overwrite.')
    app=QApplication([]);configure_theme(app,'light');w=Window(state_directory=False);old=Path.cwd();dialog=None
    try:
        w.set_application_device(a.device);w.load_example('expert_burgers')
        for key,value in [('points',16),('boundary_points',4),('layers',(8,8))]:w.change(key,value)
        options=dict(w.config.network_options);options['features']=4;w.change('network_options',options)
        for key,value in [('windows',3),('window_steps',4),('chunks',4),('balance_mode','ntk'),('balance_every',2),('ntk_points',2)]:w.change_expert_option(key,value)
        model=capture_model(w);axis=next(x for x in catalogue(model) if x['parameter']=='network_seed')
        study,files=capture(model,[dict(node_id=axis['node_id'],parameter='network_seed',mode='fixed',text='0')],w.project_workspace,execution=dict(device='cpu',wall_time_seconds=0),scene=w.scene)
        with TemporaryDirectory(prefix='margam-window-resource-capture-') as directory:
            os.chdir(directory);path=Path('expert-windows.margam-study');write(path,study,files);original=path.read_bytes();undo=w.undo_stack.index();cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone() if a.device=='cuda' else None
            dialog=w.show_resource_probe();dialog.source.setText(path.name);dialog.updates.setValue(9)
            def wait(predicate):
                deadline=time.monotonic()+90
                while not predicate() and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
                assert predicate(),dialog.status.text()
            assert dialog.review();wait(lambda:dialog.reader is None);assert dialog.prepared is not None,dialog.status.text()
            assert dialog.start();wait(lambda:not dialog.active);assert dialog.result is not None,dialog.text.toPlainText();validate_window_report(dialog.result)
            assert dialog.result['observed_transfers']==2 and dialog.phases.rowCount()==6
            assert model==capture_model(w) and undo==w.undo_stack.index() and path.read_bytes()==original and w.result is None
            assert torch.equal(cpu,torch.get_rng_state())
            if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
            assert dialog.copy_report() and json.loads(app.clipboard().text())==dialog.result
            dialog.resize(1220,980);dialog.phases.setMinimumHeight(235)
            for _ in range(4):app.processEvents()
            for cls in (QLabel,QLineEdit,QPlainTextEdit):
                for widget in dialog.findChildren(cls):validate_public_text(widget.toPlainText() if isinstance(widget,QPlainTextEdit) else widget.text(),'Window probe capture')
            for table in dialog.findChildren(QTableWidget):
                for r in range(table.rowCount()):
                    for c in range(table.columnCount()):
                        if table.item(r,c):validate_public_text(table.item(r,c).text(),'Window probe phase table')
            assert dialog.grab().save(str(image))
            record=dict(generator='examples.capture_window_resource_probe',generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_sha256=source_identity()['sha256'],file=image.name,sha256=hashlib.sha256(image.read_bytes()).hexdigest(),scope='Actual reviewed native Expert probe, nine of twelve updates, three windows and two reached handoffs. Phase peaks are not summed. Not solution accuracy, fit or paper speed.',current_model_preserved=True,parent_rng_preserved=True,study_unchanged=True,report=dialog.result)
            text=json.dumps(record,indent=2,allow_nan=False)+'\n';validate_public_text(text,'Window probe manifest');manifest.write_text(text)
        print('Captured real Expert window and teacher-transfer resource phases.')
    finally:
        if dialog is not None:
            if dialog.active:dialog.stop(force=True);wait(lambda:not dialog.active)
            dialog.shutdown_review()
        os.chdir(old);w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
