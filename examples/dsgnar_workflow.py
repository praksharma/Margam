"""Actual small DSGNAR desktop runs; correctness evidence, not paper reproduction."""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--screenshot',type=Path)
    args=parser.parse_args();os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    import torch
    torch.set_num_threads(1)
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit,QMessageBox
    from margam.ui.app import Window
    from margam.templates.poisson import build_pinn
    from examples.documentation import validate_public_text
    def fail(parent,title,message,*args):raise RuntimeError(str(title)+': '+str(message))
    QMessageBox.critical=QMessageBox.warning=fail
    app=QApplication([]);w=Window(state_directory=False)
    def finish():
        deadline=time.monotonic()+90
        while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.003)
        if w.worker is not None or w.result is None:raise RuntimeError(w.log.toPlainText())
    reports=[]
    try:
        assert w.set_application_device(args.device)
        for key in ('dsgnar_poisson_1d','dsgnar_burgers'):
            assert w.load_example(key) and w.validate_model()
            w.change('iterations',4)
            for node in w.scene.nodes():
                if node.kind in ('solution','comparison','export','paraview'):node.settings['resolution']=5
            with TemporaryDirectory(prefix='margam-dsgnar-') as directory:
                directory=Path(directory);assert w.save_model(directory/'model.json');w.undo_stack.setClean();assert w.load_model(directory/'model.json')
                assert w.start_training();finish();first=w.last_run
                assert first.headless and first.child_pid!=os.getpid()
                assert w.save_current_run(directory/'run.pt');w.undo_stack.setClean();assert w.open_saved_run(directory/'run.pt')
                assert w.resume_training(2);finish()
                retained=w.last_run.model;checkpoint=retained.training_state()
                direct=build_pinn(w.config);direct.train(6,0)
                # Exact continuation compares the same fresh-process executor.
                # The Qt host can have a different CUDA library/plan history;
                # compare its direct-engine prediction parity separately.
                from margam.execution.codec import encode_config
                from margam.execution.transport import JobTransport
                full=JobTransport().run(dict(operation='train',config=encode_config(w.config),requests=[],checkpoint=None,metadata={},dtype=w.config.dsgnar_settings['precision']))['checkpoint']
                assert full['loss_history']==retained.loss_history
                for a,b in zip(full['dsgnar']['diagnostics'],checkpoint['dsgnar']['diagnostics']):
                    assert {k:v for k,v in a.items() if k!='regularization'}=={k:v for k,v in b.items() if k!='regularization'}
                torch.testing.assert_close(torch.tensor(full['dsgnar']['lambda_history'],dtype=torch.float64),torch.tensor(checkpoint['dsgnar']['lambda_history'],dtype=torch.float64),rtol=5e-14,atol=0.)
                assert torch.equal(full['dsgnar']['generator'],checkpoint['dsgnar']['generator'])
                for name,value in full['network'].items():torch.testing.assert_close(value,retained.network.state_dict()[name],rtol=0,atol=0)
                for name,value in direct.network.state_dict().items():torch.testing.assert_close(value.cpu(),retained.network.state_dict()[name],rtol=1e-11,atol=1e-12)
                assert w.evaluate_trained_outputs();finish()
                export=next(n for n in w.scene.nodes() if n.kind=='export');assert w.export_block_data(export,directory/'solution.csv')
                if key=='dsgnar_burgers':
                    export=next(n for n in w.scene.nodes() if n.kind=='paraview');assert w.export_block_data(export,directory/'solution.pvd')
                summary=retained.dsgnar_summary();raw={name:value.item() for name,value in direct.compute_losses().items()}
                reports.append(dict(example=key,device=args.device,precision=w.config.dsgnar_settings['precision'],
                    separate_headless_process=True,checkpoint_exact_resume=True,evaluation_preserved_weights=True,regularization_diagnostic_tolerance=5e-14,
                    export_verified=True,raw_final_mse=raw,**summary))
                if args.screenshot and key=='dsgnar_poisson_1d':
                    trainer=next(n for n in w.scene.nodes() if n.kind=='dsgnar_method')
                    w.resize(1850,1100);w.properties_dock.setMinimumWidth(550);w.show();w.show_block_properties(trainer);w.fit_model();app.processEvents()
                    for cls in (QLabel,QLineEdit,QPlainTextEdit):
                        for child in w.findChildren(cls):
                            if child.isVisible():validate_public_text(child.toPlainText() if isinstance(child,QPlainTextEdit) else child.text(),'DSGNAR workflow capture')
                    args.screenshot.parent.mkdir(parents=True,exist_ok=True);assert w.grab().save(str(args.screenshot))
        report=dict(scope='Six-attempt desktop lifecycle checks; not paper accuracy, speed or memory reproduction.',cases=reports)
        text=json.dumps(report,indent=2,allow_nan=False)+'\n'
        if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
        print(text,end='')
    finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
