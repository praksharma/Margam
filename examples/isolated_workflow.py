"""Verify isolated desktop jobs and retained CPU checkpoints, without an accuracy claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=('cpu','cuda'),default='cpu');parser.add_argument('--output',type=Path);args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    import torch
    from PySide6.QtWidgets import QApplication
    from margam.ui.app import Window
    from margam.templates.poisson import build_pinn
    app=QApplication([]);w=Window(state_directory=False)
    def finish():
        deadline=time.monotonic()+45
        while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.002)
        if w.worker is not None:raise RuntimeError('Isolated job timed out.')
        if w.result is None:raise RuntimeError(w.log.toPlainText())
    try:
        if not w.set_application_device(args.device):raise RuntimeError('Requested device unavailable.')
        w.load_example('interval_poisson');w.change('iterations',3);w.change('points',8);w.change('layers',(4,))
        assert w.start_training();finish();first=w.last_run;handle=first.model
        assert first.child_pid!=os.getpid() and first.headless
        assert handle.storage_device=='cpu' and all(value.device.type=='cpu' for value in handle.network.state_dict().values())
        assert w.evaluate_trained_outputs();finish();assert w.last_run.model is handle and handle.iter==3
        with TemporaryDirectory() as directory:
            path=Path(directory)/'run.pt';assert w.save_current_run(path)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            w.undo_stack.setClean();assert w.open_saved_run(path);assert w.resume_training(2);finish()
            full=build_pinn(w.config);full.train(5,0)
            assert full.loss_history==w.last_run.model.loss_history
            assert all(torch.equal(value.cpu(),w.last_run.model.network.state_dict()[key]) for key,value in full.network.state_dict().items())
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        report=dict(scope='Actual isolated desktop lifecycle; not an accuracy or memory-capacity benchmark.',
            device=args.device,torch_version=str(torch.__version__),separate_process=True,headless_child=True,
            retained_tensor_device='cpu',initial_steps=3,resumed_steps=5,exact_resume=True,
            evaluation_preserved_handle=True,previous_saved_file_unchanged=True)
        text=json.dumps(report,indent=2)+'\n'
        if args.output is not None:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
        print(text,end='')
    finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
