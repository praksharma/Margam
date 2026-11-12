"""Make the trained Poisson accuracy checkpoint reviewable in the desktop."""
import os,time,json
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication,QMessageBox
from margam.ui.app import Window
from margam.ui.graph_model import PARAMETERS,parameters_for
from margam.templates.examples import example_config
from margam.execution.codec import load_packet


def main():
    root=Path('build/dsgnar-accuracy');report=json.loads(Path('assets/benchmarks/dsgnar-poisson-accuracy.json').read_text())
    state=load_packet(root/'poisson.pt');c=example_config('dsgnar_poisson_1d');c.device='cpu';c.network_seed=c.seed
    c.points=report['points'];c.layers=tuple(report['layers']);c.pde_weight=1.;c.boundary_weight=1.;c.dsgnar_settings=report['settings'];c.iterations=state['iter']
    def fail(parent,title,message,*args):raise RuntimeError(str(title)+': '+str(message))
    QMessageBox.warning=QMessageBox.critical=fail
    app=QApplication([]);w=Window(state_directory=False)
    try:
        w.set_application_device('cpu');w.load_example('dsgnar_poisson_1d');w.config=c
        for node in w.scene.nodes():
            if node.kind in PARAMETERS:node.settings['parameters']=parameters_for(node.kind,c)
        w.settings_changed();assert w.validate_model(),w.validation.text()
        from dataclasses import asdict
        assembled=asdict(w.scene.compile_config())
        differences={k:(value,assembled.get(k)) for k,value in state['dsgnar']['configuration'].items() if k not in ('device','iterations') and value!=assembled.get(k)}
        assert not differences,repr(differences)
        assert w.start_training(checkpoint=state)
        deadline=time.monotonic()+45
        while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.003)
        assert w.worker is None and w.result is not None,w.log.toPlainText()
        assert w.save_model(root/'poisson.margam.json') and w.save_current_run(root/'poisson.margam-run.pt')
        w.undo_stack.setClean();assert w.open_saved_run(root/'poisson.margam-run.pt')
        print('Saved and reopened trained Poisson project and run archive.',flush=True)
    finally:w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()


if __name__=='__main__':main()
