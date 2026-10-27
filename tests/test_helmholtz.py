"""Paper Helmholtz contract and the existing Fourier/LRA desktop workflow."""
import os,subprocess,sys,unittest
from dataclasses import replace
import torch
from margam.templates.examples import example_config
from margam.templates.poisson import build_pinn
from margam.components.expressions import Expression

class HelmholtzTests(unittest.TestCase):
    def test_equation_domain_boundary_and_frequency_contract(self):
        torch.set_num_threads(1)
        config=example_config('helmholtz_lra_2d');config.validate()
        self.assertEqual(config.bounds,([-1.,-1.],[1.,1.]))
        self.assertEqual((config.method,config.network_type,config.source),('lra','fourier',1.))
        self.assertEqual(config.network_options['frequencies'],[1.,2.,3.,4.])
        self.assertEqual(config.layers,(32,32))
        reference=Expression('sin(pi*x)*sin(4*pi*y)',2,target=True)
        model=build_pinn(config)
        points=(2*torch.rand(137,2)-1).requires_grad_()
        self.assertLess(model.problem.residual(points,reference(points)).abs().max().item(),1e-4)
        for _,boundary,_ in model._constraints:
            self.assertLess(reference(boundary).abs().max().item(),1e-6)
        # A positive Helmholtz mass term, not the reaction-diffusion sign.
        other=replace(config,source=4.,pde_forcing='(4-17*pi*pi)*sin(pi*x)*sin(4*pi*y)')
        self.assertLess(build_pinn(other).problem.residual(points,reference(points)).abs().max().item(),1e-4)
        with self.assertRaisesRegex(ValueError,'non-negative'):replace(config,source=-1.).validate()

    def test_example_background_outputs_and_archive(self):
        code=r'''
import time
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch
from PySide6.QtWidgets import QApplication,QMessageBox
from margam.ui.app import Window
from margam.ui.run_document import load_run
from margam.templates.examples import example_config
from dataclasses import asdict
app=QApplication([]);w=Window(state_directory=False);w.show();app.processEvents()
def warning(*args):raise AssertionError(str(args[1:]))
QMessageBox.warning=warning
assert w.example_actions['helmholtz_lra_2d'] in w.examples_menu.actions()
assert w.example_actions['helmholtz_lra_2d'] not in w.examples_1d_menu.actions()
assert w.load_example('helmholtz_lra_2d') and w.validate_model()
expected=example_config('helmholtz_lra_2d');expected.validate()
assert asdict(w.scene.compile_config())==asdict(expected)
trainer=next(n for n in w.scene.nodes() if n.kind=='lra_method')
w.scene.clearSelection();trainer.setSelected(True);w.change('iterations',25)
compact=w.collapse_recipe(trainer);config=asdict(w.scene.compile_config())
with TemporaryDirectory() as directory:
    path=Path(directory)/'helmholtz.margam.json'
    assert w.save_model(path) and w.load_model(path)
    assert asdict(w.scene.compile_config())==config
    assert w.start_training()
    deadline=time.monotonic()+30
    while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
    assert w.worker is None and w.result is not None,w.log.toPlainText()
    assert w.result['steps']==25 and not w.output_views.entries
    comparison=next(n for n in w.scene.nodes() if n.kind=='comparison')
    output=w.result['outputs'][comparison.identifier]
    assert min(row[0] for row in output['points'])==-1. and max(row[1] for row in output['points'])==1.
    assert output['reference'] is not None and output['rmse'] is not None
    path=Path(directory)/'helmholtz.margam-run.pt'
    assert w.save_current_run(path)
    data=load_run(path);assert data['checkpoint']['network_spec']['type']=='fourier'
    assert w.open_saved_run(path) and not w.output_views.entries
    assert w.resume_training(5)
    deadline=time.monotonic()+30
    while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
    assert w.worker is None and w.result['steps']==30,w.log.toPlainText()
app.clipboard().clear();w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=80)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
