"""PDE point selection follows graph arrows, independently of BC losses."""
from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
import unittest
import torch
from margam.templates.poisson import PoissonConfig,BoundarySampling,build_pinn,sample_problem
from margam.components.collocation import collocation_rows


class PDECollocationTests(unittest.TestCase):
    def test_selected_coordinates_and_physical_residual(self):
        c=PoissonConfig(points=9,boundary_points=3,layers=(5,5),device='cpu')
        interior,sets=sample_problem(c)
        for mode in ('interior','boundary','both'):
            with self.subTest(mode=mode):
                c.pde_points=mode;model=build_pinn(c)
                expected=torch.tensor(collocation_rows(c,interior,sets),dtype=model.dtype)
                torch.testing.assert_close(model.collocation_point_sample,expected,rtol=0,atol=0)
                x=expected.detach().requires_grad_(True)
                residual=model.problem.residual(x,model.network(x))
                torch.testing.assert_close(model.compute_losses()['pde'],residual.square().mean())
                self.assertEqual(sum(len(points) for condition,points,target in model._constraints),sum(map(len,sets.values())))

    def test_adaptive_boundary_anchors_and_resume(self):
        from margam.components.adaptive_sampling import sampling_defaults
        for sampler in ('rar','importance'):
            c=PoissonConfig(shape='Interval',points=8,boundary_points=2,layers=(4,4),pde_points='both')
            c.adaptive_sampling=sampling_defaults(sampler);c.adaptive_sampling.update(interval=1,candidates=12,chunk=4)
            if sampler=='rar':c.adaptive_sampling.update(batch=2,maximum=20)
            full=build_pinn(c);anchors=full.collocation_point_sample[-2:].clone();full.train(4,0)
            part=build_pinn(c);part.train(2,0);checkpoint=part.training_state()
            resumed=build_pinn(c);resumed.load_training_state(checkpoint);resumed.train(2,0)
            self.assertEqual(full.loss_history,resumed.loss_history)
            for name,value in full.network.state_dict().items():torch.testing.assert_close(value,resumed.network.state_dict()[name],rtol=0,atol=0)
            if sampler=='importance':
                torch.testing.assert_close(full.collocation_point_sample[-2:],anchors,rtol=0,atol=0)
                self.assertTrue(torch.equal(full.sampling_controller.weights[-2:],torch.ones((2,1))))
            else:torch.testing.assert_close(full.collocation_point_sample[8:10],anchors,rtol=0,atol=0)

    def test_examples_have_explicit_both_configuration(self):
        from margam.templates.examples import EXAMPLE_PROBLEMS,example_config
        for identifier in EXAMPLE_PROBLEMS:
            with self.subTest(example=identifier):
                c=example_config(identifier)
                self.assertEqual(c.pde_points,'both')
                c.points=16;c.boundary_points=2;c.layers=(4,4);c.network_type='fcnn';c.network_options={};c.device='cpu'
                c.boundary_sampling={side:BoundarySampling(2,c.boundary_recipe(side).method,c.boundary_recipe(side).seed) for side in c.sides};c.expert_settings['chunks']=2
                model=build_pinn(c);model.train(1,0)
                self.assertTrue(torch.isfinite(model.total_loss))

    def test_volume_integral_requires_interior_quadrature(self):
        c=PoissonConfig(pde_points='both',loss_formulation='integral')
        with self.assertRaisesRegex(ValueError,'interior quadrature'):c.validate()

    def test_arrow_controls_selection_and_old_document(self):
        code=r'''
from tempfile import TemporaryDirectory
from pathlib import Path
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.model_document import capture_model,decode_document
from margam.templates.poisson import build_pinn
app=QApplication([]);w=Window(state_directory=False)
pde=next(n for n in w.scene.nodes() if n.kind=='pde')
boundary=next(n for n in w.scene.nodes() if n.kind=='boundary')
assert w.validate_model() and w.config.pde_points=='both'
both=build_pinn(w.config)
wire=next(wire for wire in w.scene.wires if wire.target.parentItem() is pde and wire.target.name=='boundary_points')
w.scene.wires.remove(wire);w.scene.removeItem(wire)
assert w.validate_model() and w.config.pde_points=='interior'
interior=build_pinn(w.config)
assert len(both.collocation_point_sample)>len(interior.collocation_point_sample)
# Old models have no extra PDE arrow and retain their original point selection.
legacy=capture_model(w);legacy['version']=25;legacy['config'].pop('pde_points')
config,_,_=decode_document(legacy);assert config.pde_points=='interior'
# Boundary data can feed the primary PDE port directly, with no interior sampler.
source=next(n for n in w.scene.nodes() if n.kind=='interior')
w.delete_block(source);w.scene.connect_nodes(boundary,pde,'input')
assert w.validate_model() and w.config.pde_points=='boundary'
model=build_pinn(w.config)
assert len(model.collocation_point_sample)==sum(len(points) for condition,points,target in model._constraints)
with TemporaryDirectory() as directory:
    path=Path(directory)/'boundary-only.margam.json'
    assert w.save_model(path) and w.load_model(path) and w.validate_model()
    assert w.config.pde_points=='boundary'
w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=40)
        self.assertEqual(result.returncode,0,result.stderr)
