"""Manufactured equations, convergence, and the example/PDE desktop contract."""
import os
import subprocess
import sys
import unittest
import torch
from margam.templates.examples import EXAMPLE_PROBLEMS, example_config
from margam.templates.poisson import build_pinn
from margam.components.expressions import Expression


class ExampleTests(unittest.TestCase):
    def test_manufactured_equations_and_training(self):
        torch.set_num_threads(1)
        for key, example in EXAMPLE_PROBLEMS.items():
            with self.subTest(example=key):
                if not example.reference or key in ('burgers_1d','dsgnar_burgers','dsgnar_cavity','dsgnar_wave','dsgnar_multiscale') or key.startswith(('expert_','scale_','gpinn_','rba_','fbpinn_')):continue # Dedicated transient residual/IC tests; no manufactured exact profile.
                config=example_config(key)
                model=build_pinn(config)
                dimensions=len(config.bounds[0])
                lower,upper=(torch.tensor(v) for v in config.bounds)
                points=(lower+(upper-lower)*torch.rand(23,dimensions)).requires_grad_()
                reference=Expression(example.reference,dimensions,target=True,coordinates=config.coordinate_names)
                self.assertLess(model.problem.residual(points,reference(points)).abs().max().item(),1e-4)
                model.train(2000 if key=='helmholtz_lra_2d' else 700,display_every=0)
                points=lower+(upper-lower)*torch.rand(100,dimensions)
                self.assertLess((model.predict(points)-reference(points)).square().mean().sqrt().item(),.05)

    def test_preset_validation_and_legacy_defaults(self):
        from dataclasses import asdict
        from margam.templates.poisson import PoissonConfig
        from margam.ui.model_document import decode_document
        config=example_config('advection_diffusion')
        config.shape='Rectangle'
        with self.assertRaisesRegex(ValueError,'Interval'):config.validate()
        config.shape='Interval';config.viscosity=0.
        with self.assertRaisesRegex(ValueError,'positive diffusivity'):config.validate()
        config.viscosity=.1;config.pde_forcing='u'
        with self.assertRaises(ValueError):config.validate()
        for version in range(1,7):
            values=asdict(PoissonConfig())
            values.pop('pde_preset');values.pop('pde_forcing')
            restored,_,_=decode_document(dict(format='margam-model',version=version,config=values,nodes=[],wires=[]))
            self.assertEqual(restored.pde_preset,'')
            self.assertEqual(restored.pde_forcing,'0')

    def test_example_menu_and_alternative_pde_roundtrip(self):
        code='''
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.model_document import CURRENT_VERSION, capture_model, decode_document
from margam.templates.examples import EXAMPLE_PROBLEMS
app=QApplication([])
w=Window(state_directory=False)
assert w.examples_menu.title()=='Examples'
assert len(w.example_actions)==len(EXAMPLE_PROBLEMS)
assert w.examples_1d_menu.title()=='1D examples'
assert len(w.examples_1d_menu.actions())==9
assert w.evaluation_example_action in w.examples_1d_menu.actions()
assert w.burgers_reference_action in w.examples_1d_menu.actions()
from margam.templates.examples import example_config
for key,action in w.example_actions.items():
    ex_cfg=example_config(key)
    expected=w.dsgnar_examples_menu if key.startswith('dsgnar_') else w.fbpinn_examples_menu if key.startswith('fbpinn_') else w.rba_examples_menu if key.startswith('rba_') else w.gpinn_examples_menu if key.startswith('gpinn_') else w.scale_examples_menu if key.startswith('scale_') else w.expert_examples_menu if key.startswith('expert_') else w.examples_parametric_menu if ex_cfg.parametric_axes else w.examples_3d_menu if ex_cfg.is_3d else w.examples_1d_menu if ex_cfg.shape in ('Interval','SpaceTime1D') else w.examples_menu
    assert action in expected.actions()
    if expected in (w.examples_1d_menu,w.examples_3d_menu,w.examples_parametric_menu):assert action not in w.examples_menu.actions()
assert all(a.text().startswith('2D ') for a in (w.load_action,w.heat_action,w.flow_action))
for key in EXAMPLE_PROBLEMS:
    assert w.load_example(key)
    config=w.scene.compile_config()
    config.validate()
    expected_pde='pde' if (config.physics=='incompressible_2d' or not config.pde_preset) else 'predefined_pde'
    assert sum(n.kind==expected_pde for n in w.scene.nodes())==1,key
    assert not any(n.kind==('predefined_pde' if expected_pde=='pde' else 'pde') for n in w.scene.nodes()),key
    assert not w.output_views.entries
    document=capture_model(w)
    decode_document(document)
    assert document['version']==CURRENT_VERSION
    from tempfile import TemporaryDirectory
    from pathlib import Path
    with TemporaryDirectory() as directory:
        path=Path(directory)/'example.margam.json'
        assert w.save_model(path)
        assert w.load_model(path)
        assert capture_model(w)==document
        assert w.scene.compile_config().pde_preset==config.pde_preset
w.load_example('reaction_diffusion')
w.change('source',2.)
assert w.scene.compile_config().source==2.
w.undo_stack.undo()
assert w.scene.compile_config().source==1.
w.reset_template()
assert any(n.kind=='pde' for n in w.scene.nodes())
assert not w.scene.compile_config().pde_preset
w.undo_stack.setClean();w.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=40)
        self.assertEqual(result.returncode,0,result.stderr)
