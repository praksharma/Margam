"""Native continuous parametric PINNs: config, joint sampling, Hard BC, persistence, and live Explorer sliders."""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from margam.components.expressions import Expression
from margam.components.output_export import export_data
from margam.components.outputs import OutputRequest, evaluate_outputs, evaluation_points
from margam.templates.examples import example_config
from margam.templates.poisson import PoissonConfig, build_pinn, sample_problem, validate_parametric_axes


class ContinuousParametricPINNTests(unittest.TestCase):
    def test_parametric_axes_validation_and_sampling(self):
        axes = validate_parametric_axes([
            {'name': 'mu', 'min': 0.5, 'max': 2.0, 'default': 1.0},
            {'name': 'alpha', 'min': -1.0, 'max': 1.0, 'default': 0.0},
        ])
        self.assertEqual(len(axes), 2)
        self.assertEqual(axes[0]['name'], 'mu')

        for bad in (
            [{'name': 'x', 'min': 0.0, 'max': 1.0, 'default': 0.5}],
            [{'name': 'pi', 'min': 0.0, 'max': 1.0, 'default': 0.5}],
            [{'name': 'mu', 'min': 2.0, 'max': 1.0, 'default': 1.5}],
            [{'name': 'mu', 'min': 0.5, 'max': 2.0, 'default': 3.0}],
            [
                {'name': 'mu', 'min': 0.0, 'max': 1.0, 'default': 0.5},
                {'name': 'mu', 'min': 1.0, 'max': 2.0, 'default': 1.5},
            ],
            [{'name': f'p{i}', 'min': 0.0, 'max': 1.0, 'default': 0.5} for i in range(5)],
        ):
            with self.assertRaises(ValueError):
                validate_parametric_axes(bad)

        # Interval + 1 parameter
        cfg_1d = PoissonConfig(
            shape='Interval',
            points=16,
            boundary_points=8,
            sampling='hammersley',
            parametric_axes=({'name': 'mu', 'min': 0.5, 'max': 2.0, 'default': 1.0},),
            pde_expression='mu * laplace(u) + 2',
        )
        cfg_1d.validate()
        self.assertEqual(cfg_1d.coordinate_names, ('x', 'mu'))
        self.assertEqual(cfg_1d.spatial_time_bounds, ([0.0], [1.0]))
        self.assertEqual(cfg_1d.bounds, ([0.0, 0.5], [1.0, 2.0]))
        interior, sets = sample_problem(cfg_1d)
        self.assertEqual(len(interior[0]), 2)
        self.assertEqual(len(sets['left']), 8)
        self.assertTrue(all(abs(r[0] - 0.0) < 1e-12 and 0.5 <= r[1] <= 2.0 for r in sets['left']))
        eval_1d = evaluation_points(cfg_1d, 11)
        self.assertEqual(len(eval_1d), 11)
        self.assertTrue(all(abs(r[1] - 1.0) < 1e-12 for r in eval_1d))

        # 3D sphere obstacle + 1 parameter
        cfg_3d = PoissonConfig(
            shape='SphereObstacle3D',
            points=32,
            boundary_points=10,
            sampling='random',
            parametric_axes=({'name': 'kappa', 'min': 1.0, 'max': 4.0, 'default': 2.0},),
            pde_expression='kappa * laplace(u)',
        )
        cfg_3d.validate()
        self.assertEqual(cfg_3d.coordinate_names, ('x', 'y', 'z', 'kappa'))
        interior_3d, sets_3d = sample_problem(cfg_3d)
        self.assertEqual(len(interior_3d[0]), 4)
        self.assertEqual(len(sets_3d['sphere'][0]), 4)

    def test_parametric_poisson_2d_exact_residual_training_and_export(self):
        torch.set_num_threads(1)
        cfg = example_config('parametric_poisson_2d')
        cfg.points = 128
        cfg.boundary_points = 16
        cfg.layers = (24, 24)
        cfg.iterations = 350
        cfg.lr = 0.01
        cfg.device = 'cpu'

        model = build_pinn(cfg)
        lower, upper = (torch.tensor(v) for v in cfg.bounds)
        test_pts = (lower + (upper - lower) * torch.rand(40, 3)).requires_grad_()
        exact_expr = Expression('sin(pi*x)*sin(pi*y)/mu', 3, target=True, coordinates=cfg.coordinate_names)
        residual = model.problem.residual(test_pts, exact_expr(test_pts))
        self.assertLess(residual.abs().max().item(), 1e-4)

        model.train(cfg.iterations, display_every=0)

        # Evaluate center point (0.5, 0.5) across mu = 0.5, 1.0, 2.0 -> expected 2.0, 1.0, 0.5
        probe = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 1.0], [0.5, 0.5, 2.0]])
        preds = model.predict(probe).detach().numpy().reshape(-1)
        self.assertGreater(preds[0], preds[1] + 0.3)
        self.assertGreater(preds[1], preds[2] + 0.2)

        # Hard Dirichlet enforces exact 0 on walls at every mu
        wall = torch.tensor([[0.0, 0.5, 0.5], [1.0, 0.3, 1.7], [0.4, 0.0, 2.0], [0.6, 1.0, 0.8]])
        wall_vals = model.predict(wall).detach().numpy().reshape(-1)
        self.assertLess(float(np.max(np.abs(wall_vals))), 1e-6)

        # Evaluate default grid and verify VTI ParaView export projects cleanly to 2D spatial grid
        outputs = evaluate_outputs(model, cfg, [OutputRequest('sol', 'solution', resolution=9)])
        out = outputs['sol']
        self.assertEqual(len(out['points']), 81)
        self.assertEqual(len(out['points'][0]), 3)
        with TemporaryDirectory() as directory:
            vti_path = Path(directory) / 'parametric.vti'
            export_data(vti_path, dict(out, config=cfg), 'paraview')
            self.assertTrue(vti_path.exists())

    def test_parametric_desktop_workflow_migration_archive_and_live_sliders(self):
        code = r'''
import time
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QSlider
from margam.ui.app import Window
from margam.ui.model_document import CURRENT_VERSION, capture_model, decode_document
from margam.visualization.loaders import from_run_archive

app = QApplication([])
w = Window(state_directory=False)
w.set_application_device('cpu')
w.show()
app.processEvents()

def wait_worker():
    deadline = time.monotonic() + 30
    while w.worker is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert w.worker is None and w.result is not None, w.log.toPlainText()

try:
    assert w.load_example('parametric_poisson_2d') and w.validate_model(), w.validation.text()
    assert w.example_actions['parametric_poisson_2d'] in w.examples_parametric_menu.actions()
    assert w.config.parametric_names == ('mu',)

    # Add & remove a second parameter with Undo/Redo
    assert w.add_parametric_axis('alpha', 1.0, 3.0, 2.0)
    assert w.config.parametric_names == ('mu', 'alpha')
    w.undo_edit()
    assert w.config.parametric_names == ('mu',)
    w.redo_edit()
    assert w.config.parametric_names == ('mu', 'alpha')
    assert w.remove_parametric_axis('alpha')
    assert w.config.parametric_names == ('mu',)

    # Format 35 roundtrip & format 34 migration
    doc = capture_model(w)
    assert doc['version'] == CURRENT_VERSION
    decode_document(doc)

    legacy = deepcopy(doc)
    legacy['version'] = 34
    legacy['config'].pop('parametric_axes', None)
    legacy['config']['pde_expression'] = ''
    for node in legacy['nodes']:
        if node['kind'] == 'domain':
            node['settings']['parameters'].pop('parametric_axes', None)
    migrated, _, _ = decode_document(legacy)
    assert migrated.parametric_axes == ()

    # Short CPU training + archive + Solution Explorer live scrubbing
    w.change('points', 48)
    w.change('boundary_points', 8)
    w.change('layers', (12, 12))
    w.change('iterations', 25)
    for node in w.scene.nodes():
        if node.kind in ('solution', 'comparison', 'export', 'paraview'):
            node.settings['resolution'] = 7

    assert w.start_training()
    wait_worker()
    assert w.result['steps'] == 25

    sol_node = next(n for n in w.scene.nodes() if n.kind == 'solution')
    explorer = w.show_solution_explorer(sol_node)
    assert explorer is not None
    assert explorer.dataset is not None
    assert explorer.dataset.spatial_dim == 2
    assert explorer.dataset.live_evaluator is not None
    spin = explorer.findChild(QDoubleSpinBox, 'explorer_param_spin_mu')
    slider = explorer.findChild(QSlider, 'explorer_param_slider_mu')
    assert spin is not None and slider is not None and spin.isEnabled()

    ds_low = explorer.evaluate_live_params({'mu': 0.5}, sync=True)
    ds_high = explorer.evaluate_live_params({'mu': 2.0}, sync=True)
    assert ds_low is not None and ds_high is not None
    diff = float(np.max(np.abs(ds_low.fields['u'] - ds_high.fields['u'])))
    assert diff > 1e-4, f'Expected distinct predictions across mu, got max diff {diff}'

    xs, ys = explorer.plot_continuous_parametric_response(samples=9)
    assert len(xs) == 9 and len(ys) == 9
    assert len(explorer.probe_plot.series) == 1

    explorer.reset_continuous_params()
    assert abs(spin.value() - 1.0) < 1e-5
    explorer.close()

    with TemporaryDirectory() as directory:
        root = Path(directory)
        model_path = root / 'parametric.margam.json'
        assert w.save_model(model_path)
        run_path = root / 'parametric.margam-run.pt'
        assert w.save_current_run(run_path)

        # Saved archive should also expose a lazy checkpoint live_evaluator for parametric scrubbing
        archived = from_run_archive(run_path)
        assert archived
        arch_ds = next(iter(archived.values()))
        assert arch_ds.live_evaluator is not None
        spatial = arch_ds.points[:, list(arch_ds.spatial_indices)]
        p_low = arch_ds.live_evaluator(spatial, params={'mu': 0.5})['u']
        p_high = arch_ds.live_evaluator(spatial, params={'mu': 2.0})['u']
        assert float(np.max(np.abs(p_low - p_high))) > 1e-4

        w.undo_stack.setClean()
        assert w.load_model(model_path)
        assert w.config.parametric_names == ('mu',)
finally:
    app.clipboard().clear()
    w.undo_stack.setClean()
    w.close()
    app.processEvents()
'''
        result = subprocess.run(
            [sys.executable, '-c', code],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=120,
            env=dict(
                os.environ,
                QT_QPA_PLATFORM='offscreen',
                CUDA_VISIBLE_DEVICES='',
                PYTHONDONTWRITEBYTECODE='1',
            ),
        )
        self.assertEqual(result.returncode, 0, result.stdout + '\n' + result.stderr)
        self.assertNotIn('Traceback', result.stderr)


if __name__ == '__main__':
    unittest.main()
