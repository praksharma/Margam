from copy import deepcopy
from dataclasses import asdict
import math
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import torch

from margam.benchmarking import BenchmarkCase, Recipe, run_benchmark
from margam.components.adaptive_sampling import sampling_defaults
from margam.components.differential import derivative
from margam.components.problem import PINNProblem, PointConstraint
from margam.execution.study_runner import run_study
from margam.methods.dsgnar import DSGNARPINN, lm_step, reweight
from margam.methods.gpinn import GradientEnhancedPINN
from margam.methods.pinn import PINN
from margam.study_document import capture as capture_study, write as write_study
from margam.templates.examples import example_config
from margam.templates.poisson import BoundarySampling, PoissonConfig, build_pinn
from margam.ui.run_document import load_run, save_run


class HighSeverityNumericalRegressionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_dsgnar_max_radius_checkpoint_roundtrip_and_lm_reweight_stability(self):
        # 1. At max_radius=70.0, logspace(log10(low), log10(70.0), 8)[-1] overshoots 70.0 by 1 ULP,
        # causing DSGNAR's own training_state() to be rejected by load_training_state()!
        net = torch.nn.Linear(1, 1, bias=False).double()
        net.weight.data.zero_()
        problem = PINNProblem(lambda x, u: u, torch.ones((4, 1), dtype=torch.float64), [], residual_target=1.0)
        cfg = example_config('dsgnar_poisson_1d')
        settings = deepcopy(cfg.dsgnar_settings)
        settings.update(
            sketch_size=1,
            hashes=1,
            probes=8,
            initial_radius=30.0,
            max_radius=70.0,
            reweight_exponent=0.0,
            precision='float64',
        )
        m = DSGNARPINN(problem, deepcopy(net), settings=settings).compile(device='cpu')
        m.train(1, display_every=0)
        self.assertLessEqual(m.radius, 70.0)
        restored = DSGNARPINN(problem, deepcopy(net), settings=settings).compile(device='cpu')
        restored.load_training_state(m.training_state())
        self.assertEqual(restored.radius, m.radius)

        # 2. lm_step with float32 subnormal singular values and extreme float64 ill-conditioning
        s32 = torch.tensor([1.0, 1e-30], dtype=torch.float32)
        p32 = torch.tensor([0.5, 1e-10], dtype=torch.float32)
        v32 = torch.eye(2, dtype=torch.float32)
        step32, lam32 = lm_step(s32, p32, v32, 1.0)
        # Unconstrained step in component 1 would be -1e-10 / 1e-30 = -1e20 >> 1.0, so lam32 MUST be > 0
        # and ||step32|| must match radius 1.0
        self.assertGreater(lam32, 0.0)
        self.assertAlmostEqual(torch.linalg.vector_norm(step32).item(), 1.0, places=5)

        s64 = torch.tensor([1e15, 1e-10], dtype=torch.float64)
        p64 = torch.tensor([1.0, 10.0], dtype=torch.float64)
        v64 = torch.eye(2, dtype=torch.float64)
        step64, lam64 = lm_step(s64, p64, v64, 1.0)
        self.assertGreater(lam64, 0.0)
        self.assertAlmostEqual(torch.linalg.vector_norm(step64).item(), 1.0, places=10)

        # 3. reweight with subnormal positive sum and huge finite loss
        sub_val = math.ldexp(1.0, -1074)  # smallest positive float64 subnormal (~5e-324)
        w_sub = reweight({'a': 1.0, 'b': 1.0, 'c': 1.0}, {'a': sub_val, 'b': 0.0, 'c': 0.0}, 0.5)
        self.assertTrue(all(math.isfinite(v) and v > 0 for v in w_sub.values()))

        w_huge = reweight({'a': 1.0, 'b': 1.0}, {'a': 1e305, 'b': 1.0}, 0.5)
        self.assertTrue(all(math.isfinite(v) and v > 0 for v in w_huge.values()))
        self.assertLess(w_huge['a'], w_huge['b'])

    def test_time_marching_pre_model32_checkpoint_and_zero_local_step_guard(self):
        c = example_config('expert_burgers')
        c.points = 8
        c.boundary_points = 4
        c.layers = (4,)
        c.expert_settings.update(chunks=2, balance_every=2, windows=2, window_steps=2)
        c.iterations = 4

        m = build_pinn(c)
        m.train(3, display_every=0)
        legacy_state = m.training_state()
        legacy_state['marching']['configuration'].pop('dsgnar_settings', None)

        resumed = build_pinn(c)
        resumed.load_training_state(legacy_state)
        self.assertEqual(resumed.iter, 3)

        # Forged state with completed window and 0 active steps in next window must be rejected
        forged = m.training_state()
        forged['iter'] = 2
        forged['loss_history'] = forged['loss_history'][:2]
        forged['training_history'] = forged['training_history'][:2]
        with self.assertRaises(ValueError):
            build_pinn(c).load_training_state(forged)

    def test_adaptive_sampling_run_archive_with_owned_and_missing_boundary_data(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            b_csv = root / 'left.csv'
            b_csv.write_text('x,y\n0.0,0.2\n0.0,0.8\n')
            cfg = PoissonConfig(
                shape='Rectangle',
                points=8,
                boundary_points=2,
                layers=(4,),
                pde_points='both',
                adaptive_sampling=dict(sampling_defaults('rar'), interval=1, batch=2, candidates=8, maximum=16),
                boundary_data={'left': {'path': str(b_csv), 'coordinates': 'x,y', 'normals': '', 'values': ''}},
            )
            model = build_pinn(cfg)
            model.train(2, display_every=0)
            state = model.training_state()

            # Case 1: v4 run archive with project-owned 'margam-owned:datasets/left.csv'
            owned_rel = 'datasets/left.csv'
            portable_cfg = asdict(cfg)
            portable_cfg['layers'] = list(cfg.layers)
            portable_cfg['flow_residuals'] = list(cfg.flow_residuals)
            portable_cfg['pressure_point'] = list(cfg.pressure_point)
            portable_cfg['boundary_data']['left']['path'] = 'margam-owned:' + owned_rel
            archive_owned = {
                'format': 'margam-run',
                'version': 4,
                'model': {'format': 'margam-model', 'version': 33, 'config': portable_cfg, 'nodes': [], 'wires': []},
                'result': {'outputs': {}, 'steps': 2, 'cancelled': False, 'device': 'cpu'},
                'monitors': {},
                'checkpoint': state,
                'owned_files': {owned_rel: b_csv.read_bytes()},
            }
            owned_pt = root / 'owned.margam-run.pt'
            save_run(owned_pt, archive_owned)
            b_csv.unlink()  # External original removed; archive embeds owned copy
            loaded_owned = load_run(owned_pt)
            self.assertEqual(loaded_owned['result']['steps'], 2)

            # Case 2: Linked external boundary_data that moved; offline load_run must still succeed
            linked_cfg = deepcopy(portable_cfg)
            linked_cfg['boundary_data']['left']['path'] = str(b_csv)  # now deleted
            archive_linked = {
                'format': 'margam-run',
                'version': 4,
                'model': {'format': 'margam-model', 'version': 33, 'config': linked_cfg, 'nodes': [], 'wires': []},
                'result': {'outputs': {}, 'steps': 2, 'cancelled': False, 'device': 'cpu'},
                'monitors': {},
                'checkpoint': state,
                'owned_files': {},
            }
            linked_pt = root / 'linked.margam-run.pt'
            save_run(linked_pt, archive_linked)
            loaded_linked = load_run(linked_pt)
            self.assertEqual(loaded_linked['result']['steps'], 2)

    def test_expert_boundary_collocation_validation_causal_min_persistence_and_study_windows(self):
        # 1. Transient Expert with pde_points='boundary' must be rejected during validate()
        bad_cfg = example_config('expert_advection')
        bad_cfg.points = 8
        bad_cfg.boundary_points = 4
        bad_cfg.pde_points = 'boundary'
        with self.assertRaises(ValueError):
            bad_cfg.validate()

        # 2. _causal_min must persist across training_state / load_training_state
        cfg = example_config('expert_advection')
        cfg.points = 8
        cfg.boundary_points = 4
        cfg.layers = (4,)
        cfg.expert_settings.update(chunks=2, balance_every=2, causal_epsilon=5.0)
        m = build_pinn(cfg)
        m.train(2, display_every=0)
        causal_before = m.expert_summary()['minimum_causal_weight']
        self.assertLess(causal_before, 1.0)
        m2 = build_pinn(cfg)
        m2.load_training_state(m.training_state())
        self.assertAlmostEqual(m2.expert_summary()['minimum_causal_weight'], causal_before, places=12)

        # 3. Multi-window Expert study with unsynchronized iterations must run windows * window_steps
        with TemporaryDirectory() as tmp:
            from margam.ui.app import QApplication, Window
            from margam.ui.model_document import capture_model
            app = QApplication.instance() or QApplication([])
            w = Window(state_directory=False)
            try:
                w.load_example('expert_burgers')
                w.change('points', 8)
                w.change('boundary_points', 4)
                w.change('layers', (4,))
                sol = w.add_node('Solution Plot')
                sol.settings['resolution'] = 3
                method_node = next(n for n in w.scene.nodes() if n.kind == 'expert_method')
                w.scene.connect_nodes(method_node, sol)
                doc = capture_model(w)
                # Simulate JSON/programmatic multi-window edit without GUI spinbox iterations sync
                m_rec = next(n for n in doc['nodes'] if n['id'] == method_node.identifier)
                m_rec['settings']['parameters']['expert_settings'].update(
                    chunks=2, balance_every=2, windows=2, window_steps=2
                )
                m_rec['settings']['parameters']['iterations'] = 2  # stale!
                net_node = next(n for n in doc['nodes'] if n['kind'] == 'network')
                axes = [{'node_id': net_node['id'], 'parameter': 'network_seed', 'mode': 'fixed', 'text': '7'}]
                study, files = capture_study(
                    doc, axes, w.project_workspace, execution={'device': 'cpu', 'wall_time_seconds': 0}
                )
                study_path = Path(tmp) / 'mw.margam-study'
                write_study(study_path, study, files)
                report = run_study(study_path, Path(tmp) / 'out', device='cpu')
                self.assertEqual(report['counts'], {'completed': 1})
                self.assertEqual(report['trials'][0]['steps'], 4)
                saved = load_run(Path(tmp) / 'out/DP0001/artifacts/run.margam-run.pt')
                self.assertIn(sol.identifier, saved['result']['outputs'])
            finally:
                app.clipboard().clear()
                w.undo_stack.setClean()
                w.close()
                app.processEvents()

    def test_benchmark_runner_preserves_measures_callable_targets_and_converged_dsgnar(self):
        # 1. Integral loss measure != 1.0 and callable residual_target for gPINN
        def make_problem():
            x = torch.linspace(0.1, 0.9, 6)[:, None]
            bc = PointConstraint('bc', torch.tensor([[0.0], [1.0]]), 0.0, measure=2.0)
            return PINNProblem(
                lambda pts, u: derivative(derivative(u, pts, 0), pts, 0),
                x,
                [bc],
                residual_target=lambda pts: -2.0 * torch.ones_like(pts),
                measure=3.5,
            )

        eval_pts = torch.linspace(0.0, 1.0, 5)[:, None]
        bench = BenchmarkCase(' custom_measure '.strip(), make_problem, eval_pts, lambda x: x * (1.0 - x))

        observed_measures = []

        def build_integral(prob):
            m = PINN(prob, torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1)), loss_formulation='integral').compile(device='cpu')
            observed_measures.append(dict(m.loss_measures))
            return m

        def build_gpinn(prob):
            return GradientEnhancedPINN(
                prob,
                torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1)),
                coordinates=('x',),
                gradient_weights={'x': 0.1, 'y': 0.0, 't': 0.0},
            ).compile(device='cpu')

        def build_converged_dsgnar(prob):
            settings = dsgnar_defaults()
            settings.update(sketch_size=4, hashes=1, probes=3, min_radius=1e-4, initial_radius=2e-4)
            net = torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
            with torch.no_grad():
                for p in net.parameters():
                    p.zero_()
            return DSGNARPINN(
                prob,
                net,
                settings=settings,
            ).compile(device='cpu')

        from margam.components.dsgnar_config import dsgnar_defaults
        with TemporaryDirectory() as tmp:
            results = run_benchmark(
                bench,
                [
                    Recipe('integral', build_integral),
                    Recipe('gpinn', build_gpinn),
                    Recipe('dsgnar', build_converged_dsgnar),
                ],
                seeds=[5],
                iterations=3,
                output_dir=Path(tmp) / 'bench',
            )
            self.assertEqual([r['status'] for r in results], ['completed', 'completed', 'completed'])
            self.assertEqual(observed_measures[0], {'pde': 3.5, 'bc': 2.0})


if __name__ == '__main__':
    unittest.main()
