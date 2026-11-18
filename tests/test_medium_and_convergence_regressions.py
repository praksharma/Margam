from copy import deepcopy
import math
from unittest.mock import patch
import warnings
import unittest
import torch
from torch import nn

from margam.backend.utils import loss_metric
from margam.components.differential import derivative
from margam.components.expressions import Expression
from margam.components.outputs import OutputRequest, ReferenceSource, evaluate_outputs
from margam.components.prediction import CheckpointPredictor, prediction_model
from margam.components.problem import PINNProblem, PointConstraint
from margam.components.resources import memory_snapshot
from margam.methods.inverse_burgers import InverseBurgersNetwork, InverseBurgersPINN, burgers_residual
from margam.nn.FCNN import FullyConnected
from margam.nn.factory import build_network
from margam.nn.utils import activation, initialiser
from margam.templates.examples import example_config
from margam.templates.poisson import PoissonConfig, build_pinn


class MediumAndConvergenceRegressionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_validate_does_not_mutate_invalid_config_and_advection_diffusion_high_viscosity(self):
        cfg = example_config('advection_diffusion')
        cfg.pde_expression = ''
        cfg.shape = 'Rectangle'  # invalid for advection_diffusion_1d
        with self.assertRaises(ValueError):
            cfg.validate()
        self.assertEqual(cfg.pde_expression, '')

        # viscosity > 1e6 must not fail Expression's 1e6 literal constant check
        high_nu = PoissonConfig(
            shape='Interval',
            pde_preset='advection_diffusion_1d',
            viscosity=2e6,
            source=1.0,
            points=8,
            boundary_points=2,
            layers=(4,),
        )
        high_nu.validate()
        m = build_pinn(high_nu)
        self.assertIn('nu', high_nu.pde_expression)
        self.assertTrue(all(torch.isfinite(v).item() for v in m.compute_losses().values()))

    def test_scale_and_expert_ntk_and_evaluate_outputs_avoid_duplicate_passes(self):
        # 1. ScalePINN.compute_losses() should evaluate problem.residual once per call
        c_scale = example_config('scale_allen_cahn')
        c_scale.points = 8
        c_scale.boundary_points = 2
        c_scale.layers = (4, 4)
        c_scale.network_options['features'] = 2
        m_scale = build_pinn(c_scale)
        orig_residual = m_scale.problem.residual
        calls = []

        def counted_residual(x, u):
            calls.append(1)
            return orig_residual(x, u)

        m_scale.problem.residual = counted_residual
        m_scale.compute_losses()
        self.assertEqual(len(calls), 1)

        # 2. ExpertPINN with balance_mode='ntk' should call loss_samples() once per train step
        c_exp = example_config('expert_advection')
        c_exp.points = 8
        c_exp.boundary_points = 4
        c_exp.layers = (4,)
        c_exp.expert_settings.update(chunks=2, balance_every=1, balance_mode='ntk', ntk_points=2)
        m_exp = build_pinn(c_exp)
        with patch.object(m_exp, 'loss_samples', wraps=m_exp.loss_samples) as spy_samples:
            m_exp.train(1, display_every=0)
            self.assertEqual(spy_samples.call_count, 1)

        # 3. evaluate_outputs deduplicates identical grid predictions across Comparison -> Exports
        c_base = PoissonConfig(shape='Interval', points=6, boundary_points=2, layers=(4,))
        m_base = build_pinn(c_base)
        ref = ReferenceSource('zero', lambda pts: [0.0 for _ in pts])
        reqs = [
            OutputRequest('comp', 'comparison', resolution=11, reference=ref),
            OutputRequest('csv', 'export', resolution=11, reference=ref),
            OutputRequest('pv', 'paraview', resolution=11, reference=ref),
        ]
        with patch.object(m_base, 'predict', wraps=m_base.predict) as spy_pred:
            out = evaluate_outputs(m_base, c_base, reqs)
            self.assertEqual(spy_pred.call_count, 1)
            self.assertEqual(set(out), {'comp', 'csv', 'pv'})

    def test_modified_fourier_depth1_and_legacy_nn_backend_helpers(self):
        # Depth-1 ModifiedFourierNetwork must give active gradients to encoder_u and encoder_v
        cfg = PoissonConfig(shape='Interval', network_type='modified_fourier', layers=(6,))
        net = build_network(cfg)
        x = torch.tensor([[0.2], [0.7]])
        loss = net(x).square().sum()
        loss.backward()
        self.assertIsNotNone(net.encoder_u.weight.grad)
        self.assertIsNotNone(net.encoder_v.weight.grad)
        self.assertGreater(net.encoder_u.weight.grad.abs().sum().item() + net.encoder_v.weight.grad.abs().sum().item(), 0.0)

        # Legacy FCNN with "linear" activation and "Glorot uniform" initializer without warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fcnn = FullyConnected([1, 4, 1], 'linear', 'Glorot uniform')
            out = fcnn(torch.tensor([[0.5]]))
            self.assertEqual(out.shape, (1, 1))
            self.assertFalse(any(issubclass(w.category, UserWarning) for w in caught))

        # Singleton isolation for activation and loss_metric
        self.assertIsNot(activation('tanh'), activation('tanh'))
        self.assertIsNot(loss_metric('MSE'), loss_metric('MSE'))
        self.assertIsInstance(initialiser('Glorot uniform'), object)

        with self.assertRaises(ValueError):
            FullyConnected([1], 'tanh', 'Xavier normal')

    def test_inverse_burgers_and_expression_edge_cases(self):
        # 1. InverseBurgersNetwork matches float64 base network and works without _margam_spec
        base64 = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)).double()
        inv_net = InverseBurgersNetwork(base64, {'lambda_1': 0.5, 'log_lambda_2': -3.0})
        self.assertEqual(inv_net.lambda_1.dtype, torch.float64)
        self.assertEqual(inv_net.log_lambda_2.dtype, torch.float64)

        pts = torch.tensor([[0.2, 0.1], [0.5, 0.2]], dtype=torch.float64)
        vals = torch.tensor([[0.1], [0.2]], dtype=torch.float64)
        prob = PINNProblem(
            lambda x, u: burgers_residual(x, u, inv_net),
            pts,
            [PointConstraint('observations', pts, vals)],
        )
        inv = InverseBurgersPINN(prob, inv_net, weights={'pde': 1.0, 'observations': 1.0}, initial={'lambda_1': 0.5, 'log_lambda_2': -3.0})
        with self.assertRaises(RuntimeError):
            inv.train(0, display_every=0)
        inv.compile(lr=1e-3, device='cpu')
        inv.train(1, display_every=0)
        self.assertEqual(len(inv.inverse_history), 1)
        inv.compile(lr=1e-3, device='cpu')
        self.assertEqual(inv.inverse_history, [])

        # Underflowed lambda_2 after step must roll back iter and loss_history
        under_net = InverseBurgersNetwork(deepcopy(base64), {'lambda_1': 0.5, 'log_lambda_2': -745.0})
        under_prob = PINNProblem(
            lambda x, u: burgers_residual(x, u, under_net),
            pts,
            [PointConstraint('observations', pts, vals + 10.0)],
        )
        under_inv = InverseBurgersPINN(
            under_prob, under_net, weights={'pde': 1.0, 'observations': 1.0}, initial={'lambda_1': 0.5, 'log_lambda_2': -745.0}
        ).compile(lr=100.0, device='cpu')
        with torch.no_grad():
            under_net.log_lambda_2.fill_(-800.0)  # exp(-800) == 0.0 in float64
        with self.assertRaises(FloatingPointError):
            under_inv.train(1, display_every=0)
        self.assertEqual(under_inv.iter, 0)
        self.assertEqual(under_inv.loss_history, [])

        # 2. Expression complex / fractional negative power and 1D ny checks
        expr_const_complex = Expression('(-1)**0.5', 1, target=True)
        with self.assertRaises(ValueError):
            expr_const_complex(torch.tensor([[0.5]]))

        expr_neg_frac = Expression('(-x)**0.5', 1, target=True)
        with self.assertRaises(ValueError):
            expr_neg_frac(torch.tensor([[0.5]]))

        with self.assertRaises(ValueError):
            Expression('ny * x', 1, target=True, coordinates=('x',))

    def test_prediction_model_preserves_rng_checks_network_spec_and_auto_memory_snapshot(self):
        cfg = PoissonConfig(shape='Interval', network_type='siren', points=6, boundary_points=2, layers=(8, 8))
        m = build_pinn(cfg)
        m.train(1, display_every=0)
        state = m.training_state()

        torch.manual_seed(12345)
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
        pred = prediction_model(cfg, state)
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        if cuda_rng is not None:
            self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
        self.assertEqual(pred.predict(torch.tensor([[0.5]])).shape, (1, 1))

        # Missing network_spec on non-fcnn checkpoint must be rejected
        stripped = deepcopy(state)
        stripped.pop('network_spec', None)
        with self.assertRaises(ValueError):
            CheckpointPredictor(cfg, stripped)

        if torch.cuda.is_available():
            snap = memory_snapshot('auto')
            self.assertIsNotNone(snap['gpu_available'])
            self.assertGreater(snap['gpu_total'], 0)

    def test_analytical_convergence_across_advanced_methods(self):
        # 1. gPINN on 1D Poisson: u''(x) + pi^2 sin(pi x) = 0, u(0)=u(1)=0 => exact u(x) = sin(pi x)
        torch.manual_seed(7)
        c_gpinn = PoissonConfig(
            shape='Interval',
            method='gpinn',
            pde_preset='poisson',
            source=0.0,
            pde_forcing='pi*pi*sin(pi*x)',
            points=32,
            boundary_points=2,
            layers=(16, 16),
            lr=5e-3,
            boundary_weight=10.0,
            gpinn_weights={'x': 0.05, 'y': 0.0, 't': 0.0},
            device='cpu',
        )
        m_gpinn = build_pinn(c_gpinn)
        m_gpinn.train(400, display_every=0)
        x_eval = torch.linspace(0.0, 1.0, 41)[:, None]
        exact_sin = torch.sin(math.pi * x_eval)
        rmse_gpinn = (m_gpinn.predict(x_eval) - exact_sin).square().mean().sqrt().item()
        self.assertLess(rmse_gpinn, 0.03)

        # 2. RBA on 1D Poisson: u''(x) + 2 = 0, u(0)=u(1)=0 => exact u(x) = x*(1 - x)
        torch.manual_seed(7)
        c_rba = PoissonConfig(
            shape='Interval',
            method='rba',
            source=2.0,
            points=32,
            boundary_points=2,
            layers=(16, 16),
            lr=5e-3,
            boundary_weight=10.0,
            device='cpu',
        )
        m_rba = build_pinn(c_rba)
        m_rba.train(400, display_every=0)
        exact_quad = x_eval * (1.0 - x_eval)
        rmse_rba = (m_rba.predict(x_eval) - exact_quad).square().mean().sqrt().item()
        self.assertLess(rmse_rba, 0.02)

        # 3. ScalePINN on 1D steady reaction-diffusion: -u''(x) + u(x) = (1 + pi^2) sin(pi x) => exact sin(pi x)
        torch.manual_seed(7)
        c_scale = PoissonConfig(
            shape='Interval',
            method='scale',
            pde_preset='reaction_diffusion',
            source=1.0,
            pde_forcing='(1 + pi*pi)*sin(pi*x)',
            sampling='random',
            points=48,
            boundary_points=2,
            layers=(16, 16),
            lr=5e-3,
            boundary_weight=10.0,
            device='cpu',
        )
        c_scale.scale_settings.update(tau_sc=0.5, tau_alpha=2.0, gamma=0.01, strength=1.0, fresh=False)
        m_scale = build_pinn(c_scale)
        m_scale.train(450, display_every=0)
        rmse_scale = (m_scale.predict(x_eval) - exact_sin).square().mean().sqrt().item()
        self.assertLess(rmse_scale, 0.03)

        # 4. ExpertPINN & 2-window TimeMarchingPINN on 1D advection u_t + c u_x = 0:
        # exact periodic solution on x in [0, 2*pi], c=1: u(x, t) = sin(x - t)
        for windows in (1, 2):
            torch.manual_seed(7)
            c_exp = example_config('expert_advection')
            c_exp.device = 'cpu'
            c_exp.t_max = 0.4
            c_exp.y_max = 0.4
            c_exp.source = 1.0  # wave speed beta = 1.0
            c_exp.points = 64
            c_exp.boundary_points = 16
            c_exp.layers = (16, 16)
            c_exp.lr = 5e-3
            c_exp.boundary_weight = 10.0
            c_exp.expert_settings.update(
                chunks=4,
                causal_epsilon=0.5,
                balance_every=25,
                windows=windows,
                window_steps=300,
            )
            m_exp = build_pinn(c_exp)
            m_exp.train(600, display_every=0)
            xs = torch.linspace(c_exp.x_min, c_exp.x_max, 21)
            ts = torch.linspace(0.0, 0.4, 5)
            grid = torch.cartesian_prod(xs, ts)
            exact_adv = torch.sin(grid[:, 0:1] - c_exp.source * grid[:, 1:2])
            rmse_exp = (m_exp.predict(grid) - exact_adv).square().mean().sqrt().item()
            self.assertLess(rmse_exp, 0.08)

        # 5. InverseBurgersPINN parameter recovery on exact traveling wave u(x,t) = 1 - tanh((x - t)/(2*nu))
        # Exact PDE: u_t + 1.0 * u * u_x - 0.1 * u_xx = 0 (lambda_1 = 1.0, lambda_2 = 0.1)
        torch.manual_seed(7)
        nu_true = 0.1
        xs_b = torch.linspace(-0.5, 0.5, 15, dtype=torch.float64)
        ts_b = torch.linspace(0.0, 0.3, 9, dtype=torch.float64)
        xt = torch.cartesian_prod(xs_b, ts_b)
        u_exact = 1.0 - torch.tanh((xt[:, 0:1] - xt[:, 1:2]) / (2.0 * nu_true))
        base_net = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1)).double()
        inv_net = InverseBurgersNetwork(base_net, {'lambda_1': 0.2, 'log_lambda_2': math.log(0.02)})
        inv_prob = PINNProblem(
            lambda x, u: burgers_residual(x, u, inv_net),
            xt,
            [PointConstraint('observations', xt, u_exact)],
        )
        inv_model = InverseBurgersPINN(
            inv_prob,
            inv_net,
            weights={'pde': 1.0, 'observations': 20.0},
            initial={'lambda_1': 0.2, 'log_lambda_2': math.log(0.02)},
        ).compile(lr=1e-2, device='cpu')
        inv_model.train(600, display_every=0)
        est = inv_model.estimated_parameters()
        # Initial errors: |0.2 - 1.0| = 0.80, |0.02 - 0.10| = 0.08; verify substantial convergence to true (1.0, 0.1)
        self.assertLess(abs(est['lambda_1'] - 1.0), 0.15)
        self.assertLess(abs(est['lambda_2'] - 0.1), 0.03)


if __name__ == '__main__':
    unittest.main()
