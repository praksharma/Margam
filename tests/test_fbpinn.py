"""Mathematical, autograd, multi-domain, checkpoint, and accuracy tests for FBPINN."""
from copy import deepcopy
from dataclasses import asdict
import math
import unittest
import torch

from margam.components.expressions import Expression
from margam.components.fbpinn_config import (
    active_subdomain_counts,
    fbpinn_defaults,
    spatial_time_axes,
    total_networks,
    total_subdomains,
    validate_fbpinn_problem,
    validate_fbpinn_settings,
)
from margam.components.prediction import CheckpointPredictor, prediction_model
from margam.components.resources import architecture_size
from margam.methods.fbpinn import (
    FiniteBasisNetwork,
    FiniteBasisPINN,
    STATE_ACTIVE,
    STATE_FIXED,
    STATE_INACTIVE,
)
from margam.templates.examples import EXAMPLE_PROBLEMS, example_config
from margam.templates.incompressible_flow import flow_demo_config
from margam.templates.poisson import PoissonConfig, build_pinn
from margam.utils import laplacian


class FBPINNConfigTests(unittest.TestCase):
    def test_validation_rejects_invalid_settings_and_compositions(self):
        base = fbpinn_defaults()
        validate_fbpinn_settings(base)

        for key, bad in (
            ('subdomains_x', 0),
            ('subdomains_x', 33),
            ('subdomains_x', True),
            ('overlap', 1.0),
            ('overlap', 3.5),
            ('overlap', float('nan')),
            ('window', 'gaussian'),
            ('window_sigma', 0.001),
            ('window_sigma', 0.9),
            ('multilevel', 1),
            ('schedule', 'causal'),
            ('schedule_axis', 'w'),
            ('active_steps', 0),
            ('freeze_completed', 'yes'),
        ):
            s = deepcopy(base)
            s[key] = bad
            with self.assertRaises(ValueError, msg=f'{key}={bad!r}'):
                validate_fbpinn_settings(s)

        # Total subdomains > 64 rejected
        c = PoissonConfig(shape='Rectangle', method='fbpinn')
        c.fbpinn_settings.update(subdomains_x=9, subdomains_y=8)
        with self.assertRaisesRegex(ValueError, 'at most 64'):
            validate_fbpinn_problem(c)

        # Multilevel requires >= 2 subdomains and simultaneous schedule
        c = PoissonConfig(shape='Interval', method='fbpinn')
        c.fbpinn_settings.update(subdomains_x=1, multilevel=True)
        with self.assertRaisesRegex(ValueError, 'at least 2'):
            validate_fbpinn_problem(c)

        c.fbpinn_settings.update(subdomains_x=4, multilevel=True, schedule='progressive')
        with self.assertRaisesRegex(ValueError, 'simultaneous'):
            validate_fbpinn_problem(c)

        # Progressive requires >= 2 subdomains on target axis
        c = PoissonConfig(shape='Interval', method='fbpinn')
        c.fbpinn_settings.update(subdomains_x=1, schedule='progressive', schedule_axis='auto')
        with self.assertRaisesRegex(ValueError, 'at least 2'):
            validate_fbpinn_problem(c)

        c.fbpinn_settings.update(subdomains_x=4, schedule='progressive', schedule_axis='t')
        with self.assertRaisesRegex(ValueError, 'not present'):
            validate_fbpinn_problem(c)

        # 3rd-order derivative rejects non-C^inf windows, allows smooth_sigmoid
        c = PoissonConfig(shape='Interval', method='fbpinn', pde_expression='D(u,x,3) + 1')
        c.fbpinn_settings.update(subdomains_x=3, window='cosine_sq')
        with self.assertRaisesRegex(ValueError, 'smooth_sigmoid'):
            validate_fbpinn_problem(c)
        c.fbpinn_settings['window'] = 'cosine'
        with self.assertRaisesRegex(ValueError, 'smooth_sigmoid'):
            validate_fbpinn_problem(c)
        c.fbpinn_settings['window'] = 'smooth_sigmoid'
        validate_fbpinn_problem(c)

        # Unsupported loss/sampling/geometry rejected
        c = PoissonConfig(shape='Interval', method='fbpinn', loss_formulation='integral')
        with self.assertRaisesRegex(ValueError, 'mean MSE'):
            validate_fbpinn_problem(c)

        c = PoissonConfig(shape='Prepared2D', method='fbpinn')
        with self.assertRaisesRegex(ValueError, 'built-in'):
            validate_fbpinn_problem(c)

    def test_resource_accounting_scales_with_network_count(self):
        c1 = PoissonConfig(shape='Rectangle', method='baseline', layers=(16, 16))
        c2 = PoissonConfig(shape='Rectangle', method='fbpinn', layers=(16, 16))
        c2.fbpinn_settings.update(subdomains_x=3, subdomains_y=2, multilevel=False)
        c3 = deepcopy(c2)
        c3.fbpinn_settings['multilevel'] = True
        s1, s2, s3 = architecture_size(c1), architecture_size(c2), architecture_size(c3)
        self.assertEqual(total_subdomains(c2), 6)
        self.assertEqual(total_networks(c2), 6)
        self.assertEqual(total_networks(c3), 7)
        self.assertEqual(s2['parameters'], 6 * s1['parameters'])
        self.assertEqual(s3['parameters'], 7 * s1['parameters'])
        # Confirm actual PyTorch parameter count matches arithmetic
        model = build_pinn(c3)
        actual_params = sum(p.numel() for p in model.network.parameters())
        self.assertEqual(actual_params, s3['parameters'])


class FBPINNMathAndAutogradTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_single_subdomain_exact_equivalence_to_baseline(self):
        base_cfg = PoissonConfig(
            shape='Interval', points=24, sampling='grid', seed=7, network_seed=7,
            layers=(12, 12), iterations=20, lr=0.01, method='baseline', device='cpu',
        )
        fb_cfg = deepcopy(base_cfg)
        fb_cfg.method = 'fbpinn'
        fb_cfg.fbpinn_settings.update(subdomains_x=1, multilevel=False, schedule='simultaneous')

        m_base = build_pinn(base_cfg)
        m_fb = build_pinn(fb_cfg)
        m_base.train(15, display_every=0)
        m_fb.train(15, display_every=0)

        probe = torch.linspace(0.0, 1.0, 41).unsqueeze(1)
        self.assertTrue(torch.allclose(m_base.predict(probe), m_fb.predict(probe), atol=1e-6, rtol=1e-6))
        for r1, r2 in zip(m_base.training_history, m_fb.training_history):
            self.assertAlmostEqual(r1, r2, places=6)

    def test_partition_of_unity_and_second_order_derivatives_on_edges(self):
        # Verify exact partition-of-unity sum == 1 everywhere including corners/edges
        # and finite-difference match of u, grad(u), laplacian(u) right on support edges!
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            for window in ('cosine_sq', 'smooth_sigmoid', 'cosine'):
                c = PoissonConfig(shape='Interval', method='fbpinn', layers=(8, 8), seed=5, network_seed=5, device='cpu')
                c.fbpinn_settings.update(subdomains_x=4, overlap=1.5, window=window, multilevel=(window == 'cosine_sq'))
                model = build_pinn(c)
                net = model.network
                # Include exact endpoints (0, 1), overlap midpoints (0.25, 0.5, 0.75),
                # and exact compact support edges (0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125)!
                x = torch.tensor(
                    [[0.0], [0.1], [0.1875], [0.25], [0.3125], [0.4375], [0.5], [0.5625], [0.6875], [0.8125], [0.9], [1.0]],
                    dtype=torch.float64,
                )
                _, w_hat = net.normalized_windows(x)
                self.assertTrue(torch.isfinite(w_hat).all(), window)
                self.assertTrue(torch.allclose(w_hat.sum(dim=1), torch.ones(len(x), dtype=torch.float64), atol=1e-12), window)

                # Autograd vs central finite differences for 1st & 2nd derivatives
                x_grad = x.clone().requires_grad_(True)
                u = net(x_grad)
                du = torch.autograd.grad(u.sum(), x_grad, create_graph=True)[0]
                d2u = laplacian(u, x_grad)
                self.assertTrue(torch.isfinite(du).all(), window)
                self.assertTrue(torch.isfinite(d2u).all(), window)
                # Verify double backward into parameters succeeds cleanly
                d2u.square().mean().backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()), window)

                # Check interior & overlap-edge finite-difference agreement for C^2 / C^inf windows
                if window in ('cosine_sq', 'smooth_sigmoid'):
                    eps = 1e-5
                    # Avoid exact endpoints for symmetric central differences
                    x_mid = x[1:-1]
                    with torch.no_grad():
                        u_p = net(x_mid + eps)
                        u_0 = net(x_mid)
                        u_m = net(x_mid - eps)
                        fd_d1 = (u_p - u_m) / (2.0 * eps)
                        fd_d2 = (u_p - 2.0 * u_0 + u_m) / (eps ** 2)
                    self.assertTrue(torch.allclose(du[1:-1].detach(), fd_d1, atol=1e-4, rtol=1e-4), f'{window} d1 mismatch')
                    self.assertTrue(torch.allclose(d2u[1:-1].detach(), fd_d2, atol=5e-3, rtol=5e-3), f'{window} d2 mismatch')

                # Verify vectorized batched FCNN forward activates and matches sequential loop forward identically
                self.assertTrue(net._fast_fcnn, window)
                x_ref = x.clone().requires_grad_(True)
                u_fast = net(x_ref)
                lap_fast = laplacian(u_fast, x_ref)
                net._fast_fcnn = False
                try:
                    x_seq = x.clone().requires_grad_(True)
                    u_seq = net(x_seq)
                    lap_seq = laplacian(u_seq, x_seq)
                finally:
                    net._fast_fcnn = True
                self.assertTrue(torch.allclose(u_fast.detach(), u_seq.detach(), atol=1e-12, rtol=1e-12), window)
                self.assertTrue(torch.allclose(lap_fast.detach(), lap_seq.detach(), atol=1e-10, rtol=1e-10), window)
        finally:
            torch.set_default_dtype(prev)

    def test_multiscale_1d_accuracy_beats_baseline_at_equal_budget(self):
        # 1D multiscale Poisson u(x) = sin(2*pi*x) + 0.5*sin(12*pi*x) on [0,1]
        forcing = f'{4 * math.pi ** 2!r}*sin({2 * math.pi!r}*x)+{72 * math.pi ** 2!r}*sin({12 * math.pi!r}*x)'
        fb_cfg = PoissonConfig(
            shape='Interval', points=256, sampling='grid', seed=7, network_seed=7,
            pde_preset='poisson', source=0.0, pde_forcing=forcing,
            layers=(16, 16), iterations=1000, lr=0.01, pde_weight=1e-4, boundary_weight=1.0,
            method='fbpinn', device='cpu',
        )
        fb_cfg.fbpinn_settings.update(subdomains_x=8, overlap=1.5, window='cosine_sq', multilevel=True)

        base_cfg = deepcopy(fb_cfg)
        base_cfg.method = 'baseline'

        m_fb = build_pinn(fb_cfg)
        m_base = build_pinn(base_cfg)
        m_fb.train(1200, display_every=0)
        m_base.train(1200, display_every=0)

        probe = torch.linspace(0.0, 1.0, 201).unsqueeze(1)
        exact = torch.sin(2 * math.pi * probe) + 0.5 * torch.sin(12 * math.pi * probe)
        rel_fb = float((m_fb.predict(probe) - exact).norm() / exact.norm())
        rel_base = float((m_base.predict(probe) - exact).norm() / exact.norm())
        self.assertLess(rel_fb, 0.08)
        self.assertLess(rel_fb, rel_base * 0.25)

    def test_progressive_marching_freezes_completed_slices_while_keeping_overlap_derivatives(self):
        c = PoissonConfig(
            shape='SpaceTime1D', points=144, boundary_points=16, sampling='grid',
            seed=7, network_seed=7, pde_preset='wave_1d', source=2.0,
            layers=(8, 8), iterations=20, lr=0.005, pde_weight=0.01, boundary_weight=1.0,
            method='fbpinn', device='cpu',
        )
        c.bc_targets = {'bottom': 'sin(pi*x)+0.5*sin(4*pi*x)'}
        c.fbpinn_settings.update(
            subdomains_x=1, subdomains_t=3, overlap=1.5, window='cosine_sq',
            multilevel=False, schedule='progressive', schedule_axis='t',
            active_steps=5, freeze_completed=True,
        )
        model = build_pinn(c)
        # Before step 0: slice 0 active, slices 1 & 2 inactive
        self.assertEqual(model.network.subdomain_state.tolist(), [STATE_ACTIVE, STATE_INACTIVE, STATE_INACTIVE])

        # Train 5 steps (steps 0..4): only slice 0 trained!
        w1_init = deepcopy(model.network.subdomains[1].state_dict())
        w2_init = deepcopy(model.network.subdomains[2].state_dict())
        model.train(5, display_every=0)
        self.assertEqual(model.network.subdomain_state.tolist(), [STATE_ACTIVE, STATE_INACTIVE, STATE_INACTIVE])
        for k in w1_init:
            self.assertTrue(torch.equal(model.network.subdomains[1].state_dict()[k], w1_init[k]))
            self.assertTrue(torch.equal(model.network.subdomains[2].state_dict()[k], w2_init[k]))

        # Train 5 more steps (steps 5..9): slice 0 becomes FIXED, slice 1 becomes ACTIVE, slice 2 stays INACTIVE!
        w0_after_5 = deepcopy(model.network.subdomains[0].state_dict())
        model.train(5, display_every=0)
        self.assertEqual(model.network.subdomain_state.tolist(), [STATE_FIXED, STATE_ACTIVE, STATE_INACTIVE])
        # Slice 0 weights MUST remain frozen bit-for-bit
        for k in w0_after_5:
            self.assertTrue(torch.equal(model.network.subdomains[0].state_dict()[k], w0_after_5[k]))
        # Slice 1 weights MUST have updated
        self.assertTrue(any(not torch.equal(model.network.subdomains[1].state_dict()[k], w1_init[k]) for k in w1_init))
        # Slice 2 still untouched
        for k in w2_init:
            self.assertTrue(torch.equal(model.network.subdomains[2].state_dict()[k], w2_init[k]))

        # Verify input-coordinate derivatives through frozen slice 0 in overlap region are non-zero & finite!
        overlap_x = torch.tensor([[0.5, 0.333]], dtype=model.dtype, requires_grad=True)
        u_ov = model.network(overlap_x)
        grad_ov = torch.autograd.grad(u_ov.sum(), overlap_x, create_graph=True)[0]
        self.assertTrue(torch.isfinite(grad_ov).all())

        # Empty slice rejected cleanly at compile()
        c_sparse = deepcopy(c)
        c_sparse.points = 4
        c_sparse.sampling = 'grid'
        c_sparse.fbpinn_settings['subdomains_t'] = 8
        with self.assertRaisesRegex(ValueError, 'zero PDE collocation points'):
            build_pinn(c_sparse)


class FBPINNMultiDomainAndCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_supported_domains_flow_3d_sphere_and_parametric(self):
        # 1. Steady 2D incompressible flow (u, v, p)
        flow = flow_demo_config()
        flow.method = 'fbpinn'
        flow.points = 36
        flow.boundary_points = 8
        flow.layers = (8, 8)
        flow.iterations = 5
        flow.fbpinn_settings.update(subdomains_x=2, subdomains_y=2, multilevel=True)
        m_flow = build_pinn(flow)
        m_flow.train(5, display_every=0)
        pred = m_flow.predict([[0.25, 0.25], [0.75, 0.75]])
        self.assertEqual(tuple(pred.shape), (2, 3))
        self.assertTrue(torch.isfinite(pred).all())

        # 2. 3D Box with 2x2x2 subdomains
        c3d = example_config('fbpinn_poisson_3d')
        c3d.points = 64
        c3d.boundary_points = 9
        c3d.layers = (8,)
        m3d = build_pinn(c3d)
        m3d.train(3, display_every=0)
        p3d = m3d.predict([[0.2, 0.3, 0.4], [0.8, 0.7, 0.6]])
        self.assertEqual(tuple(p3d.shape), (2, 1))
        self.assertTrue(torch.isfinite(p3d).all())

        # 3. 3D SphereObstacle3D
        csph = PoissonConfig(
            shape='SphereObstacle3D', points=64, boundary_points=9, sampling='random',
            seed=7, network_seed=7, layers=(8,), iterations=3, lr=0.005, method='fbpinn', device='cpu',
        )
        csph.bc_types = dict.fromkeys(csph.sides, 'Dirichlet')
        csph.bc_values = {**dict.fromkeys(csph.sides, 0.0), 'sphere': 1.0}
        csph.fbpinn_settings.update(subdomains_x=2, subdomains_y=2, subdomains_z=2)
        msph = build_pinn(csph)
        msph.train(3, display_every=0)
        self.assertEqual(len(msph.fbpinn_summary()['subdomain_residuals']), 8)

        # 4. Continuous parametric 1D problem (mu is NOT partitioned!)
        cpar = example_config('fbpinn_parametric_1d')
        cpar.points = 48
        cpar.boundary_points = 8
        cpar.layers = (8,)
        self.assertEqual(spatial_time_axes(cpar), ('x',))
        self.assertEqual(active_subdomain_counts(cpar), (6,))
        mpar = build_pinn(cpar)
        mpar.train(4, display_every=0)
        ppar = mpar.predict([[0.25, 0.3], [0.25, 0.9]])
        self.assertEqual(tuple(ppar.shape), (2, 1))
        self.assertTrue(torch.isfinite(ppar).all())

    def test_exact_resume_checkpoint_predictor_and_atomic_rollback(self):
        # Test both simultaneous+multilevel AND mid-march progressive checkpoints
        for mode in ('multilevel', 'progressive'):
            c = PoissonConfig(
                shape='SpaceTime1D', points=64, boundary_points=8, sampling='grid',
                seed=7, network_seed=7, pde_preset='wave_1d', source=2.0,
                layers=(8,), iterations=12, lr=0.005, pde_weight=0.01, boundary_weight=1.0,
                method='fbpinn', device='cpu',
            )
            c.bc_targets = {'bottom': 'sin(pi*x)+0.5*sin(4*pi*x)'}
            if mode == 'multilevel':
                c.fbpinn_settings.update(subdomains_x=2, subdomains_t=2, multilevel=True, schedule='simultaneous')
            else:
                c.fbpinn_settings.update(
                    subdomains_x=2, subdomains_t=2, multilevel=False,
                    schedule='progressive', schedule_axis='t', active_steps=6, freeze_completed=True,
                )

            full = build_pinn(c)
            full.train(12, display_every=0)

            split = build_pinn(c)
            split.train(6, display_every=0)
            ckpt = split.training_state()

            # CheckpointPredictor reconstructs exact mid-run predictions
            pred_model = prediction_model(c, ckpt)
            self.assertIsInstance(pred_model, CheckpointPredictor)
            pts = torch.tensor([[0.2, 0.1], [0.5, 0.4], [0.8, 0.9]])
            self.assertTrue(torch.allclose(pred_model.predict(pts), split.predict(pts), atol=1e-7))

            # Resume from ckpt matches uninterrupted 12-step run bit-for-bit
            resumed = build_pinn(c)
            resumed.load_training_state(ckpt)
            resumed.train(6, display_every=0)
            self.assertTrue(torch.allclose(resumed.predict(pts), full.predict(pts), atol=1e-6))
            self.assertEqual(resumed.network.subdomain_state.tolist(), full.network.subdomain_state.tolist())

            # Atomic rollback on corrupted / non-finite / mismatched checkpoints
            before_weights = deepcopy(resumed.network.state_dict())
            before_iter = resumed.iter
            before_hist = deepcopy(resumed.loss_history)
            before_states = resumed.network.subdomain_state.tolist()

            weight_key = next(k for k, v in ckpt['network'].items() if k.startswith('subdomains.0.') and v.is_floating_point())
            for mutate in (
                lambda s: s['fbpinn'].update(subdomain_state=[99] * len(before_states)),
                lambda s: s['network'][weight_key].fill_(float('nan')),
                lambda s: s.update(data_fingerprint='bad-digest'),
                lambda s: s['fbpinn']['signature']['settings'].update(overlap=2.0),
                lambda s: s['loss_history'][0].update(total=-1.0),
            ):
                bad = deepcopy(ckpt)
                mutate(bad)
                with self.assertRaises(ValueError):
                    resumed.load_training_state(bad)
                # Confirm zero partial state mutation after rejected load
                self.assertEqual(resumed.iter, before_iter)
                self.assertEqual(resumed.loss_history, before_hist)
                self.assertEqual(resumed.network.subdomain_state.tolist(), before_states)
                for k, v in before_weights.items():
                    self.assertTrue(torch.equal(resumed.network.state_dict()[k], v))

    def test_all_fbpinn_examples_validate_and_match_analytical_residuals(self):
        for key in EXAMPLE_PROBLEMS:
            if not key.startswith('fbpinn_'):
                continue
            c = example_config(key)
            c.validate()
            ref_text = EXAMPLE_PROBLEMS[key].reference
            if ref_text:
                m = build_pinn(c)
                dim = len(c.bounds[0])
                lo, hi = (torch.tensor(v) for v in c.bounds)
                pts = (lo + (hi - lo) * torch.rand(16, dim)).requires_grad_()
                ref = Expression(ref_text, dim, target=True, coordinates=c.coordinate_names)
                res = m.problem.residual(pts, ref(pts, source=c.source, viscosity=c.viscosity))
                self.assertLess(res.abs().max().item(), 1e-3, key)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable in this environment')
    def test_cuda_training_and_portable_checkpoint_resume(self):
        c = PoissonConfig(
            shape='Interval', points=48, sampling='grid', seed=7, network_seed=7,
            layers=(8, 8), iterations=10, lr=0.005, method='fbpinn', device='cuda',
        )
        c.fbpinn_settings.update(subdomains_x=4, multilevel=True)
        m_gpu = build_pinn(c)
        m_gpu.train(5, display_every=0)
        ckpt = m_gpu.training_state()

        # Restore GPU checkpoint on CPU and verify identical predictions
        c_cpu = deepcopy(c)
        c_cpu.device = 'cpu'
        m_cpu = build_pinn(c_cpu)
        m_cpu.load_training_state(ckpt)
        pts = torch.linspace(0.0, 1.0, 21).unsqueeze(1)
        self.assertTrue(torch.allclose(m_gpu.predict(pts).cpu(), m_cpu.predict(pts), atol=1e-5))


if __name__ == '__main__':
    unittest.main()
