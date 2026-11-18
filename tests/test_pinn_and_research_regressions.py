from copy import deepcopy
import unittest
import torch
from torch import nn

from margam.components.problem import PINNProblem, PointConstraint
from margam.methods.pinn import PINN
from margam.research import StepDecay, Trainer
from margam.templates.poisson import PoissonConfig, build_pinn


class PINNAndResearchRegressionTests(unittest.TestCase):
    def _simple_problem_and_net(self):
        torch.manual_seed(0)
        x = torch.linspace(0.1, 0.9, 6)[:, None]
        bc_x = torch.tensor([[0.0], [1.0]])
        problem = PINNProblem(
            lambda pts, u: u,
            x,
            [PointConstraint('bc', bc_x, 0.0)],
        )
        net = nn.Sequential(nn.Linear(1, 4), nn.Tanh(), nn.Linear(4, 1))
        return problem, net

    def test_pinn_rejects_nonfinite_gradients_without_stepping(self):
        torch.manual_seed(0)
        x = torch.tensor([[0.2], [0.5]])
        bc_x = torch.tensor([[0.0], [1.0]])
        # At (u - u.detach()) == 0, 1 + sqrt(0) = 1 has finite MSE = 1
        # while d/du sqrt(u - u.detach()) = 1 / (2 * 0) = +inf.
        problem = PINNProblem(
            lambda pts, u: 1.0 + torch.sqrt(u - u.detach()),
            x,
            [PointConstraint('bc', bc_x, 0.0)],
        )
        net = nn.Sequential(nn.Linear(1, 4), nn.Tanh(), nn.Linear(4, 1))
        model = PINN(problem, net).compile(lr=1e-2, device='cpu')
        before = deepcopy(model.network.state_dict())
        with self.assertRaises(FloatingPointError):
            model.train(1, display_every=0)
        self.assertEqual(model.iter, 0)
        self.assertEqual(model.loss_history, [])
        for k, v in before.items():
            torch.testing.assert_close(model.network.state_dict()[k], v, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.network.parameters()))

    def test_pinn_and_hard_dirichlet_reject_changed_global_weights_on_resume(self):
        for method in ('baseline', 'hard_dirichlet'):
            with self.subTest(method=method):
                c1 = PoissonConfig(shape='Interval', method=method, points=8, boundary_points=2, layers=(4,), pde_weight=1.0, boundary_weight=1.0)
                m1 = build_pinn(c1)
                m1.train(2, display_every=0)
                state = m1.training_state()

                c2 = deepcopy(c1)
                c2.pde_weight = 2.5
                with self.assertRaises(ValueError):
                    build_pinn(c2).load_training_state(state)

                if method == 'baseline':
                    c3 = deepcopy(c1)
                    c3.boundary_weight = 4.0
                    with self.assertRaises(ValueError):
                        build_pinn(c3).load_training_state(state)

    def test_pinn_load_training_state_is_atomic_rejects_negative_second_moment_and_allows_script_steps(self):
        problem, net = self._simple_problem_and_net()
        m1 = PINN(problem, deepcopy(net)).compile(lr=1e-3, device='cpu')
        m1.train(2, display_every=0)
        good_state = m1.training_state()

        # Corrupt optimizer tensor: live target model must remain intact on failure
        target = PINN(problem, deepcopy(net)).compile(lr=1e-3, device='cpu')
        target.train(1, display_every=0)
        live_before = deepcopy(target.network.state_dict())
        live_iter = target.iter

        bad_nan = deepcopy(good_state)
        first_key = next(iter(bad_nan['optimiser']['state']))
        bad_nan['optimiser']['state'][first_key]['exp_avg'] = torch.full_like(
            bad_nan['optimiser']['state'][first_key]['exp_avg'], float('nan')
        )
        with self.assertRaises(ValueError):
            target.load_training_state(bad_nan)
        self.assertEqual(target.iter, live_iter)
        for k, v in live_before.items():
            torch.testing.assert_close(target.network.state_dict()[k], v, rtol=0, atol=0)

        # Negative exp_avg_sq must be rejected before Adam produces NaNs
        bad_neg = deepcopy(good_state)
        bad_neg['optimiser']['state'][first_key]['exp_avg_sq'] = -torch.ones_like(
            bad_neg['optimiser']['state'][first_key]['exp_avg_sq']
        )
        with self.assertRaises(ValueError):
            target.load_training_state(bad_neg)

        # Missing required per-parameter Adam fields (e.g., exp_avg_sq or max_exp_avg_sq) must be rejected
        # before overwriting live network or optimizer state
        for missing_field in ('step', 'exp_avg', 'exp_avg_sq'):
            bad_missing = deepcopy(good_state)
            del bad_missing['optimiser']['state'][first_key][missing_field]
            with self.assertRaises(ValueError):
                target.load_training_state(bad_missing)
            self.assertEqual(target.iter, live_iter)
            for k, v in live_before.items():
                torch.testing.assert_close(target.network.state_dict()[k], v, rtol=0, atol=0)

        # AMSGrad enabled requires max_exp_avg_sq in baseline PINN too
        ams_model = PINN(problem, deepcopy(net)).compile(
            lr=1e-3,
            device='cpu',
            optimiser=lambda params, lr: torch.optim.Adam(params, lr=lr, amsgrad=True),
        )
        ams_model.train(1, display_every=0)
        ams_state = ams_model.training_state()
        ams_first = next(iter(ams_state['optimiser']['state']))
        del ams_state['optimiser']['state'][ams_first]['max_exp_avg_sq']
        ams_target = PINN(problem, deepcopy(net)).compile(
            lr=1e-3,
            device='cpu',
            optimiser=lambda params, lr: torch.optim.Adam(params, lr=lr, amsgrad=True),
        )
        with self.assertRaises(ValueError):
            ams_target.load_training_state(ams_state)

        # DSGNAR uses trust-region updates (optimiser.state == {}), so iter > 0 must load cleanly,
        # whereas non-empty Adam state on DSGNAR or empty Adam state on trained PINN must fail
        dsg_cfg = PoissonConfig(shape='Interval', method='dsgnar', points=6, boundary_points=2, layers=(4,), device='cpu')
        dsg_cfg.dsgnar_settings['sketch_size'] = 4
        dsg_model = build_pinn(dsg_cfg)
        dsg_model.train(1, 0)
        dsg_state = dsg_model.training_state()
        self.assertEqual(dsg_state['iter'], 1)
        self.assertEqual(dsg_state['optimiser']['state'], {})
        dsg_restored = build_pinn(dsg_cfg).load_training_state(dsg_state)
        self.assertEqual(dsg_restored.iter, 1)
        bad_empty_adam = deepcopy(good_state)
        bad_empty_adam['optimiser']['state'] = {}
        with self.assertRaises(ValueError):
            target.load_training_state(bad_empty_adam)

        # Script-level checkpoints beyond 100000 steps must load cleanly
        long_state = deepcopy(good_state)
        long_state['iter'] = 100001
        long_state['loss_history'] = [long_state['loss_history'][0]] * 100001
        long_state['training_history'] = [long_state['training_history'][0]] * 100001
        restored = PINN(problem, deepcopy(net)).compile(lr=1e-3, device='cpu')
        restored.load_training_state(long_state)
        self.assertEqual(restored.iter, 100001)

        # Checkpoint fingerprints must hash the frozen retained targets, not re-run callbacks
        captured_val = [0.0]
        x_pts = torch.linspace(0.1, 0.9, 6)[:, None]
        bc_pts = torch.tensor([[0.0], [1.0]])
        mut_prob = PINNProblem(
            lambda pts, u: u,
            x_pts,
            [PointConstraint('bc', bc_pts, lambda pts: torch.full_like(pts, captured_val[0]))],
        )
        model_a = PINN(mut_prob, deepcopy(net)).compile(lr=1e-3, device='cpu')
        model_a.train(1, display_every=0)
        captured_val[0] = 1.0
        state_a = model_a.training_state()
        model_b = PINN(mut_prob, deepcopy(net)).compile(lr=1e-3, device='cpu')
        with self.assertRaises(ValueError):
            model_b.load_training_state(state_a)

        # A torch.rand_like target evaluated once at compile must accept its own checkpoint
        rand_prob = PINNProblem(
            lambda pts, u: u,
            x_pts,
            [PointConstraint('bc', bc_pts, lambda pts: torch.rand_like(pts))],
        )
        rand_model = PINN(rand_prob, deepcopy(net)).compile(lr=1e-3, device='cpu')
        rand_model.train(1, display_every=0)
        rand_state = rand_model.training_state()
        rand_model.load_training_state(rand_state)
        self.assertEqual(rand_model.iter, 1)

    def test_transcendental_targets_canonicalize_across_devices(self):
        # Verify callable target receives CPU tensor even when compiled on CUDA (if available)
        seen_devices = []
        x = torch.linspace(0.1, 0.9, 8)[:, None]
        bc_x = torch.tensor([[0.0], [1.0]])

        def target_fn(pts):
            seen_devices.append(pts.device.type)
            return torch.sin(3.141592653589793 * pts)

        problem = PINNProblem(lambda pts, u: u, x, [PointConstraint('bc', bc_x, target_fn)])
        net = nn.Sequential(nn.Linear(1, 4), nn.Tanh(), nn.Linear(4, 1))
        cpu_model = PINN(problem, deepcopy(net)).compile(lr=1e-3, device='cpu')
        self.assertTrue(all(d == 'cpu' for d in seen_devices))

        if torch.cuda.is_available():
            seen_devices.clear()
            cuda_model = PINN(problem, deepcopy(net)).compile(lr=1e-3, device='cuda')
            self.assertTrue(all(d == 'cpu' for d in seen_devices))
            self.assertEqual(cpu_model.data_fingerprint(), cuda_model.data_fingerprint())

            for method in ('baseline', 'self_adaptive', 'lra', 'gpinn'):
                cfg = PoissonConfig(
                    shape='Interval',
                    method=method,
                    points=12,
                    boundary_points=2,
                    layers=(8, 8),
                    pde_expression='laplace(u) + sin(pi * x)',
                    bc_targets={'left': 'sin(0.3)', 'right': 'cos(0.7)'},
                    device='cuda',
                )
                m_cuda = build_pinn(cfg)
                m_cuda.train(2, display_every=0)
                state = m_cuda.training_state()
                cfg_cpu = deepcopy(cfg)
                cfg_cpu.device = 'cpu'
                m_cpu = build_pinn(cfg_cpu)
                m_cpu.load_training_state(state)
                m_cpu.train(1, display_every=0)
                self.assertEqual(m_cpu.iter, 3)

    def test_research_trainer_precompile_properties_and_checkpoint_guards(self):
        problem, net = self._simple_problem_and_net()
        trainer = Trainer(problem, deepcopy(net), checkpoint_id='test-v1')
        self.assertIsNone(trainer.pointwise_microbatch)
        self.assertIsNone(trainer.schedule)
        with self.assertRaises(RuntimeError):
            trainer.scheduled_rates()

        trainer.compile(lr=1e-3, device='cpu')
        with self.assertRaises(ValueError):
            trainer.scheduled_rates()

        # Non-finite gradient in full-batch research Trainer must clear p.grad
        bad_problem = PINNProblem(
            lambda pts, u: 1.0 + torch.sqrt(u - u.detach()),
            torch.tensor([[0.2], [0.5]]),
            [PointConstraint('bc', torch.tensor([[0.0], [1.0]]), 0.0)],
        )
        bad_trainer = Trainer(bad_problem, deepcopy(net)).compile(lr=1e-3, device='cpu')
        with self.assertRaises(FloatingPointError):
            bad_trainer.train(1, display_every=0)
        self.assertTrue(all(p.grad is None for p in bad_trainer.network.parameters()))

        # AMSGrad checkpoint missing max_exp_avg_sq and negative exp_avg_sq must fail restore
        ams = Trainer(problem, deepcopy(net), checkpoint_id='ams-v1').compile(
            optimizer=lambda params, lr: torch.optim.Adam(params, lr=lr, amsgrad=True),
            lr=1e-3,
            device='cpu',
        )
        ams.train(2, display_every=0)
        ams_state = ams.training_state()

        missing_max = deepcopy(ams_state)
        first_id = next(iter(missing_max['optimizer']['state']))
        del missing_max['optimizer']['state'][first_id]['max_exp_avg_sq']
        fresh_ams = Trainer(problem, deepcopy(net), checkpoint_id='ams-v1').compile(
            optimizer=lambda params, lr: torch.optim.Adam(params, lr=lr, amsgrad=True),
            lr=1e-3,
            device='cpu',
        )
        with self.assertRaises(ValueError):
            fresh_ams.load_training_state(missing_max)

        neg_sq = deepcopy(ams_state)
        neg_sq['optimizer']['state'][first_id]['exp_avg_sq'] = -torch.ones_like(
            neg_sq['optimizer']['state'][first_id]['exp_avg_sq']
        )
        with self.assertRaises(ValueError):
            fresh_ams.load_training_state(neg_sq)

        # Scheduled trainer with dirtied lr can still be restored from a valid checkpoint
        sched_trainer = Trainer(problem, deepcopy(net), checkpoint_id='sched-v1').compile(
            lr=1e-3,
            device='cpu',
            schedule=StepDecay(every=1, gamma=0.5),
        )
        sched_trainer.train(2, display_every=0)
        sched_state = sched_trainer.training_state()
        sched_trainer.optimizer.param_groups[0]['lr'] = 0.999
        sched_trainer.load_training_state(sched_state)
        self.assertAlmostEqual(sched_trainer.optimizer.param_groups[0]['lr'], 1e-3 * 0.25)


if __name__ == '__main__':
    unittest.main()
