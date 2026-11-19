"""Unit and scientific tests for the PyTorch + Margam JAXPI2 / arXiv:2507.08972v2 port."""
from functools import partial
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch
from margam.research import (
    ArchConfig,
    Jaxpi2CropTrainer,
    Jaxpi2MultiStageCropTrainer,
    Jaxpi2Optimizer,
    Jaxpi2TrainingConfig,
    MultiStageSolution,
    OptimConfig,
    PINNProblem,
    PointConstraint,
    Trainer,
    create_arch,
    derivative,
    jaxpi2_causal_residual_losses,
    multi_stage_navier_stokes,
    navier_stokes,
    pseudo_time_residuals,
    stable_swish,
    vorticity_3d,
)
from margam.research.jaxpi2 import (
    CropBoundaryReference,
    CropSampler,
    Domain,
    FourierEmbs,
    SpatialPools,
    SphereField,
    compute_pts_weights,
    get_orthogonal_matrix,
    get_orthogonal_matrix_qr,
    reject_sampling,
)


def analytic_reference(axes, dt=0.5):
    t, x, y, z = np.meshgrid(np.arange(3) * dt, *axes, indexing="ij")
    return np.stack(
        (
            4 + 2 * t + 3 * x + 5 * y - 7 * z,
            1 + t + x + 2 * y + 3 * z,
            2 * t - x + y,
            3 * t + y - z,
        ),
        axis=1,
    )


class Jaxpi2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self._orig_dtype = torch.get_default_dtype()

    def tearDown(self):
        torch.set_default_dtype(self._orig_dtype)

    def _check_swish_third_derivative(self, device, dtype):
        x = torch.linspace(-30, 30, 1201, dtype=dtype, device=device, requires_grad=True)
        output = stable_swish(x)
        torch.testing.assert_close(
            output,
            torch.nn.functional.silu(x),
            atol=1e-6 if dtype == torch.float32 else 1e-14,
            rtol=1e-6 if dtype == torch.float32 else 1e-14,
        )
        d = output
        for _ in range(3):
            d = torch.autograd.grad(d.sum(), x, create_graph=True)[0]
            self.assertTrue(torch.isfinite(d).all())

        # Analytic 3rd derivative of x * sigmoid(x): s(1-s)[3(1-2s) + x(1 - 6s + 6s^2)]
        z = x.detach().cpu().double().numpy()
        s = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
        sm = s * (1.0 - s)
        expected = sm * (3.0 * (1.0 - 2.0 * s) + z * (1.0 - 6.0 * s + 6.0 * s * s))
        actual = d.detach().cpu().double()
        target = torch.from_numpy(expected)
        torch.testing.assert_close(
            actual,
            target,
            atol=2e-6 if dtype == torch.float32 else 1e-13,
            rtol=1e-4 if dtype == torch.float32 else 1e-11,
        )

    def test_swish_third_derivative_is_finite_and_matches_analytic_expression(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(device="cpu", dtype=dtype):
                self._check_swish_third_derivative("cpu", dtype)
        if torch.cuda.is_available():
            for dtype in (torch.float32, torch.float64):
                with self.subTest(device="cuda:0", dtype=dtype):
                    self._check_swish_third_derivative("cuda:0", dtype)

    def test_piratenet_architectures_rwf_and_periodicity(self):
        torch.manual_seed(7)
        for arch_name in ("Mlp", "ModifiedMlp", "PirateNet"):
            for reparam in (None, dict(type="weight_fact", mean=1.0, stddev=0.1)):
                cfg = ArchConfig(
                    arch_name=arch_name,
                    input_dim=4,
                    out_dim=4,
                    num_layers=2,
                    hidden_dim=8,
                    activation="swish",
                    nonlinearity=0.0,
                    periodicity=dict(period=(2.0, math.pi), axis=(1, 2), trainable=(False, True)),
                    fourier_emb=dict(embed_scale=1.5, embed_dim=8),
                    reparam=reparam,
                )
                net = create_arch(cfg).double()
                q = torch.tensor([[0.2, 0.3, -0.4, 0.5], [0.7, -1.1, 0.8, -0.2]], dtype=torch.float64)
                y = net(q)
                self.assertEqual(y.shape, (2, 4))
                self.assertTrue(torch.isfinite(y).all())
                q_shift = q.clone()
                q_shift[:, 1] += math.pi
                torch.testing.assert_close(y, net(q_shift), atol=1e-13, rtol=1e-13)
                self.assertTrue(len(net.reference_parameter_map()) > 0)

        # PirateNet with alpha=0 reduces to a linear combination of the first-layer Fourier features
        cfg0 = ArchConfig(
            arch_name="PirateNet",
            input_dim=4,
            out_dim=4,
            num_layers=2,
            hidden_dim=8,
            activation="swish",
            nonlinearity=0.0,
            fourier_emb=dict(embed_scale=1.0, embed_dim=8),
        )
        pnet = create_arch(cfg0).double()
        self.assertEqual(pnet.alphas(), (0.0, 0.0))
        q = torch.randn(3, 4, dtype=torch.float64)
        emb = pnet.fourier_emb(q)
        torch.testing.assert_close(pnet(q), pnet.output(emb), atol=1e-14, rtol=1e-14)

        with self.assertRaisesRegex(AssertionError, "hidden_dim"):
            create_arch(
                ArchConfig(arch_name="PirateNet", hidden_dim=8, fourier_emb=dict(embed_scale=1.0, embed_dim=6))
            )
        with self.assertRaisesRegex(AssertionError, "even"):
            FourierEmbs(4, embed_scale=1.0, embed_dim=7)

    def test_jaxpi2_componentwise_causal_and_pseudo_time(self):
        r = torch.tensor(
            [[1.0, 2.0, 0.5, 1.5], [3.0, 0.0, 1.5, 0.5], [2.0, 1.0, 1.0, 2.0], [0.0, 2.0, 2.0, 0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        losses, min_gamma, comp_gamma = jaxpi2_causal_residual_losses(
            r, chunks=2, epsilon=0.5, componentwise=True
        )
        self.assertFalse(min_gamma.requires_grad)
        self.assertFalse(comp_gamma.requires_grad)
        chunk0 = 0.5 * (r[0].detach().square() + r[1].detach().square())
        chunk1 = 0.5 * (r[2].detach().square() + r[3].detach().square())
        expected_gamma1 = torch.exp(-0.5 * chunk0)
        torch.testing.assert_close(comp_gamma[0], torch.ones(4, dtype=torch.float64))
        torch.testing.assert_close(comp_gamma[1], expected_gamma1)
        torch.testing.assert_close(min_gamma[1], expected_gamma1.min())
        torch.testing.assert_close(losses, 0.5 * (chunk0 + expected_gamma1 * chunk1))

        # Shared-minimum causal gating (componentwise=False)
        losses_shared, _, _ = jaxpi2_causal_residual_losses(
            r, chunks=2, epsilon=0.5, componentwise=False
        )
        torch.testing.assert_close(losses_shared, 0.5 * (chunk0 + expected_gamma1.min() * chunk1))

        # Pseudo-time residual shift and weight update
        sol_pred = torch.ones_like(r) * 2.0
        sol_prev = torch.ones_like(r) * 1.5
        shifted = pseudo_time_residuals(r, sol_pred, sol_prev, [1.0, 2.0, 3.0, 4.0])
        torch.testing.assert_close(
            shifted, r + torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
        )
        w_pts = compute_pts_weights(r, r * 0.5, r * 10.0, sol_pred, sol_prev, shrink_enabled=True)
        self.assertEqual(w_pts.shape, (4,))
        self.assertTrue(torch.all((w_pts >= 1e-2) & (w_pts <= 100.0)))

    def test_multistage_linearized_navier_stokes_exact_quadratic_remainder_and_vorticity(self):
        """Verify r_NS(prev + eps*diff) - eps * r_stage(diff; prev) == eps^2 * (diff . grad) diff."""
        q = torch.tensor(
            [[0.2, 1.1, -0.3, 0.6], [0.7, -0.9, 0.4, 1.2]],
            dtype=torch.float64,
            requires_grad=True,
        )

        def prev_fn(z):
            t, x, y, w = z.split(1, dim=1)
            return torch.cat(
                (t + x * x + 2 * y * w, 2 * t - x * y + w * w, 3 * t + y * y - x * w, t * x + y * y + w * w),
                dim=1,
            )

        def diff_fn(z):
            t, x, y, w = z.split(1, dim=1)
            return torch.cat(
                (x * y + t * w, y * w - x * x, x * w + y * t, x - 2 * y + 3 * w),
                dim=1,
            )

        eps = 1e-2
        reynolds = 300.0
        diff_pred = diff_fn(q)
        r_stage = multi_stage_navier_stokes(q, diff_pred, prev_fn, eps=eps, reynolds=reynolds)

        q_comp = q.detach().clone().requires_grad_(True)
        comp_pred = prev_fn(q_comp) + eps * diff_fn(q_comp)
        r_comp = navier_stokes(q_comp, comp_pred, reynolds=reynolds)

        t, x, y, z = q.detach().split(1, dim=1)
        ud, vd, wd, _ = diff_pred.detach().split(1, dim=1)
        nl_u = ud * y + vd * x + wd * t
        nl_v = ud * (-2 * x) + vd * z + wd * y
        nl_w = ud * z + vd * t + wd * x
        expected_rem = (eps ** 2) * torch.cat((nl_u, nl_v, nl_w, torch.zeros_like(nl_u)), dim=1)

        torch.testing.assert_close(r_comp - eps * r_stage, expected_rem, atol=1e-13, rtol=1e-13)

        vor = vorticity_3d(q, diff_pred)
        expected_vor = torch.cat((t - y, t - z, -2 * x - x), dim=1)
        torch.testing.assert_close(vor, expected_vor, atol=1e-13, rtol=1e-13)

        # MultiStageSolution and reject_sampling
        s0 = torch.nn.Linear(4, 4).double()
        s1 = torch.nn.Linear(4, 4).double()
        ms_sol = MultiStageSolution([s0, s1], [1.0, 1e-2])
        torch.testing.assert_close(ms_sol.forward_prev(q), s0(q))
        torch.testing.assert_close(ms_sol(q), s0(q) + 1e-2 * s1(q))

        gen = torch.Generator().manual_seed(3)
        dom = torch.tensor([[0.0, 1.0], [-1.0, 1.0]], dtype=torch.float64)
        sampled = reject_sampling(gen, 16, lambda pts: pts[:, 0].square(), dom, threshold=0.25, batch_size=32)
        self.assertEqual(sampled.shape, (16, 2))

    def test_soap_orthogonal_matrix_rocm_fallback_and_research_checkpoint(self):
        real_eigh = torch.linalg.eigh
        call_count = {"n": 0}

        def flaky_eigh(mat, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise torch.linalg.LinAlgError("simulated rocSOLVER syevd repeated-eigenvalue failure")
            return real_eigh(mat, *args, **kwargs)

        with patch.object(torch.linalg, "eigh", side_effect=flaky_eigh):
            gg_zero = torch.zeros((16, 16), dtype=torch.float32)
            q = get_orthogonal_matrix(gg_zero, torch.float32)
        self.assertIsNotNone(q)
        self.assertEqual(q.shape, (16, 16))
        self.assertEqual(call_count["n"], 2)
        eye = torch.eye(16, dtype=torch.float32)
        self.assertTrue(torch.allclose(q.T @ q, eye, atol=1e-5))

        # Sub-FLT_MIN preconditioner underflow test in get_orthogonal_matrix_qr
        tiny_gg = [torch.eye(8, dtype=torch.float32) * 1e-21, torch.eye(8, dtype=torch.float32) * 1e-21]
        init_q = [torch.eye(8, dtype=torch.float32), torch.eye(8, dtype=torch.float32)]
        exp_avg_sq = torch.zeros((8, 8), dtype=torch.float32)
        q_out, _ = get_orthogonal_matrix_qr(tiny_gg, init_q, exp_avg_sq, torch.float64)
        for q_mat in q_out:
            self.assertTrue(torch.allclose(q_mat.T @ q_mat, torch.eye(8), atol=1e-5))

        # Standard margam.research.Trainer checkpoint save/restore with Jaxpi2Optimizer (SOAP + Schedule-Free)
        def make_fixed_trainer():
            torch.manual_seed(11)
            net = torch.nn.Sequential(
                torch.nn.Linear(1, 8),
                torch.nn.Tanh(),
                torch.nn.Linear(8, 1),
            ).double()
            prob = PINNProblem(
                lambda x, u: derivative(u, x, 0, 2) + 2.0,
                torch.linspace(0.0, 1.0, 18, dtype=torch.float64)[1:-1, None],
                [PointConstraint("bc", torch.tensor([[0.0], [1.0]], dtype=torch.float64))],
            )
            ocfg = OptimConfig(
                optimizer="soap",
                learning_rate=1e-3,
                warmup_steps=2,
                decay_steps=50,
                schedule_free=True,
                precondition_frequency=2,
                qr_dtype="float64",
            )
            return Trainer(prob, net, checkpoint_id="jaxpi2-soap-fixed-v1").compile(
                optimizer=partial(Jaxpi2Optimizer, config=ocfg),
                lr=ocfg.learning_rate,
                device="cpu",
            )

        full = make_fixed_trainer()
        full.train(4, display_every=0)
        split = make_fixed_trainer()
        split.train(2, display_every=0)
        with TemporaryDirectory() as tmp_dir:
            ckpt = Path(tmp_dir) / "fixed_soap.pt"
            split.save_checkpoint(ckpt)
            restored = make_fixed_trainer().load_checkpoint(ckpt)
            restored.train(2, display_every=0)

        self.assertEqual(full.loss_history, restored.loss_history)
        for k, v in full.eval_state_dict().items():
            self.assertTrue(torch.equal(v, restored.eval_state_dict()[k]))

    def _run_crop_trainer_lifecycle(self, device):
        torch.set_default_dtype(torch.float64)
        domain = Domain()
        axes = [np.linspace(a, b, 5) for a, b in zip(domain.lower, domain.upper)]
        reference = CropBoundaryReference(analytic_reference(axes), axes, 0.5, 1.0)
        pools = SpatialPools.from_domain(domain, volume_count=32, boundary_count=16, seed=42)

        def make_trainer(include_pressure_ic=True):
            torch.manual_seed(19)
            q = torch.tensor([[0.0, 1.0, 0.7, 0.6], [0.0, 2.0, -0.7, 0.8], [0.0, 3.0, 0.9, -0.7]])
            values = torch.tensor([[1.0, 0.1, 0.2, -0.1], [0.9, -0.2, 0.1, 0.2], [0.8, 0.2, -0.1, -0.2]])
            sampler = CropSampler(
                domain,
                q,
                values,
                reference=reference,
                residual_count=8,
                boundary_count=2,
                initial_count=3,
                spatial_pools=pools,
            )
            acfg = ArchConfig(
                arch_name="PirateNet",
                input_dim=4,
                out_dim=4,
                num_layers=1,
                hidden_dim=4,
                activation="swish",
                fourier_emb=dict(embed_scale=1.0, embed_dim=4),
                nonlinearity=0.2,
            )
            net = SphereField(create_arch(acfg))
            tcfg = Jaxpi2TrainingConfig(
                reynolds=300.0,
                causal_enabled=True,
                chunks=2,
                balance_start=0,
                balance_every=2,
                pseudo_time_enabled=True,
                pseudo_time_strategy="dynamic",
                pts_start=1,
                pts_every=2,
                include_pressure_ic=include_pressure_ic,
                optim=OptimConfig(
                    optimizer="soap",
                    learning_rate=1e-3,
                    warmup_steps=2,
                    decay_steps=100,
                    schedule_free=True,
                    precondition_frequency=2,
                    qr_dtype="float64",
                ),
            )
            return Jaxpi2CropTrainer(net, sampler, tcfg).compile(device=device)

        uninterrupted = make_trainer()
        uninterrupted.train(4, display_every=0)

        split = make_trainer()
        split.train(2, display_every=0)
        with TemporaryDirectory() as tmp_dir:
            ckpt = Path(tmp_dir) / "jaxpi2_ckpt.pt"
            split.save_checkpoint(ckpt)
            resumed = make_trainer().load_checkpoint(ckpt)
            resumed.train(2, display_every=0)

        for a, b in zip(uninterrupted.network.parameters(), resumed.network.parameters()):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(uninterrupted.weights, resumed.weights)
        self.assertEqual(uninterrupted.pts_weights, resumed.pts_weights)
        eval_a = uninterrupted.eval_state_dict()
        eval_b = resumed.eval_state_dict()
        for k in eval_a:
            self.assertTrue(torch.equal(eval_a[k], eval_b[k]))

        # Window transfer learning (including velocity-only IC option for Window 2)
        next_window = make_trainer(include_pressure_ic=False).transfer_from(resumed)
        self.assertNotIn("p_ic", next_window.weights)
        for a, b in zip(resumed.network.parameters(), next_window.network.parameters()):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(next_window.optimizer._step_count, 0)

        # Stage-1 Multi-Stage correction trainer on the cropped sphere problem
        prev_net = SphereField(
            create_arch(
                ArchConfig(
                    arch_name="PirateNet",
                    input_dim=4,
                    out_dim=4,
                    num_layers=1,
                    hidden_dim=4,
                    activation="swish",
                    fourier_emb=dict(embed_scale=1.0, embed_dim=4),
                    nonlinearity=0.2,
                )
            )
        )
        prev_net.load_state_dict(resumed.eval_state_dict())
        stage1_net = SphereField(
            create_arch(
                ArchConfig(
                    arch_name="PirateNet",
                    input_dim=4,
                    out_dim=4,
                    num_layers=1,
                    hidden_dim=4,
                    activation="swish",
                    fourier_emb=dict(embed_scale=2.0, embed_dim=4),
                    nonlinearity=0.2,
                )
            )
        )
        ms_cfg = Jaxpi2TrainingConfig(
            reynolds=300.0,
            causal_enabled=False,
            balance_start=0,
            balance_every=2,
            optim=OptimConfig(
                optimizer="soap",
                learning_rate=1e-3,
                warmup_steps=1,
                decay_steps=100,
                schedule_free=True,
                precondition_frequency=2,
                qr_dtype="float64",
            ),
        )
        ms_trainer = Jaxpi2MultiStageCropTrainer(
            stage1_net, prev_net, 1e-3, resumed.sampler, ms_cfg
        ).compile(device=device)
        ms_trainer.train(2, display_every=0)
        self.assertEqual(ms_trainer.iter, 2)
        self.assertTrue(all(math.isfinite(v) for v in ms_trainer.weights.values()))

    def test_jaxpi2_crop_trainer_soap_schedule_free_resume_transfer_and_multistage_cpu(self):
        self._run_crop_trainer_lifecycle("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_jaxpi2_crop_trainer_soap_schedule_free_resume_transfer_and_multistage_cuda(self):
        self._run_crop_trainer_lifecycle("cuda:0")


if __name__ == "__main__":
    unittest.main()
