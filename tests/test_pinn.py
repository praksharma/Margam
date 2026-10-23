"""Behavioural checks for baseline training and the legacy entry point."""
import contextlib
import io
import unittest
import torch
import DeepINN as dp
from DeepINN.components import weighted_loss


def poisson_residual(x, u):
    return dp.utils.laplacian(u, x) + 2


def problem():
    return dp.PINNProblem(
        poisson_residual,
        lambda: torch.linspace(0, 1, 34)[1:-1, None],
        [dp.PointConstraint("boundary", torch.tensor([[0.0], [1.0]]))],
    )


def network():
    return torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(),
                               torch.nn.Linear(16, 16), torch.nn.Tanh(),
                               torch.nn.Linear(16, 1))


class PINNTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_poisson_converges(self):
        model = dp.PINN(problem(), network()).compile(lr=0.01, device="cpu")
        initial = sum(v.item() for v in model.compute_losses().values())
        model.train(1000, display_every=0)
        x = torch.linspace(0, 1, 101)[:, None]
        rmse = torch.mean((model.predict(x) - x * (1-x))**2).sqrt().item()
        final = sum(v.item() for v in model.compute_losses().values())
        self.assertLess(rmse, 0.02)
        self.assertLess(final, initial * 0.01)
        self.assertEqual(model.iter, 1000)

    def test_exact_steps_resume_display_and_recompile(self):
        model = dp.PINN(problem(), network()).compile(device="cpu")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            model.train(3, display_every=2)
            model.train(2, display_every=2)
            model.train(0)
        self.assertEqual(model.iter, 5)
        self.assertEqual(len(model.training_history), 5)
        self.assertEqual(len(output.getvalue().splitlines()), 2)
        self.assertEqual(set(model.loss_history[0]), {"pde", "boundary", "total"})
        model.compile(device="cpu")
        self.assertEqual(model.iter, 0)
        self.assertEqual(model.training_history, [])
        model.train(1, display_every=0)

    def test_weighted_terms_and_shape_validation(self):
        self.assertEqual(weighted_loss({"pde": torch.tensor(2.), "bc": torch.tensor(3.)},
                                       {"bc": 4}).item(), 14)
        for weights in ({"typo": 1}, {"pde": -1}, {"pde": float("nan")}):
            with self.assertRaises(ValueError):
                dp.PINN(problem(), network(), weights=weights).compile(device="cpu")
        p = problem()
        p.constraints[0].target = torch.zeros(2, 2)
        with self.assertRaises(ValueError):
            dp.PINN(p, network()).compile(device="cpu").train(1, display_every=0)

    def test_derivative_condition_float64_and_input_ownership(self):
        x = torch.linspace(0, 1, 5, dtype=torch.float64)[:, None]
        p = dp.PINNProblem(lambda x, u: u, x, [dp.PointConstraint(
            "neumann", x, lambda x: torch.ones_like(x),
            lambda x, u: torch.autograd.grad(u.sum(), x, create_graph=True)[0])])
        model = dp.PINN(p, torch.nn.Linear(1, 1).double()).compile(device="cpu")
        model.train(3, display_every=0)
        self.assertFalse(x.requires_grad)
        self.assertIsNone(x.grad)
        self.assertEqual(model.predict(x).dtype, torch.float64)
        model.network.eval()
        model.predict(x)
        self.assertFalse(model.network.training)

    def test_invalid_state_and_constraints(self):
        model = dp.PINN(problem(), network())
        with self.assertRaises(RuntimeError):
            model.train(1)
        model.compile(device="cpu")
        for args in ((-1, 1), (1.5, 1), (1, -1)):
            with self.assertRaises(ValueError):
                model.train(*args)
        p = problem()
        p.constraints = [p.constraints[0], p.constraints[0]]
        with self.assertRaises(ValueError):
            dp.PINN(p, network()).compile(device="cpu")

    def test_legacy_domain_interface(self):
        space = dp.spaces.R1('x')
        interval = dp.domains.Interval(space, 0, 1)
        domain = dp.domain.Generic(
            poisson_residual,
            dp.constraint.PDE(interval, "grid", 16),
            dp.constraint.DirichletBC(interval, lambda x: x[:, 0]*0, "grid", 2))
        model = dp.Model(domain, network())
        model.compile("adam", 0.001, "MSE", "cpu")
        model.train(iterations=3, display_every=0)
        self.assertEqual(model.iter, 3)
        self.assertEqual(set(model.compute_losses()), {"pde", "boundary"})

    def test_fully_connected_batch_invariance_and_derivatives(self):
        net = dp.nn.FullyConnected([1, 8, 1], "tanh", "Xavier normal",
                                   input_bounds=([0.0], [1.0])).double()
        x = torch.tensor([[0.2], [0.5], [0.9]], dtype=torch.float64, requires_grad=True)
        batched = net(x)
        separate = torch.cat([net(row[None]) for row in x])
        torch.testing.assert_close(batched, separate)
        derivative = torch.autograd.grad(batched.sum(), x, create_graph=True)[0]
        second = torch.autograd.grad(derivative.sum(), x)[0]
        self.assertTrue(torch.isfinite(second).all())
        raw = dp.nn.FullyConnected([1, 8, 1], "tanh", "Xavier normal")
        self.assertTrue(torch.isfinite(raw(torch.tensor([[0.5]]))).all())
        with self.assertRaises(ValueError):
            dp.nn.FullyConnected([1, 8, 1], "tanh", "Xavier normal",
                                 input_bounds=([0.0], [0.0]))

    def test_multiple_conditions_and_vector_residual(self):
        p = dp.PINNProblem(lambda x, u: u, torch.ones(3, 1), [
            dp.PointConstraint("initial", torch.zeros(2, 1), torch.zeros(2, 2)),
            dp.PointConstraint("observations", torch.ones(4, 1), torch.ones(4, 2))])
        model = dp.PINN(p, torch.nn.Linear(1, 2), weights={"initial": 2}).compile(device="cpu")
        model.train(2, display_every=0)
        self.assertEqual(set(model.compute_losses()), {"pde", "initial", "observations"})

    def test_nonfinite_loss_does_not_update(self):
        p = problem()
        p.residual = lambda x, u: u * float("nan")
        model = dp.PINN(p, network()).compile(device="cpu")
        before = [p.detach().clone() for p in model.network.parameters()]
        with self.assertRaises(FloatingPointError):
            model.train(1, display_every=0)
        for old, new in zip(before, model.network.parameters()):
            self.assertTrue(torch.equal(old, new))
        self.assertEqual(model.iter, 0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
class CUDAPINNTests(unittest.TestCase):
    def test_gpu_poisson_and_auto_device(self):
        torch.manual_seed(7)
        net = dp.nn.FullyConnected([1, 16, 16, 1], "tanh", "Xavier normal",
                                   input_bounds=([0.0], [1.0]))
        model = dp.PINN(problem(), net).compile(lr=0.01)
        self.assertEqual(model.device.type, "cuda")
        self.assertTrue(all(p.is_cuda for p in net.parameters()))
        self.assertTrue(net.input_lower.is_cuda)
        self.assertTrue(model.collocation_point_sample.is_cuda)
        self.assertTrue(all(x.is_cuda and y.is_cuda for _, x, y in model._constraints))
        initial = sum(v.item() for v in model.compute_losses().values())
        model.train(1000, display_every=0)
        x = torch.linspace(0, 1, 101, device="cuda")[:, None]
        prediction = model.predict(x)
        self.assertTrue(prediction.is_cuda)
        self.assertLess(((prediction - x*(1-x))**2).mean().sqrt().item(), 0.02)
        self.assertLess(sum(v.item() for v in model.compute_losses().values()), initial * 0.01)
        self.assertGreater(torch.cuda.max_memory_allocated(), 0)

    def test_cpu_gpu_losses_agree(self):
        torch.manual_seed(7)
        cpu = dp.PINN(problem(), network()).compile(device="cpu")
        gpu_net = network()
        gpu_net.load_state_dict(cpu.network.state_dict())
        gpu = dp.PINN(problem(), gpu_net).compile(device="cuda")
        for name, value in cpu.compute_losses().items():
            torch.testing.assert_close(value, gpu.compute_losses()[name].cpu(),
                                       rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
