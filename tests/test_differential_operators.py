import unittest
import torch
from torch import nn

from margam.templates.poisson import PoissonConfig, build_pinn
from margam.utils.differentialoperators import (
    convective,
    div,
    grad,
    jac,
    laplacian,
    matrix_div,
    normal_derivative,
    partial,
    rot,
    sym_grad,
)


class AffineLinearNet(nn.Module):
    def __init__(self, in_features=2, out_features=1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class DifferentialOperatorsTests(unittest.TestCase):
    def test_partial_linear_and_exact_polynomial_derivatives(self):
        x = torch.tensor([[0.5], [1.25], [-2.0]], dtype=torch.float64, requires_grad=True)
        u_linear = 3.0 * x
        du_dx = partial(u_linear, x)
        self.assertEqual(du_dx.dtype, torch.float64)
        self.assertTrue(torch.allclose(du_dx, torch.full_like(x, 3.0), atol=1e-12))

        u_quad = x ** 2 + 4.0 * x - 1.0
        d2u_dx2 = partial(u_quad, x, x)
        self.assertEqual(d2u_dx2.dtype, torch.float64)
        self.assertTrue(torch.allclose(d2u_dx2, torch.full_like(x, 2.0), atol=1e-12))

        d3u_dx3 = partial(u_quad, x, x, x)
        self.assertEqual(d3u_dx3.shape, (3, 1))
        self.assertEqual(d3u_dx3.dtype, torch.float64)
        self.assertTrue(torch.allclose(d3u_dx3, torch.zeros_like(x), atol=1e-12))

    def test_partial_mixed_and_parameterized_linear_second_derivative(self):
        x = torch.tensor([[0.5], [1.5]], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[2.0], [-1.0]], dtype=torch.float64, requires_grad=True)
        u = (x ** 2) * (y ** 3)
        mixed = partial(u, x, y)
        expected = 6.0 * x * (y ** 2)
        self.assertTrue(torch.allclose(mixed, expected, atol=1e-12))

        net = AffineLinearNet(1, 1).double()
        out = net(x)
        second = partial(out, x, x)
        self.assertEqual(second.shape, (2, 1))
        self.assertEqual(second.dtype, torch.float64)
        self.assertTrue(torch.allclose(second, torch.zeros_like(x), atol=1e-12))

    def test_laplacian_analytical_and_parameterized_linear_or_relu(self):
        xy = torch.tensor([[0.2, 0.7], [-0.4, 1.1]], dtype=torch.float64, requires_grad=True)
        x, y = xy[:, 0:1], xy[:, 1:2]
        # u(x, y) = x^3 + 2 x y^2 => u_xx + u_yy = 6 x + 4 x = 10 x
        u = x ** 3 + 2.0 * x * (y ** 2)
        lap = laplacian(u, xy)
        self.assertEqual(lap.dtype, torch.float64)
        self.assertTrue(torch.allclose(lap, 10.0 * x, atol=1e-12))

        # Parameterized affine network has grad_fn wrt weights but zero 2nd derivative wrt xy
        net = AffineLinearNet(2, 1).double()
        lap_linear = laplacian(net(xy), xy)
        self.assertEqual(lap_linear.shape, (2, 1))
        self.assertEqual(lap_linear.dtype, torch.float64)
        self.assertTrue(torch.allclose(lap_linear, torch.zeros((2, 1), dtype=torch.float64), atol=1e-12))

        # Partially linear in y: u = x^2 + w * y
        w = nn.Parameter(torch.tensor(2.5, dtype=torch.float64))
        u_semi = xy[:, 0:1] ** 2 + w * xy[:, 1:2]
        lap_semi = laplacian(u_semi, xy)
        self.assertTrue(torch.allclose(lap_semi, torch.full((2, 1), 2.0, dtype=torch.float64), atol=1e-12))

    def test_poisson_neumann_and_laplacian_with_constant_or_relu_network(self):
        cfg = PoissonConfig(
            shape='Rectangle',
            points=8,
            boundary_points=4,
            layers=(4,),
            bc_types={'left': 'Dirichlet', 'right': 'Neumann', 'bottom': 'Dirichlet', 'top': 'Neumann'},
        )
        model = build_pinn(cfg)
        bias = nn.Parameter(torch.tensor([[1.0]]))

        class ConstantBiasNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = bias

            def forward(self, x):
                return self.bias.expand(len(x), 1)

        model.network = ConstantBiasNet()
        model.compile(lr=cfg.lr, device='cpu')
        losses = model.compute_losses()
        self.assertTrue(all(torch.isfinite(v).item() for v in losses.values()))

    def test_div_and_matrix_div_preserve_float64_and_handle_empty_axis(self):
        xy = torch.tensor([[0.1, 0.2], [0.3, -0.4]], dtype=torch.float64, requires_grad=True)
        x, y = xy[:, 0:1], xy[:, 1:2]
        # F(x, y) = (x^3 + y, 2 y^3 - x) => div F = 3 x^2 + 6 y^2
        vec = torch.cat([x ** 3 + y, 2.0 * (y ** 3) - x], dim=1)
        d = div(vec, xy)
        expected = 3.0 * (x ** 2) + 6.0 * (y ** 2)
        self.assertEqual(d.dtype, torch.float64)
        self.assertTrue(torch.allclose(d, expected, atol=1e-14, rtol=0.0))

        # Empty derivative slice must not raise UnboundLocalError
        empty = torch.zeros((2, 0), dtype=torch.float64, requires_grad=True)
        d_with_empty = div(vec, empty, xy)
        self.assertEqual(d_with_empty.dtype, torch.float64)
        self.assertTrue(torch.allclose(d_with_empty, expected, atol=1e-14, rtol=0.0))

        # Matrix field M(x, y) = [[x^2, x*y], [x*y^2, x^3*y]]
        # row 0 div: d(x^2)/dx + d(x*y)/dy = 2x + x = 3x
        # row 1 div: d(x*y^2)/dx + d(x^3*y)/dy = y^2 + x^3
        row0 = torch.cat([x ** 2, x * y], dim=1)
        row1 = torch.cat([x * (y ** 2), (x ** 3) * y], dim=1)
        mat = torch.stack([row0, row1], dim=1)
        mdiv = matrix_div(mat, xy)
        expected_mdiv = torch.cat([3.0 * x, (y ** 2) + (x ** 3)], dim=1)
        self.assertEqual(mdiv.dtype, torch.float64)
        self.assertTrue(torch.allclose(mdiv, expected_mdiv, atol=1e-14, rtol=0.0))

    def test_rot_preserves_dtype_and_device_with_analytical_curl(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for dev in devices:
            xyz = torch.tensor(
                [[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]],
                dtype=torch.float64,
                device=dev,
                requires_grad=True,
            )
            x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
            # F = (y*z, x*z^2, x^2*y)
            # curl F = (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy)
            #        = (x^2 - 2*x*z, y - 2*x*y, z^2 - z)
            field = torch.cat([y * z, x * (z ** 2), (x ** 2) * y], dim=1)
            curl = rot(field, xyz)
            expected = torch.cat([x ** 2 - 2.0 * x * z, y - 2.0 * x * y, z ** 2 - z], dim=1)
            self.assertEqual(curl.device.type, dev)
            self.assertEqual(curl.dtype, torch.float64)
            self.assertTrue(torch.allclose(curl, expected, atol=1e-14, rtol=0.0))

    def test_grad_normal_jac_convective_and_sym_grad_analytical(self):
        xy = torch.tensor([[0.3, -0.5], [1.2, 0.4]], dtype=torch.float64, requires_grad=True)
        x, y = xy[:, 0:1], xy[:, 1:2]
        u = (x ** 2) * y + 3.0 * y
        g = grad(u, xy)
        expected_g = torch.cat([2.0 * x * y, x ** 2 + 3.0], dim=1)
        self.assertEqual(g.dtype, torch.float64)
        self.assertTrue(torch.allclose(g, expected_g, atol=1e-14))

        normals = torch.tensor([[0.6, 0.8], [1.0, 0.0]], dtype=torch.float64)
        nd = normal_derivative(u, normals, xy)
        self.assertTrue(torch.allclose(nd, (expected_g * normals).sum(dim=1, keepdim=True), atol=1e-14))

        vec = torch.cat([x ** 2 + 2.0 * y, 3.0 * x * y], dim=1)
        j = jac(vec, xy)
        expected_j = torch.stack(
            [torch.cat([2.0 * x, torch.full_like(x, 2.0)], dim=1), torch.cat([3.0 * y, 3.0 * x], dim=1)],
            dim=1,
        )
        self.assertEqual(j.dtype, torch.float64)
        self.assertTrue(torch.allclose(j, expected_j, atol=1e-14))

        vel = torch.tensor([[1.5, -0.5], [-2.0, 0.25]], dtype=torch.float64)
        conv = convective(vec, vel, xy)
        expected_conv = torch.bmm(expected_j, vel.unsqueeze(2)).squeeze(2)
        self.assertTrue(torch.allclose(conv, expected_conv, atol=1e-14))

        sg = sym_grad(vec, xy)
        expected_sg = 0.5 * (expected_j + expected_j.transpose(1, 2))
        self.assertTrue(torch.allclose(sg, expected_sg, atol=1e-14))
        self.assertIn(r'\nabla', sym_grad.__doc__)


if __name__ == '__main__':
    unittest.main()
