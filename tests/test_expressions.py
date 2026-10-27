"""Restricted syntax, autograd derivatives and manufactured non-Poisson physics."""
import unittest
import torch
from margam.components.expressions import Expression
from margam.templates.poisson import PoissonConfig, build_pinn, make_poisson_reference


class ExpressionTests(unittest.TestCase):
    def test_derivatives_and_normal(self):
        points=torch.tensor([[.2,.3],[.7,.4]],requires_grad=True)
        u=points[:,0:1]**3+points[:,1:2]**2
        torch.testing.assert_close(Expression('laplace(u)',2)(points,u),6*points[:,0:1]+2)
        torch.testing.assert_close(Expression('D(D(u,x),x) + D(u,y)',2)(points,u),6*points[:,0:1]+2*points[:,1:2])
        torch.testing.assert_close(Expression('dn(u)',2)(points,u,normal=(-1.,0.)),-3*points[:,0:1]**2)
        torch.testing.assert_close(Expression('laplace(u)',2)(points,points[:,0:1]),torch.zeros(2,1))
        torch.testing.assert_close(Expression('sin(pi*x)',2,target=True)(points),torch.sin(torch.pi*points[:,0:1]))

    def test_restricted_syntax(self):
        for text in ('__import__("os")','u.detach()','x[0]','[x]','lambda: x','u**100','D(u,z)','y','sqrt(u)','1e999'):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):Expression(text,1,target=text=='sqrt(u)')
        with self.assertRaises(ValueError):Expression('1/0',1)(torch.ones(2,1))

    def test_custom_pde_robin_and_spatial_targets(self):
        config=PoissonConfig(shape='Interval',pde_expression='D(D(u,x),x) + u - x*x - 2')
        config.bc_types['right']='Custom';config.bc_operators['right']='u + dn(u)'
        config.bc_targets['right']='3*x'
        model=build_pinn(config)
        x=torch.tensor([[.2],[.7]],requires_grad=True)
        torch.testing.assert_close(model.problem.residual(x,x*x),torch.zeros_like(x))
        bc=model.problem.constraints[1]
        x=torch.tensor([[1.]],requires_grad=True)
        torch.testing.assert_close(bc.operator(x,x*x),torch.tensor([[3.]]))
        torch.testing.assert_close(torch.as_tensor(bc.target(x)),torch.tensor([[3.]]))
        with self.assertRaisesRegex(ValueError,'Preset references'):make_poisson_reference(config)
        torch.set_num_threads(1);model.train(500,display_every=0)
        x=torch.linspace(0,1,21)[:,None]
        self.assertLess((model.predict(x)-x*x).square().mean().sqrt().item(),.05)

    def test_variable_corner_consistency(self):
        config=PoissonConfig(bc_targets={'bottom':'x*(1-x)'})
        config.validate()
        config.bc_targets['bottom']='x'
        with self.assertRaisesRegex(ValueError,'corner'):config.validate()
