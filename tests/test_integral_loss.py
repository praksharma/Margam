"""Integral-of-squared-error measures, gradients, components and continuation."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import torch
from margam.components.problem import PINNProblem,PointConstraint
from margam.methods.pinn import PINN
from margam.templates.poisson import PoissonConfig,build_pinn
from margam.components.adaptive_sampling import sampling_defaults


class ConstantNetwork(torch.nn.Module):
    def __init__(self,components=1):
        super().__init__();self.value=torch.nn.Parameter(torch.tensor(2.,dtype=torch.float64));self.components=components
    def forward(self,x):return self.value.expand(len(x),self.components)


class IntegralLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_measure_not_sample_count_and_component_sum(self):
        for rows in (3,19):
            for components in (1,3):
                problem=PINNProblem(lambda x,u:u,torch.ones(rows,2),[PointConstraint('wall',torch.zeros(7,2),0.,measure=3.)],measure=6.)
                integral=PINN(problem,ConstantNetwork(components),loss_formulation='integral').compile(device='cpu')
                losses=integral.compute_losses()
                self.assertAlmostEqual(losses['pde'].item(),6*4*components)
                self.assertAlmostEqual(losses['wall'].item(),3*4*components)
                self.assertAlmostEqual(torch.autograd.grad(losses['pde'],integral.network.value)[0].item(),6*4*components)
                mean=PINN(problem,ConstantNetwork(components)).compile(device='cpu')
                self.assertEqual(mean.compute_losses()['pde'].item(),4.)

    def test_square_before_integrating_prevents_residual_cancellation(self):
        problem=PINNProblem(lambda x,u:x*u,torch.tensor([[-1.],[1.]]),measure=2.)
        model=PINN(problem,ConstantNetwork(),loss_formulation='integral').compile(device='cpu')
        self.assertEqual(model.compute_losses()['pde'].item(),8.)
        self.assertEqual((model.network(model.collocation_point_sample)*model.collocation_point_sample).mean().square().item(),0.)

    def test_template_side_measures_and_point_gauges(self):
        c=PoissonConfig(x_min=-1.,x_max=2.,y_min=-2.,y_max=0.,points=9,layers=(8,),loss_formulation='integral')
        m=build_pinn(c)
        self.assertEqual(m.loss_measures,dict(pde=6.,left=2.,right=2.,bottom=3.,top=3.))
        c.shape='Interval';m=build_pinn(c)
        self.assertEqual(m.loss_measures,dict(pde=3.,left=1.,right=1.))
        c=PoissonConfig(x_max=3.,y_max=2.,physics='incompressible_2d',pressure_point=(0.,0.),points=9,layers=(8,),loss_formulation='integral')
        m=build_pinn(c)
        self.assertEqual(m.loss_measures['left:u'],2.)
        self.assertEqual(m.loss_measures['top:v'],3.)
        self.assertEqual(m.loss_measures['pressure_gauge'],1.)
        m.train(3,display_every=0)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in m.loss_history[-1].values()))

    def test_importance_correction_multiplies_volume_once(self):
        settings=sampling_defaults('importance');settings.update(interval=2,candidates=32,chunk=8)
        c=PoissonConfig(x_max=3.,y_max=2.,points=9,layers=(8,),loss_formulation='integral',adaptive_sampling=settings)
        m=build_pinn(c);m.train(3,display_every=0)
        values,target,rows=m.loss_samples()['pde'];values,target=m._aligned(values,target,rows)
        expected=(6.*m.sampling_controller.weights*(values-target).square().sum(1,keepdim=True)).mean()
        torch.testing.assert_close(m.compute_losses()['pde'],expected)
        # Exact expectation of inverse-probability quadrature for non-unit volume.
        probabilities=torch.tensor([.1,.2,.7],dtype=torch.float64)
        errors=torch.tensor([1.,4.,9.],dtype=torch.float64)
        torch.testing.assert_close((probabilities*6*errors/(3*probabilities)).sum(),6*errors.mean())

    def test_all_methods_resume_integral_losses_and_reject_changed_mode(self):
        for method in ('baseline','hard_dirichlet','self_adaptive','lra'):
            c=PoissonConfig(x_max=3.,y_max=2.,points=9,layers=(8,),method=method,loss_formulation='integral')
            if method=='self_adaptive':c.sa_boundary=True
            continuous=build_pinn(c);continuous.train(6,display_every=0)
            partial=build_pinn(c);partial.train(3,display_every=0);state=partial.training_state()
            resumed=build_pinn(c);resumed.load_training_state(state);resumed.train(3,display_every=0)
            self.assertEqual(resumed.loss_history,continuous.loss_history)
            mean=deepcopy(c);mean.loss_formulation='mean'
            with self.assertRaisesRegex(ValueError,'loss formulation'):build_pinn(mean).load_training_state(state)
            bad=deepcopy(state);bad['loss_formulation']['measures']['pde']=1.
            with self.assertRaisesRegex(ValueError,'loss formulation'):build_pinn(c).load_training_state(bad)
            if method=='self_adaptive':
                errors=resumed.point_errors()
                expected=sum((error*resumed.adaptive_weights[name].square()).mean()*resumed.loss_scale(name)*resumed.weights.get(name,1.) for name,error in errors.items())
                torch.testing.assert_close(resumed.objective(errors),expected)

    def test_archive_roundtrip_and_legacy_default(self):
        from margam.ui.run_document import save_run,load_run
        from margam.ui.model_document import decode_document
        for case in ('baseline','self_adaptive','lra','hard_dirichlet','importance'):
            method='baseline' if case=='importance' else case
            c=PoissonConfig(shape='Interval',x_max=3.,points=8,layers=(8,),loss_formulation='integral',method=method)
            if case=='importance':
                c.adaptive_sampling=sampling_defaults('importance');c.adaptive_sampling.update(interval=1,candidates=32,chunk=8)
            m=build_pinn(c);m.train(3,display_every=0)
            data=dict(format='margam-run',version=2,model=dict(format='margam-model',version=14,config=asdict(c),nodes=[],wires=[]),result=dict(outputs={},steps=3,cancelled=False,device='cpu'),monitors={},checkpoint=m.training_state())
            with TemporaryDirectory() as directory:
                p=Path(directory)/'integral.pt';save_run(p,data);loaded=load_run(p)
                n=build_pinn(c);n.load_training_state(loaded['checkpoint']);n.train(2,display_every=0);m.train(2,display_every=0)
                self.assertEqual(n.loss_history,m.loss_history)
        old=asdict(PoissonConfig());old.pop('loss_formulation')
        restored,_,_=decode_document(dict(format='margam-model',version=13,config=old,nodes=[],wires=[]))
        self.assertEqual(restored.loss_formulation,'mean')

    def test_invalid_or_unavailable_quadrature_is_explicitly_rejected(self):
        c=PoissonConfig(loss_formulation='integral',adaptive_sampling=sampling_defaults('rar'))
        with self.assertRaisesRegex(ValueError,'RAR anchors'):build_pinn(c)
        c.adaptive_sampling={};c.interior_data=dict(path='missing.csv',coordinates='x,y',normals='',values='')
        with self.assertRaisesRegex(ValueError,'quadrature'):build_pinn(c)
        c.interior_data={};c.boundary_data={'left':dict(path='missing.csv',coordinates='x,y',normals='',values='')}
        with self.assertRaisesRegex(ValueError,'quadrature'):build_pinn(c)
        for measure in (0.,-1.,float('nan')):
            p=PINNProblem(lambda x,u:u,torch.ones(3,1),measure=measure)
            with self.assertRaisesRegex(ValueError,'measures'):PINN(p,ConstantNetwork(),loss_formulation='integral').compile()
