"""Independent DSGNAR arithmetic, rejected-step state and exact continuation."""
from copy import deepcopy
from dataclasses import asdict
import math
import unittest
from unittest.mock import patch
import torch
from margam.methods.dsgnar import (DSGNARPINN,cosine_transform,right_sketch,lift_step,
                                  count_add,lm_step,choose_radius,reweight,next_ratio)
from margam.components.dsgnar_config import dsgnar_defaults
from margam.components import PINNProblem
from margam.templates.examples import example_config
from margam.templates.poisson import build_pinn


def config(device='cpu',precision='float64'):
    c=example_config('dsgnar_poisson_1d');c.device=device;c.layers=(4,);c.points=7;c.boundary_points=3
    c.dsgnar_settings.update(sketch_size=8,hashes=2,batch_size=4,probes=4,precision=precision)
    return c


class DSGNARTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_fft_cosine_and_adjoint_match_dense_definition(self):
        for n in (1,5,8):
            x=torch.randn(3,n,dtype=torch.float64);k=torch.arange(n,dtype=x.dtype)[:,None];j=torch.arange(n,dtype=x.dtype)[None,:]
            dct=math.sqrt(2/n)*torch.cos(math.pi/n*(j+.5)*k);dct[0]/=math.sqrt(2)
            torch.testing.assert_close(cosine_transform(x),x@dct.T,atol=2e-14,rtol=2e-14)
            torch.testing.assert_close(cosine_transform(cosine_transform(x),True),x,atol=2e-14,rtol=2e-14)
            operators=(torch.ones(n,dtype=x.dtype),torch.arange(n-1,-1,-1),torch.arange(min(3,n)))
            y=torch.randn(min(3,n),dtype=x.dtype)
            torch.testing.assert_close(right_sketch(x,operators)@y,x@lift_step(y,operators))

    def test_count_sketch_collision_and_partition_equivalence(self):
        jac=torch.arange(15,dtype=torch.float64).reshape(5,3);values=torch.arange(5,dtype=jac.dtype)
        buckets=torch.tensor([[0,1,0,2,1],[2,1,0,0,1]]);signs=torch.tensor([[1,-1,1,1,-1],[-1,1,1,-1,1]],dtype=jac.dtype)
        projection=torch.zeros(3,5,dtype=jac.dtype)
        for h in range(2):
            for i in range(5):projection[buckets[h,i],i]+=signs[h,i]/math.sqrt(2)
        matrix=torch.zeros(3,3,dtype=jac.dtype);vector=torch.zeros(3,dtype=jac.dtype)
        count_add(matrix,vector,jac,values,buckets,signs)
        torch.testing.assert_close(matrix,projection@jac);torch.testing.assert_close(vector,projection@values)
        other=torch.zeros_like(matrix);rv=torch.zeros_like(vector)
        for a,b in ((0,3),(3,5)):count_add(other,rv,jac[a:b],values[a:b],buckets[:,a:b],signs[:,a:b])
        torch.testing.assert_close(other,matrix);torch.testing.assert_close(rv,vector)

    def test_blocked_physics_jacobian_matches_analytic_polynomial(self):
        from margam.components.differential import derivative
        class Polynomial(torch.nn.Module):
            def __init__(self):super().__init__();self.coefficients=torch.nn.Parameter(torch.tensor([.3,-.2],dtype=torch.float64))
            def forward(self,x):return self.coefficients[0]*x.square()+self.coefficients[1]*x
        points=torch.linspace(0,1,5,dtype=torch.float64)[:,None]
        problem=PINNProblem(lambda x,u:derivative(derivative(u,x,0),x,0)+u,points,[])
        settings=dsgnar_defaults();settings.update(sketch_size=2,batch_size=2,hashes=1)
        m=DSGNARPINN(problem,Polynomial(),settings=settings).compile(device='cpu')
        operators=(torch.ones(2,dtype=torch.float64),torch.arange(2),torch.arange(2))
        rng=m.generator.get_state();matrix,vector,losses=m._sketch(operators,2)
        m.generator.set_state(rng);projection=torch.zeros(2,5,dtype=torch.float64)
        for start in range(0,5,2):
            count=min(2,5-start);buckets=torch.randint(2,(1,count),generator=m.generator)
            signs=torch.randint(2,(1,count),generator=m.generator)*2-1
            for j in range(count):projection[buckets[0,j],start+j]+=signs[0,j]
        jac=torch.cat((2+points.square(),points),1);residual=jac@m.network.coefficients.detach()
        torch.testing.assert_close(matrix,projection@right_sketch(jac,operators)/math.sqrt(5))
        torch.testing.assert_close(vector,projection@residual/math.sqrt(5))
        self.assertAlmostEqual(losses['pde'],residual.square().mean().item())

    def test_svd_secular_solve_rank_deficiency_and_radius(self):
        s=torch.tensor([3.,1.,0.],dtype=torch.float64);r=torch.tensor([1.,2.,4.],dtype=s.dtype);v=torch.eye(3,dtype=s.dtype)
        step,lam=lm_step(s,r,v,.2)
        self.assertGreater(lam,0);self.assertAlmostEqual(torch.linalg.vector_norm(step).item(),.2,places=13)
        torch.testing.assert_close((torch.diag(s.square())+lam*v)@step,-s*r)
        step,lam=lm_step(s,r,v,100.);self.assertEqual(lam,0);torch.testing.assert_close(step,torch.tensor([-1/3,-2.,0.],dtype=s.dtype))
        step,lam=lm_step(s*0,r,v,1.);self.assertEqual(lam,0);self.assertTrue(torch.equal(step,step*0))

    def test_monotone_probes_weights_and_stage_switch(self):
        self.assertEqual(choose_radius([1.,2.,4.],[.9,.8,.7],.5),4.)
        self.assertEqual(choose_radius([1.,2.,4.],[-1.,.8,.7],.5),1.)
        chosen=choose_radius([1.,2.,4.,8.],[.9,.5,.5,.1],.3);self.assertTrue(4<chosen<8)
        self.assertTrue(2<choose_radius([1.,2.,4.,8.],[.9,.3,-1.,-2.],.075)<4)
        self.assertTrue(1<choose_radius([1.,2.,4.,8.],[.9,.5,.5,.1],.7)<2)
        weights=reweight(dict(a=1.,b=1.),dict(a=4.,b=1.),.5)
        self.assertAlmostEqual(sum(weights.values()),2);self.assertLess(weights['a'],weights['b'])
        self.assertEqual(reweight(dict(a=2.,b=1.),dict(a=0.,b=0.),.05),dict(a=2.,b=1.))
        s=dsgnar_defaults();s['history_window']=3
        self.assertEqual(next_ratio([.1,1.,10.],s,.075),.5)
        self.assertEqual(next_ratio([10.,1.,.1],s,.075),.075)
        s['stage_grace']=5
        self.assertEqual(next_ratio([.1,1.,10.],s,.075),.075)
        self.assertEqual(next_ratio([.001,.01,.1,1.,10.,100.],s,.075),.5)

    def test_linear_problem_acceptance_and_no_adam_updates(self):
        net=torch.nn.Linear(1,1,bias=False).double();net.weight.data.zero_()
        p=PINNProblem(lambda x,u:u,torch.ones(1,1),[],residual_target=1.)
        settings=dsgnar_defaults();settings.update(sketch_size=1,hashes=1,initial_radius=.1,reweight_exponent=0.)
        m=DSGNARPINN(p,net,settings=settings).compile(device='cpu')
        m.train(2,0);self.assertTrue(all(row['accepted'] for row in m.diagnostics))
        self.assertTrue(all(row['objective_after']<row['objective_before'] for row in m.diagnostics))
        self.assertEqual(m.optimiser.state_dict()['state'],{})

    def _resume(self,device,precision):
        c=config(device,precision);full=build_pinn(c);full.train(4,0)
        split=build_pinn(c);split.train(2,0);snapshot=split.training_state()
        restored=build_pinn(c);restored.load_training_state(snapshot);restored.train(2,0)
        self.assertEqual(full.loss_history,restored.loss_history);self.assertEqual(full.diagnostics,restored.diagnostics)
        for k,v in full.network.state_dict().items():torch.testing.assert_close(v,restored.network.state_dict()[k],rtol=0,atol=0)
        self.assertEqual(len(full.collocation_point_sample),9)
        self.assertEqual(full.dtype,getattr(torch,precision));self.assertEqual(torch.get_default_dtype(),torch.float32)

    def test_cpu_exact_resume_both_precisions_and_remainder_rows(self):
        for precision in ('float32','float64'):self._resume('cpu',precision)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_exact_resume_both_precisions(self):
        for precision in ('float32','float64'):self._resume('cuda',precision)

    def test_double_precision_laplacian_and_resume_ignore_global_default(self):
        from margam.utils import laplacian
        x=torch.tensor([[.123456789123],[.987654321987]],dtype=torch.float64,requires_grad=True)
        value=laplacian(x.pow(4),x)
        self.assertEqual(value.dtype,torch.float64)
        torch.testing.assert_close(value,12*x.square(),rtol=1e-15,atol=1e-15)
        c=config();full=build_pinn(c);full.train(4,0);split=build_pinn(c);split.train(2,0)
        previous=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            restored=build_pinn(c);restored.load_training_state(split.training_state());restored.train(2,0)
        finally:torch.set_default_dtype(previous)
        self.assertEqual(full.loss_history,restored.loss_history);self.assertEqual(full.diagnostics,restored.diagnostics)
        for k,v in full.network.state_dict().items():torch.testing.assert_close(v,restored.network.state_dict()[k],rtol=0,atol=0)

    def test_reject_rollback_and_early_stop(self):
        m=build_pinn(config());before=deepcopy(m.network.state_dict());m.settings['min_radius']=.5
        with patch.object(m,'_losses',return_value={name:1e30 for name in m.names}):m.train(3,0)
        self.assertEqual(m.iter,1);self.assertTrue(m.converged);self.assertFalse(m.diagnostics[0]['accepted'])
        for k,v in before.items():torch.testing.assert_close(v,m.network.state_dict()[k],rtol=0,atol=0)
        m=build_pinn(config());before=deepcopy(m.network.state_dict());rng=m.generator.get_state()
        with patch.object(m,'_losses',side_effect=MemoryError('injected')),self.assertRaises(MemoryError):m.train(1,0)
        self.assertEqual(m.iter,0);self.assertTrue(torch.equal(rng,m.generator.get_state()))
        for k,v in before.items():torch.testing.assert_close(v,m.network.state_dict()[k],rtol=0,atol=0)

    def test_converged_checkpoint_child_emits_no_empty_progress(self):
        from margam.execution.codec import encode_config
        from margam.execution.transport import JobTransport
        c=config();c.dsgnar_settings['min_radius']=.5
        model=build_pinn(c)
        with patch.object(model,'_losses',return_value={name:1e30 for name in model.names}):model.train(1,0)
        self.assertTrue(model.converged)
        events=[]
        packet=JobTransport().run(dict(operation='train',config=encode_config(c),checkpoint=model.training_state(),requests=[],metadata={},dtype='float64'),events.append)
        self.assertEqual(packet['result']['steps'],1)
        self.assertEqual([event['kind'] for event in events],['ready'])

    def test_recorded_objective_uses_same_evaluation_batches_as_probes(self):
        c=example_config('dsgnar_burgers');c.device='cpu';c.points=129;c.boundary_points=7
        c.dsgnar_settings.update(precision='float32',sketch_size=8,batch_size=4,probes=3)
        model=build_pinn(c);physical=model._losses()
        expected=sum(model.weights[name]*value for name,value in physical.items())
        model.train(1,0)
        self.assertEqual(model.diagnostics[0]['objective_before'],expected)
        self.assertEqual({key:model.loss_history[0][key] for key in physical},physical)

    def test_checkpoint_rejects_changed_physics_history_weights_rng(self):
        c=config();m=build_pinn(c);m.train(2,0);snapshot=m.training_state()
        for mutate in (lambda s:s['dsgnar']['weights'].update(pde=999.),lambda s:s['dsgnar'].update(radius=99.),
                       lambda s:s['dsgnar'].update(generator=torch.zeros(1,dtype=torch.uint8)),lambda s:s['loss_history'][0].update(total=999.)):
            state=deepcopy(snapshot);mutate(state)
            with self.assertRaises(ValueError):build_pinn(c).load_training_state(state)
        legacy=deepcopy(snapshot);legacy['dsgnar']['settings'].pop('stage_grace');legacy['dsgnar']['configuration']['dsgnar_settings'].pop('stage_grace')
        build_pinn(c).load_training_state(legacy)
        changed=deepcopy(c);changed.source=3.
        with self.assertRaises(ValueError):build_pinn(changed).load_training_state(snapshot)

    def test_unsupported_compositions_are_explicit(self):
        for attribute,value in [('loss_formulation','integral'),('physics','incompressible_2d'),('shape','Prepared2D')]:
            c=config();c.pde_preset='';setattr(c,attribute,value)
            with self.assertRaises(ValueError):c.validate()
