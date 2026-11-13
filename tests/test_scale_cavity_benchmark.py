"""Paper cavity jets and objective match native Scale-PINN without upstream code."""
from copy import deepcopy
import unittest
import torch
from margam.research.scale_cavity import ScaleCavityNetwork,cavity_terms
from margam.components.differential import derivative
from margam.components.problem import PINNProblem,PointConstraint
from margam.methods.scale import ScalePINN
from margam.templates.incompressible_flow import flow_demo_config
from margam.components.expressions import Expression
from margam.templates.poisson import FLOW_RESIDUALS
from examples.scale_cavity_benchmark import protocol,learning_rate


class ScaleCavityBenchmarkTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_paper_parameter_counts(self):
        for width,count in ((32,15424),(64,59520)):
            self.assertEqual(sum(p.numel() for p in ScaleCavityNetwork(width).parameters()),count)

    def test_pure_coordinate_derivatives_match_autograd(self):
        with torch.random.fork_rng():
            torch.manual_seed(7);network=ScaleCavityNetwork(4).double()
        points=torch.tensor([[.13,.27],[.47,.68],[.8,.2]],dtype=torch.float64,requires_grad=True)
        values=network(points);jet=network.jets(points)
        torch.testing.assert_close(jet[0],values,rtol=1e-12,atol=1e-12)
        for field in range(3):
            for axis in range(2):
                first=derivative(values[:,field:field+1],points,axis);second=derivative(first,points,axis)
                torch.testing.assert_close(jet[1+axis,:,field],first[:,0],rtol=1e-10,atol=1e-11)
                torch.testing.assert_close(jet[3+axis,:,field],second[:,0],rtol=1e-10,atol=1e-10)

    def test_objective_and_parameter_gradients_match_native_scale(self):
        with torch.random.fork_rng():
            torch.manual_seed(9);network=ScaleCavityNetwork(4).double();previous=deepcopy(network).requires_grad_(False)
            with torch.no_grad():
                for p in previous.parameters():p.add_(.001*torch.randn_like(p))
        points=torch.tensor([[.2,.3],[.6,.7],[.2,1.],[0.,.4]],dtype=torch.float64)
        wall=torch.tensor([False,False,True,True]);targets=torch.tensor([[0.,0.],[0.,0.],[1.,0.],[0.,0.]],dtype=torch.float64)
        c=flow_demo_config();c.method='scale';c.scale_settings.update(fresh=False,tau_sc=.06,tau_alpha=1.,gamma=1/3200);c.viscosity=1/3200
        expressions=[Expression(text,2,fields=('u','v','p')) for text in FLOW_RESIDUALS]
        def residual(x,u):return torch.cat([expr(x,u,viscosity=c.viscosity,source=0.) for expr in expressions],1)
        problem=PINNProblem(residual,points[~wall],[PointConstraint('u_bc',points[wall],targets[wall,0:1],lambda x,u:u[:,0:1]),PointConstraint('v_bc',points[wall],targets[wall,1:2],lambda x,u:u[:,1:2])])
        native=ScalePINN(problem,deepcopy(network),config=c);native.weights={'pde':3.,'u_bc':15.,'v_bc':15.};native.compile(device='cpu')
        native.previous.load_state_dict(previous.state_dict())
        terms=native.compute_losses();expected=3*terms['pde']+15*(terms['u_bc']+terms['v_bc'])
        actual=cavity_terms(network,previous,points,wall,targets,reynolds=3200,tau_sc=.06,tau_alpha=1.)
        loss=actual[0]+15*(actual[1]+actual[2])
        torch.testing.assert_close(actual[0],3*terms['pde'],rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(actual[3],3*torch.tensor(native._latest['physical_pde'],dtype=torch.float64),rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(loss,expected,rtol=1e-10,atol=1e-10)
        gradients=torch.autograd.grad(loss,tuple(network.parameters()));gold=torch.autograd.grad(expected,tuple(native.network.parameters()))
        for got,want in zip(gradients,gold):torch.testing.assert_close(got,want,rtol=1e-9,atol=1e-9)
        self.assertTrue(all(p.grad is None for p in previous.parameters()))
        changed=targets.clone();changed[~wall]=12345
        unchanged=cavity_terms(network,previous,points,wall,changed,reynolds=3200,tau_sc=.06,tau_alpha=1.)
        torch.testing.assert_close(actual[1],unchanged[1],rtol=0,atol=0);torch.testing.assert_close(actual[2],unchanged[2],rtol=0,atol=0)

    def test_cli_cpu_checkpoint_continuation_and_reference_identity(self):
        import os,subprocess,sys
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import numpy as np
        with TemporaryDirectory() as folder:
            root=Path(folder);points=np.array([(x,y) for x in np.linspace(0,1,15) for y in np.linspace(0,1,15)])
            values=np.zeros((len(points),3));values[:,0]=points[:,0]*(1-points[:,0])*points[:,1]*(1-points[:,1]);values[points[:,1]==1,0]=1
            reference=root/'LDC_RE3200_100x100_subsample200x200.csv'
            np.savetxt(reference,np.column_stack((points,values)),delimiter=',',header='x,y,u,v,p',comments='')
            base=[sys.executable,'-m','examples.scale_cavity_benchmark','--reference-directory',str(root),'--device','cpu','--eager','--report-every','1','--batch-size','64','--boundary-draws','8','--retain-best']
            environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
            def run(extra):
                result=subprocess.run(base+extra,cwd=Path(__file__).resolve().parents[1],env=environment,capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stderr)
            run(['--updates','2','--output',str(root/'split')])
            run(['--updates','4','--output',str(root/'split'),'--resume',str(root/'split/checkpoint.pt')])
            run(['--updates','4','--output',str(root/'full')])
            a=torch.load(root/'split/checkpoint.pt',weights_only=True);b=torch.load(root/'full/checkpoint.pt',weights_only=True)
            self.assertEqual(a['step'],4);self.assertEqual(a['history'],b['history'])
            self.assertTrue(torch.equal(a['sampler_rng'],b['sampler_rng']))
            for name,value in a['network'].items():torch.testing.assert_close(value,b['network'][name],rtol=0,atol=0)
            for name,value in a['previous'].items():torch.testing.assert_close(value,b['previous'][name],rtol=0,atol=0)
            # A warm restart is explicitly different from continuation: reset
            # optimizer/sampler, retain source provenance and count prior updates.
            warm=['--learning-rate','.0001','--warm-start',str(root/'full/checkpoint.pt')]
            run(warm+['--updates','2','--output',str(root/'warm-split')])
            run(['--learning-rate','.0001','--updates','4','--output',str(root/'warm-split'),'--resume',str(root/'warm-split/checkpoint.pt')])
            run(warm+['--updates','4','--output',str(root/'warm-full')])
            a=torch.load(root/'warm-split/checkpoint.pt',weights_only=True);b=torch.load(root/'warm-full/checkpoint.pt',weights_only=True)
            self.assertEqual(a['signature']['initialization']['kind'],'warm_start_reset_adam')
            self.assertEqual(a['signature']['initialization']['source_updates'],4)
            self.assertEqual(a['history'],b['history'])
            from examples.plot_scale_cavity import inspect_run
            best_model,_,best_report,_,_=inspect_run(root/'warm-full',root,best=True)
            self.assertEqual(best_report['selection']['kind'],'minimum_reported_reference_error')
            self.assertEqual(best_report['total_updates'],4+best_report['steps'])
            self.assertLessEqual(best_report['steps'],4)
            self.assertEqual(next(best_model.parameters()).dtype,torch.float32)
            for name,value in a['network'].items():torch.testing.assert_close(value,b['network'][name],rtol=0,atol=0)
            changed=values.copy();inside=((points>0)&(points<1)).all(1);changed[inside,0]+=1
            np.savetxt(reference,np.column_stack((points,changed)),delimiter=',',header='x,y,u,v,p',comments='')
            rejected=subprocess.run(base+['--updates','5','--output',str(root/'split'),'--resume',str(root/'split/checkpoint.pt')],cwd=Path(__file__).resolve().parents[1],env=environment,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(rejected.returncode,0);self.assertIn('protocol/data/backend differs',rejected.stderr)
            self.assertEqual(torch.load(root/'split/checkpoint.pt',weights_only=True)['step'],4)
            from examples.scale_cavity_benchmark import load_reference
            np.savetxt(reference,np.column_stack((points,values+1)),delimiter=',',header='x,y,u,v,p',comments='')
            with self.assertRaisesRegex(ValueError,'boundary labels'):load_reference(root,3200)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_compiled_loss_and_parameter_gradients_match_eager_jets(self):
        import os
        os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','1')
        with torch.random.fork_rng():
            torch.manual_seed(7);network=ScaleCavityNetwork(4).double().cuda()
        previous=deepcopy(network).requires_grad_(False)
        points=torch.tensor([[.2,.3],[.6,.7],[.2,1.],[0.,.4]],dtype=torch.float64,device='cuda')
        wall=torch.tensor([False,False,True,True],device='cuda');targets=torch.tensor([[0.,0.],[0.,0.],[1.,0.],[0.,0.]],dtype=torch.float64,device='cuda')
        def objective(x,b,y):
            terms=cavity_terms(network,previous,x,b,y,reynolds=7500,tau_sc=.095,tau_alpha=.5)
            return terms[0]+10*(terms[1]+terms[2])
        expected=objective(points,wall,targets);gold=torch.autograd.grad(expected,tuple(network.parameters()))
        actual=torch.compile(objective,fullgraph=True)(points,wall,targets)
        gradients=torch.autograd.grad(actual,tuple(network.parameters()))
        torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-9)
        for got,want in zip(gradients,gold):torch.testing.assert_close(got,want,rtol=1e-8,atol=1e-8)

    def test_notebook_schedules_and_initial_step(self):
        for re,exponent in ((3200,.8),(7500,1.2)):
            p=protocol(re);self.assertEqual(p['cosine_exponent'],exponent)
            self.assertEqual(learning_rate(0,p),p['lr']);self.assertAlmostEqual(learning_rate(50000,p),1e-10)
            rates=[learning_rate(i,p) for i in (0,1,25000,49999,50000)]
            self.assertEqual(rates,sorted(rates,reverse=True))
        with self.assertRaises(ValueError):protocol(20000)


if __name__=='__main__':unittest.main()
