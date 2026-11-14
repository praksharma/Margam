"""Independent quadrature, original-author values and explicit output-only wiring."""
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib,json,os,subprocess,sys,unittest
import numpy as np
import torch
from margam.components.burgers_reference import burgers_reference,VISCOSITY


class BurgersReferenceTests(unittest.TestCase):
    def test_original_observation_values_and_quadrature_convergence(self):
        root=Path(__file__).resolve().parents[1]
        path=root/'assets/training/inverse_burgers/observations.csv'
        expected=json.loads((path.parent/'source.json').read_text())['observations_sha256']
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),expected)
        data=np.loadtxt(path,delimiter=',',skiprows=1);values=burgers_reference(data[:,:2])[:,0]
        self.assertLess(np.max(np.abs(values-data[:,2])),1e-9)
        self.assertLess(np.linalg.norm(values-data[:,2])/np.linalg.norm(data[:,2]),1e-10)
        np.testing.assert_allclose(values,burgers_reference(data[:,:2],order=256)[:,0],rtol=0,atol=2e-13)

    def test_initial_walls_symmetry_and_physical_residual(self):
        x=np.linspace(-1,1,65)
        values=burgers_reference(np.column_stack((x,np.zeros_like(x))))[:,0]
        np.testing.assert_allclose(values,-np.sin(np.pi*x),rtol=0,atol=2e-16)
        times=np.linspace(0,1,17)
        for wall in (-1,1):self.assertTrue(np.all(burgers_reference(np.column_stack((np.full_like(times,wall),times)))==0))
        points=np.array([[.01,.8],[.1,1],[.4,.5],[.8,.1],[0,.8],[-.2,.01]])
        opposite=points.copy();opposite[:,0]*=-1
        np.testing.assert_allclose(burgers_reference(points),-burgers_reference(opposite),rtol=0,atol=3e-14)
        # Independently differentiate the Gaussian expectation with Torch softmax.
        nodes,weights=np.polynomial.hermite.hermgauss(256)
        q=torch.tensor(points,dtype=torch.float64,requires_grad=True)
        shifted=q[:,0:1]-2*torch.sqrt(VISCOSITY*q[:,1:2])*torch.tensor(nodes)
        phase=torch.pi*shifted
        mass=torch.softmax(torch.tensor(np.log(weights))-torch.cos(phase)/(2*torch.pi*VISCOSITY),dim=1)
        u=-(mass*torch.sin(phase)).sum(1,keepdim=True)
        first=torch.autograd.grad(u.sum(),q,create_graph=True)[0]
        second=torch.autograd.grad(first[:,0].sum(),q,create_graph=True)[0][:,0:1]
        residual=first[:,1:2]+u*first[:,0:1]-VISCOSITY*second
        self.assertLess(residual.abs().max().item(),1e-8)
        np.testing.assert_allclose(u.detach().numpy(),burgers_reference(points),rtol=0,atol=1e-13)

    def test_row_order_chunk_boundaries_and_invalid_domains(self):
        rng=np.random.default_rng(18);points=np.column_stack((rng.uniform(-1,1,1100),rng.uniform(0,1,1100)))
        whole=burgers_reference(points);self.assertEqual(whole.dtype,np.float64)
        np.testing.assert_array_equal(whole,np.concatenate([burgers_reference(points[:510]),burgers_reference(points[510:])]))
        order=rng.permutation(len(points));np.testing.assert_array_equal(whole[order],burgers_reference(points[order]))
        for points in ([],[[0]],[[0,0,1]],[[True,False]],[[1+1j,0]],[[np.nan,0]],[[0,-.1]],[[0,1.1]],[[-1.1,0]],[[1.1,0]]):
            with self.subTest(points=points),self.assertRaises(ValueError):burgers_reference(points)
        for order in (True,0,8,32,129):
            with self.assertRaises(ValueError):burgers_reference([[0,.5]],order=order)

    def test_generated_dataset_hashes_frames_and_fresh_folder_guard(self):
        from examples.burgers_reference import generate
        with TemporaryDirectory() as directory:
            output=Path(directory)/'reference'
            metadata=generate(output,spatial_points=17,time_points=5)
            data=np.genfromtxt(output/'reference.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
            self.assertEqual(len(data),85);self.assertEqual(len(set(data['point_id'])),85)
            np.testing.assert_array_equal(np.unique(data['t']),np.linspace(0,1,5))
            self.assertEqual(metadata['csv_sha256'],hashlib.sha256((output/'reference.csv').read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):generate(output)
            for n,t in ((2,3),(True,5),(3,1),(4097,1001)):
                with self.assertRaises(ValueError):generate(Path(directory)/'bad',spatial_points=n,time_points=t)
        stock=Path(__file__).resolve().parents[1]/'assets/evaluation/burgers-reference'
        metadata=json.loads((stock/'source.json').read_text())
        self.assertEqual(metadata['grid']['rows'],2827)
        self.assertEqual(hashlib.sha256((stock/'reference.csv').read_bytes()).hexdigest(),metadata['csv_sha256'])




class BurgersReferenceUITests(unittest.TestCase):
    def workflow(self,device):
        with TemporaryDirectory() as directory:
            output=Path(directory)/'workflow'
            result=subprocess.run([sys.executable,'-m','examples.burgers_reference_workflow','--device',device,'--output',str(output)],
                cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=100,
                env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'))
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            report=json.loads((output/'result.json').read_text())
            self.assertEqual(report['reference_rows'],2827);self.assertEqual(report['pvd_frames'],11)
            self.assertTrue(report['exact_resume'] and report['reference_evaluation_only'])
            self.assertEqual(report['final_steps'],5)
            for name,digest in report['source_sha256'].items():self.assertEqual(hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest(),digest)

    def test_cpu_explicit_reference_owned_save_resume_csv_and_pvd(self):self.workflow('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_explicit_reference_owned_save_resume_csv_and_pvd(self):self.workflow('cuda')


if __name__=='__main__':unittest.main()
