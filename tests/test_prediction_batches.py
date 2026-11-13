"""Bounded device inference preserves point identity, fields and model state."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from margam.components.prediction import predict_in_batches,prediction_model
from margam.components.outputs import OutputRequest,ReferenceSource,evaluate_outputs
from margam.components.evaluation_data import PointData
from margam.templates.poisson import PoissonConfig,build_pinn


class PredictionBatchTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_bounded_rows_duplicates_ids_and_float64_coordinates(self):
        points=[[.123456789012345],[.9],[.123456789012345],[.3],[.7]]
        class Coordinate:
            dtype=torch.float64;iter=2
            def __init__(self):self.calls=[]
            def predict(self,x):
                self.calls.append(len(x));self.assert_dtype=x.dtype
                return x[:,0:1]
        m=Coordinate();input=PointData(points,['c','a','b','e','d'],None,{})
        ref=PointData([points[i] for i in (4,1,3,0,2)],['d','a','e','c','b'],[[points[i][0]] for i in (4,1,3,0,2)],{})
        c=PoissonConfig(shape='Interval')
        data=evaluate_outputs(m,c,[OutputRequest('x','comparison',3,None,input,ref)],prediction_batch_size=2)['x']
        self.assertEqual(m.calls,[2,2,1]);self.assertEqual(m.assert_dtype,torch.float64)
        self.assertEqual(data['points'],points);self.assertEqual(data['point_ids'],input.point_ids)
        self.assertEqual(data['values'],[row[0] for row in points]);self.assertEqual(data['rmse'],0.)

    def test_default_research_output_does_not_change_batch_membership(self):
        class Coupled:
            iter=0
            def __init__(self):self.calls=[]
            def predict(self,x):self.calls.append(len(x));return x-x.mean(0)
        m=Coupled();evaluate_outputs(m,PoissonConfig(shape='Interval'),[OutputRequest('x','solution',11)])
        self.assertEqual(m.calls,[11])

    def test_validation_late_failure_and_shape_changes_are_not_partial_results(self):
        class Model:
            def predict(self,x):return x[:,0:1]
        for bad in (0,-1,True,1.5,100001):
            with self.assertRaises(ValueError):predict_in_batches(Model(),[[0.]],bad)
        for points in ([],[[float('nan')]],[[float('inf')]],torch.ones(3)):
            with self.assertRaises(ValueError):predict_in_batches(Model(),points,2)
        class Changing:
            def __init__(self):self.n=0
            def predict(self,x):
                self.n+=1
                if self.n==2:return torch.full((len(x),1),float('nan'))
                return x[:,0:1]
        with self.assertRaisesRegex(ValueError,'finite'):predict_in_batches(Changing(),[[0.],[.1],[.2]],2)
        class Fields(Changing):
            def predict(self,x):self.n+=1;return x.repeat(1,self.n)
        with self.assertRaisesRegex(ValueError,'consistent'):predict_in_batches(Fields(),[[0.],[.1],[.2]],2)

    def _network_parity(self,device):
        from margam.components.network_config import NETWORK_NAMES
        for kind in NETWORK_NAMES:
            c=PoissonConfig(points=4,boundary_points=2,layers=(4,4),network_type=kind,device=device,network_seed=7)
            model=build_pinn(c);points=torch.rand(19,2,dtype=torch.float64)
            before=deepcopy(model.network.state_dict());rng=torch.get_rng_state().clone();mode=model.network.training
            expected=model.predict(points).cpu();actual=predict_in_batches(model,points,7)
            self.assertEqual(actual.device.type,'cpu');self.assertEqual(model.network.training,mode)
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for k,value in before.items():torch.testing.assert_close(value,model.network.state_dict()[k],rtol=0,atol=0)
            predictor=prediction_model(c,model.training_state())
            torch.testing.assert_close(predict_in_batches(predictor,points,7),expected,rtol=2e-5,atol=2e-6)

    def test_all_native_networks_cpu(self):self._network_parity('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_all_native_networks_cuda(self):self._network_parity('cuda')

    def test_real_child_large_coupled_predictions_and_retained_precision(self):
        from margam.execution.codec import encode_config,encode_request
        from margam.execution.transport import JobTransport
        from margam.execution.retained import RetainedModel
        from margam.templates.incompressible_flow import flow_demo_config
        c=flow_demo_config();c.device='cpu';c.layers=(4,);c.points=4;c.boundary_points=2
        model=build_pinn(c);model.train(1,0);checkpoint=model.training_state()
        points=[[i/5001,(5000-i)/5001] for i in range(5001)];ids=['p'+str(i) for i in range(5001)]
        source=PointData(points,ids,None,{})
        request=OutputRequest('out','export',3,None,source)
        payload=dict(operation='evaluate',config=encode_config(c),checkpoint=checkpoint,requests=[encode_request(request)],metadata={},dtype='float32')
        result=JobTransport().run(payload)['result']['outputs']['out']
        self.assertEqual(result['point_ids'],ids);self.assertEqual(result['points'],points);self.assertEqual(result['fields'],['u','v','p'])
        expected=model.predict(torch.tensor(points)).cpu()
        torch.testing.assert_close(torch.tensor(result['values']),expected,rtol=2e-5,atol=2e-6)
        for k,v in checkpoint['network'].items():torch.testing.assert_close(v,model.network.state_dict()[k],rtol=0,atol=0)
        raw=dict(payload,operation='predict',requests=[],points=torch.tensor(points))
        predicted=JobTransport().run(raw)['result']['prediction']
        self.assertEqual(predicted.device.type,'cpu')
        torch.testing.assert_close(predicted,expected,rtol=2e-5,atol=2e-6)
        # Parent compatibility path must encode float64 coordinates at the trained
        # precision, even if its own Torch default remains float32.
        state={'network':{'weight':torch.ones(1,dtype=torch.float64)},'iter':0,'loss_history':[],'training_history':[]}
        retained=RetainedModel(c,state,{},'cpu');x=.123456789012345
        with patch.object(JobTransport,'run',return_value={'result':{'prediction':torch.zeros(1)}}) as run:
            retained.predict([[x]])
        encoded=run.call_args.args[0]['points'];self.assertEqual(encoded.dtype,torch.float64);self.assertEqual(encoded.item(),x)

    def test_expression_reference_uses_float64_in_isolated_declaration(self):
        from margam.execution.codec import decode_request
        c=PoissonConfig(shape='Interval')
        request=decode_request(dict(identifier='x',kind='comparison',resolution=3,reference=dict(kind='expression',texts=['x']),point_data=None,reference_data=None),c)
        x=.123456789012345
        self.assertEqual(request.reference.evaluate([[x]])[0],x)

    def test_time_windows_and_physical_frames_preserve_order(self):
        from margam.templates.examples import example_config
        c=example_config('expert_advection');c.points=8;c.boundary_points=4;c.layers=(4,);c.network_type='fcnn';c.network_options={};c.expert_settings.update(windows=2,window_steps=1,chunks=2,balance_every=10);c.iterations=2
        model=build_pinn(c);model.train(2,0)
        points=torch.tensor([[.2,1.],[.1,0.],[.3,.5],[.2,.5],[.2,1.]])
        state=deepcopy(model.training_state());expected=model.predict(points).cpu();actual=predict_in_batches(model,points,2)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
        self.assertEqual(model.iter,state['iter'])
        for k,v in state['network'].items():torch.testing.assert_close(v,model.training_state()['network'][k],rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'outside'):predict_in_batches(model,[[.2,1.1]],1)


if __name__=='__main__':unittest.main()
