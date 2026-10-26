"""Prediction and reference evaluation must be explicitly requested."""
import unittest
from unittest.mock import Mock
import torch
from margam.components.outputs import OutputRequest, ReferenceSource, evaluate_outputs
from margam.templates.poisson import PoissonConfig, make_interval_reference


class OutputRequestTests(unittest.TestCase):
    def test_no_outputs_do_not_evaluate_predictions(self):
        model=Mock()
        self.assertEqual(evaluate_outputs(model,PoissonConfig(),[]),{})
        self.assertEqual(evaluate_outputs(model,PoissonConfig(),[OutputRequest('a','monitor')]),{})
        model.predict.assert_not_called()

    def test_prediction_has_no_implicit_reference(self):
        model=Mock(iter=12)
        model.predict.side_effect=lambda points:points[:,0:1]*(1-points[:,0:1])
        result=evaluate_outputs(model,PoissonConfig(shape='Interval'),[OutputRequest('a','solution',21)])['a']
        self.assertEqual(len(result['values']),21)
        self.assertIsNone(result['reference'])
        self.assertIsNone(result['rmse'])

    def test_explicit_reference_and_resolution(self):
        model=Mock(iter=12)
        model.predict.side_effect=lambda points:points[:,0:1]*(1-points[:,0:1])
        config=PoissonConfig(shape='Interval')
        requests=[OutputRequest('first','comparison',21,make_interval_reference(config)),OutputRequest('second','solution',11)]
        outputs=evaluate_outputs(model,config,requests)
        self.assertLess(outputs['first']['rmse'],1e-7)
        self.assertEqual(len(outputs['second']['points']),11)
        with self.assertRaisesRegex(ValueError,'explicit reference'):
            evaluate_outputs(model,config,[OutputRequest('bad','comparison')])
        with self.assertRaisesRegex(ValueError,'intervals only'):
            make_interval_reference(PoissonConfig())

    def test_mismatched_reference_is_rejected(self):
        model=Mock(iter=1)
        model.predict.side_effect=lambda points:torch.zeros(len(points),1)
        with self.assertRaisesRegex(ValueError,'match'):
            evaluate_outputs(model,PoissonConfig(shape='Interval'),[
                OutputRequest('a','comparison',11,ReferenceSource('bad',lambda points:[0.]))])
