"""Archived prediction preserves method weights and works without training files."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import torch
from margam.components.prediction import prediction_model
from margam.components.network_config import NETWORK_NAMES
from margam.templates.poisson import PoissonConfig,build_pinn
from margam.templates.examples import example_config


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def check_prediction(self,c,steps=2):
        c.device='cpu';c.points=8;c.boundary_points=2;c.layers=(4,4)
        model=build_pinn(c);model.train(steps,0);state=model.training_state();before=deepcopy(state)
        points=torch.tensor([c.bounds[0],c.bounds[1]])
        restored=prediction_model(c,state)
        torch.testing.assert_close(restored.predict(points),model.predict(points),rtol=0,atol=0)
        self.assertEqual(restored.iter,steps)
        self.assertEqual(restored.training_state()['loss_history'],before['loss_history'])
        for name,value in before['network'].items():torch.testing.assert_close(value,restored.training_state()['network'][name],rtol=0,atol=0)

    def test_methods_architectures_and_time_windows(self):
        for method in ('baseline','self_adaptive','lra','hard_dirichlet'):
            with self.subTest(method=method):self.check_prediction(PoissonConfig(shape='Interval',method=method))
        for network in NETWORK_NAMES:
            with self.subTest(network=network):self.check_prediction(PoissonConfig(shape='Interval',network_type=network))
        for key in ('scale_allen_cahn','expert_torus'):
            with self.subTest(case=key):
                c=example_config(key);c.network_options['features']=2
                if c.method=='expert':c.expert_settings.update(chunks=2,balance_every=2)
                self.check_prediction(c)
        c=example_config('expert_burgers');c.network_options['features']=2
        c.expert_settings.update(chunks=2,balance_every=2,windows=2,window_steps=2)
        self.check_prediction(c,4)

    def test_prediction_without_original_training_files(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'interior.csv';path.write_text('x\n0.2\n0.7\n')
            c=PoissonConfig(shape='Interval',device='cpu',layers=(4,4),interior_data=dict(path=str(path),coordinates='x',normals='',values=''))
            model=build_pinn(c);model.train(2,0);state=model.training_state();path.unlink()
            restored=prediction_model(c,state);x=torch.tensor([[.3],[.8]])
            torch.testing.assert_close(restored.predict(x),model.predict(x),rtol=0,atol=0)
            with self.assertRaises(OSError):build_pinn(c) # Resuming training still requires its data.
