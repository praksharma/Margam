"""Profiling is scoped, isolated and numerically transparent."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import torch
from examples.profile_dsgnar import PhaseProfiler,assert_same_state,worker
from examples.dsgnar_analytic import build


class ProfileTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_complete_cpu_attempt_and_state_parity(self):
        report=worker('wave','cpu',4,attempts=1,points=8,boundary=4)
        self.assertTrue(report['checkpoint_parity']);self.assertTrue(report['global_rng_preserved'])
        self.assertEqual(report['measured_attempts'],1);self.assertEqual(report['phases']['svd']['calls'],1)
        self.assertEqual(report['phases']['full_objective']['calls'],18)
        self.assertGreaterEqual(report['phases']['jacobian_and_sketch']['seconds'],report['phases']['jacobian_blocks']['seconds'])
        self.assertNotIn('peak_allocated_bytes',report['memory']);self.assertGreater(report['control_seconds_per_attempt'],0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_svd_peak_and_complete_attempt_peak(self):
        report=worker('wave','cuda',16,attempts=1,points=16,boundary=8,width=16,sketch=256,svd_memory=True)
        self.assertTrue(report['checkpoint_parity']);self.assertEqual(report['sketch_size'],256)
        calls=report['memory']['svd_calls'];self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]['solver_device'],'cuda:0')
        self.assertGreater(calls[0]['allocated_growth_bytes'],0)
        self.assertGreaterEqual(report['memory']['peak_allocated_bytes'],calls[0]['peak_allocated_bytes'])

    def test_explicit_architecture_precision_and_requested_sketch(self):
        report=worker('wave','cpu',4,attempts=1,points=8,boundary=4,width=4,depth=2,sketch=8,precision='float32')
        self.assertEqual(report['precision'],'float32');self.assertEqual(report['sketch_size'],8)
        self.assertEqual(report['requested_sketch_size'],8);self.assertTrue(report['checkpoint_parity'])

    def test_instrumentation_restores_methods_and_globals_on_failure(self):
        from margam.methods import dsgnar
        model=build('wave',width=4,depth=2,points=7,boundary=5,batch=3,sketch=8,probes=3)
        snapshot=model.training_state();svd=torch.linalg.svd;lm=dsgnar.lm_step;original=model._sketch
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with PhaseProfiler(model):
                with patch.object(model,'_losses',side_effect=RuntimeError('injected')):model.train(1,0)
        self.assertIs(torch.linalg.svd,svd);self.assertIs(dsgnar.lm_step,lm);self.assertEqual(model._sketch,original)
        self.assertNotIn('_jacobian_batches',model.__dict__);assert_same_state(snapshot,model.training_state())

    def test_changed_source_does_not_publish_report(self):
        from types import SimpleNamespace
        from examples.profile_dsgnar import main
        with TemporaryDirectory() as td:
            output=Path(td)/'profile.json';output.write_text('retained')
            args=['profile','--case','wave','--batches','4','--output',str(output)]
            with patch.object(sys,'argv',args),patch('examples.profile_dsgnar.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='{"seconds_per_attempt":1}')):
                with patch('examples.profile_dsgnar.profile_source',side_effect=[{'source':'before'},{'source':'after'}]):
                    with self.assertRaisesRegex(RuntimeError,'Source changed'):main()
            self.assertEqual(output.read_text(),'retained')

    def test_isolated_cli_and_failed_worker_preserve_existing_report(self):
        root=Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as td:
            p=Path(td)/'profile.json';args=[sys.executable,'-m','examples.profile_dsgnar','--case','wave','--batches','4','--points','8','--boundary','4','--attempts','1','--output',str(p)]
            result=subprocess.run(args,cwd=root,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stderr);report=json.loads(p.read_text());self.assertTrue(report['fresh_process_per_configuration'])
            self.assertTrue(report['cases'][0]['checkpoint_parity']);before=p.read_bytes()
            result=subprocess.run(args+['--batches','0'],cwd=root,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0);self.assertEqual(p.read_bytes(),before)
            result=subprocess.run(args+['--device','cuda'],cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0);self.assertEqual(p.read_bytes(),before)


if __name__=='__main__':unittest.main()
