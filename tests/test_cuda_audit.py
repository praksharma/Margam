"""An audit must not turn skipped GPU checks into a passing validation report."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from examples.validate_cuda import AuditResult,audit_status,cuda_case


class CUDAAuditTests(unittest.TestCase):
    def run_case(self,method):
        class Case(unittest.TestCase):pass
        Case.test_check=method
        return unittest.TextTestRunner(stream=io.StringIO(),resultclass=AuditResult).run(unittest.TestSuite([Case('test_check')]))

    def test_only_executed_successes_pass(self):
        result=self.run_case(lambda self:None)
        self.assertEqual(audit_status(result),'passed')
        result=self.run_case(unittest.skip('GPU absent')(lambda self:None))
        self.assertEqual(audit_status(result),'incomplete');self.assertEqual(result.records[0]['status'],'skipped')
        def fail(self):self.fail('local diagnostics must not enter a portable record')
        result=self.run_case(unittest.expectedFailure(fail))
        self.assertEqual(audit_status(result),'incomplete')
        self.assertNotIn('local diagnostics',json.dumps(result.records))
        result=unittest.TextTestRunner(stream=io.StringIO(),resultclass=AuditResult).run(unittest.TestSuite())
        self.assertEqual(audit_status(result),'incomplete')

    def test_skipped_subtests_are_incomplete(self):
        def partial(self):
            with self.subTest(stage='unavailable'):self.skipTest('not executed')
        result=self.run_case(partial)
        self.assertEqual(audit_status(result),'incomplete')
        self.assertEqual(result.records[0]['status'],'skipped')

    def test_subtest_failures_and_unexpected_success_are_not_green(self):
        def subtest(self):
            with self.subTest(device='example'):self.fail('failed subtest')
        result=self.run_case(subtest)
        self.assertEqual(audit_status(result),'failed');self.assertEqual(result.records[0]['status'],'failed')
        result=self.run_case(unittest.expectedFailure(lambda self:None))
        self.assertEqual(audit_status(result),'failed')
        self.assertGreaterEqual(result.records[0]['seconds'],0)

    def test_explicit_gui_case_and_mocked_unavailability_selection(self):
        for identifier,selected in [
            ('test_pinn.CUDAPINNTests.test_cpu_gpu_losses_agree',True),
            ('test_application_device.ApplicationDeviceTests.test_training_inference_and_resume_follow_selection',True),
            ('test_application_device.ApplicationDeviceTests.test_unavailable_gpu_and_running_lock',False),
            ('test_other.Case.test_cpu_training',False)]:
            self.assertEqual(cuda_case(SimpleNamespace(id=lambda:identifier)),selected)

    def test_unavailable_cli_reports_no_execution_and_preserves_existing_file(self):
        root=Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            path=Path(directory)/'report.json';command=[sys.executable,'-m','examples.validate_cuda','--output',str(path)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
            result=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,2,result.stderr)
            data=json.loads(path.read_text());self.assertEqual(data['status'],'unavailable');self.assertEqual(data['tests'],[])
            self.assertNotIn('/home/',path.read_text())
            before=path.read_bytes()
            result=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,2);self.assertEqual(path.read_bytes(),before)
