"""Run the existing CUDA-specific regression cases and emit a portable audit.

Uses the current root environment; never installs drivers or dependencies.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest

MODULES=('test_pinn.py','test_benchmarking.py','test_boundary_weights.py',
    'test_expert.py','test_expert_extensions.py','test_scale.py','test_gpinn.py',
    'test_rba.py','test_inverse_burgers.py','test_research.py',
    'test_research_schedules.py','test_integral_constraints.py','test_resources.py','test_dsgnar.py','test_dsgnar_profile.py','test_dsgnar_ui.py','test_isolated_jobs.py','test_sample_points.py','test_application_device.py',
    'test_dataset_storage.py')
EXPLICIT={'test_application_device.ApplicationDeviceTests.test_training_inference_and_resume_follow_selection'}
# This case deliberately mocks GPU unavailability; it belongs to CPU UI coverage.
EXCLUDED={'test_application_device.ApplicationDeviceTests.test_unavailable_gpu_and_running_lock'}


def flatten(suite):
    for value in suite:
        if isinstance(value,unittest.TestSuite):yield from flatten(value)
        else:yield value


def cuda_case(test):
    identifier=test.id()
    return identifier not in EXCLUDED and ('cuda' in identifier.lower() or 'gpu' in identifier.lower() or identifier in EXPLICIT)


def collect(root):
    cases=[]
    for filename in MODULES:
        loader=unittest.TestLoader();suite=loader.discover(str(root/'tests'),pattern=filename)
        if loader.errors:raise RuntimeError('Cannot discover CUDA checks in '+filename+'; run its unittest module for diagnostics.')
        selected=[test for test in flatten(suite) if cuda_case(test)]
        if not selected:raise RuntimeError('No CUDA checks selected from '+filename+'. Update the explicit audit manifest.')
        cases.extend(selected)
    return cases


class AuditResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.records=[];self.current={};self.started={}

    def startTest(self,test):
        super().startTest(test);self.started[test.id()]=time.perf_counter();self.current[test.id()]='running'

    def addSuccess(self,test):
        super().addSuccess(test)
        if self.current[test.id()]=='running':self.current[test.id()]='passed'
    def addFailure(self,test,error):super().addFailure(test,error);self.current[test.id()]='failed'
    def addError(self,test,error):super().addError(test,error);self.current[test.id()]='error'
    def addSkip(self,test,reason):
        super().addSkip(test,reason)
        self.current[getattr(test,'test_case',test).id()]='skipped'
    def addExpectedFailure(self,test,error):super().addExpectedFailure(test,error);self.current[test.id()]='expected_failure'
    def addUnexpectedSuccess(self,test):super().addUnexpectedSuccess(test);self.current[test.id()]='unexpected_success'

    def addSubTest(self,test,subtest,error):
        super().addSubTest(test,subtest,error)
        if error is not None:self.current[test.id()]='failed'

    def stopTest(self,test):
        self.records.append(dict(test=test.id(),status=self.current.pop(test.id()),seconds=time.perf_counter()-self.started.pop(test.id())))
        super().stopTest(test)


def audit_status(result):
    # A green skipped/expected-failure suite is not proof that GPU checks ran.
    if not result.wasSuccessful():return 'failed'
    return 'passed' if result.records and not result.skipped and not result.expectedFailures and all(row['status']=='passed' for row in result.records) else 'incomplete'


def source_identity(root):
    digest=hashlib.sha256()
    paths=sorted((root/'margam').rglob('*.py'))+[root/'tests'/name for name in MODULES]
    for path in paths:
        name=path.relative_to(root).as_posix().encode();data=path.read_bytes()
        digest.update(len(name).to_bytes(8,'big'));digest.update(name)
        digest.update(len(data).to_bytes(8,'big'));digest.update(data)
    try:revision=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,check=True).stdout.strip()
    except (OSError,subprocess.CalledProcessError):revision=None
    return dict(base_commit=revision,runtime_and_test_sha256=digest.hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Optional JSON report destination. Existing files need --overwrite.')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    if args.output is not None and args.output.exists() and not args.overwrite:parser.error('Report already exists; use --overwrite explicitly.')
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    import torch
    root=Path(__file__).resolve().parents[1]
    report=dict(format='margam-cuda-validation',version=1,
        scope='Targeted existing CUDA regression checks, not the full test suite or a paper-accuracy reproduction.',
        python=platform.python_version(),torch=torch.__version__,compiled_cuda=torch.version.cuda,
        source=source_identity(root),status='unavailable',tests=[])
    code=2
    if torch.cuda.is_available():
        torch.set_num_threads(1)
        device=torch.cuda.current_device();properties=torch.cuda.get_device_properties(device)
        report['device']=dict(index=device,name=properties.name,capability=list(torch.cuda.get_device_capability(device)),memory_bytes=properties.total_memory)
        cases=collect(root)
        result=unittest.TextTestRunner(stream=sys.stderr,verbosity=2,resultclass=AuditResult).run(unittest.TestSuite(cases))
        report.update(status=audit_status(result),tests=result.records)
        code=0 if report['status']=='passed' else 1 if report['status']=='failed' else 2
    else:report['device']=None
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        # Reports intentionally contain no traceback messages or absolute paths.
        args.output.write_text(text)
    print(text,end='')
    return code


if __name__=='__main__':raise SystemExit(main())
