"""Slurm bundle validation and real local Bash/CLI workflows; no scheduler claims."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import test_study_cli as fixture
from margam.slurm import export_job,read_job,run_job,resources
from margam.execution.study_runner import atomic_json
from margam.study_results import collect


class SlurmTests(unittest.TestCase):
    setUpClass=classmethod(fixture.StudyCLITests.setUpClass.__func__)
    setUp=fixture.StudyCLITests.setUp
    tearDown=fixture.StudyCLITests.tearDown
    wait=fixture.StudyCLITests.wait
    study=fixture.StudyCLITests.study
    cli=fixture.StudyCLITests.cli

    def snapshot(self,path):return {p.relative_to(path).as_posix():p.read_bytes() for p in path.rglob('*') if p.is_file()}
    def export(self,*,gpu=False,values='[1,2]'):
        path=self.study(values);output=self.root/'bundle with spaces $(literal)'
        job=export_job(path,output,cpus=1,memory_mb=1024,minutes=5,gpu=gpu)
        return output,job
    def bash(self,bundle,*args):
        # Slurm spools the script: use another CWD, and the submission directory env.
        spool=self.root/'spool.sbatch';spool.write_bytes((bundle/'job.sbatch').read_bytes())
        return subprocess.run(['bash',str(spool),*args],cwd=self.root,capture_output=True,text=True,timeout=100,
            env=dict(os.environ,SLURM_SUBMIT_DIR=str(bundle),MARGAM_PYTHON=sys.executable,QT_QPA_PLATFORM='offscreen'))

    def test_cli_export_is_data_only_resource_explicit_and_no_submission(self):
        path=self.study('[1]');output=self.root/'export'
        child=self.cli('study','slurm-export',path,'--output',output,'--cpus','2','--memory-mb','2048','--minutes','20','--partition','compute','--account','research')
        self.assertEqual(child.returncode,0,child.stderr);self.assertFalse(json.loads(child.stdout)['submitted'])
        job=read_job(output);self.assertEqual(job['resources']['cpus'],2);self.assertEqual(job['execution']['device'],'cpu')
        self.assertFalse((output/'results').exists());self.assertNotIn('--gpus',(output/'job.sbatch').read_text())
        check=subprocess.run(['bash','-n',str(output/'job.sbatch')],capture_output=True,text=True);self.assertEqual(check.returncode,0,check.stderr)
        code="import sys; from margam.slurm import read_job,export_job; read_job(sys.argv[1]); export_job(sys.argv[2],sys.argv[3],cpus=1,memory_mb=1,minutes=1); assert 'torch' not in sys.modules and 'PySide6' not in sys.modules"
        child=subprocess.run([sys.executable,'-c',code,str(output),str(path),str(self.root/'pure-export')],capture_output=True,text=True,timeout=20);self.assertEqual(child.returncode,0,child.stderr)
        before=self.snapshot(output)
        with self.assertRaises(FileExistsError):export_job(path,output,cpus=1,memory_mb=1,minutes=1)
        self.assertEqual(before,self.snapshot(output))

    def test_resources_and_injection_rejected_before_any_output(self):
        path=self.study('[1]')
        cases=[dict(cpus=True),dict(cpus=0),dict(memory_mb=-1),dict(minutes=0),dict(minutes=1.5),dict(gpu=1),
            dict(partition='cpu\n#SBATCH --wrap=bad'),dict(account='$(touch bad)'),dict(account='--comment=x'),dict(exports='yes')]
        for changes in cases:
            args=dict(cpus=1,memory_mb=1,minutes=1);args.update(changes);output=self.root/'bad'
            with self.assertRaises(ValueError):export_job(path,output,**args)
            self.assertFalse(output.exists())
        path.write_bytes(b'not a study')
        with self.assertRaises(ValueError):export_job(path,self.root/'bad',cpus=1,memory_mb=1,minutes=1)
        self.assertFalse((self.root/'bad').exists())

    def test_script_study_and_manifest_tampering_refused_without_training(self):
        bundle,job=self.export(values='[1]')
        for name in ('job.sbatch','study.margam-study'):
            target=bundle/name;original=target.read_bytes();target.write_bytes(original+b'changed')
            before=self.snapshot(bundle)
            with patch('margam.execution.study_runner.run_study',side_effect=AssertionError('No numerical launch')):
                with self.assertRaises(ValueError):run_job(bundle)
            self.assertEqual(before,self.snapshot(bundle));target.write_bytes(original)
        changed=deepcopy(job);changed['resources']['gpu']=True;atomic_json(bundle/'job.json',changed)
        with self.assertRaises(ValueError):read_job(bundle)
        atomic_json(bundle/'job.json',job)
        target=bundle/'job.sbatch';other=self.root/'other';other.write_bytes(target.read_bytes());target.unlink();target.symlink_to(other)
        with self.assertRaisesRegex(ValueError,'symlink'):read_job(bundle)

    def test_cpu_bash_execution_collect_and_explicit_noop_resume(self):
        bundle,job=self.export();child=self.bash(bundle);self.assertEqual(child.returncode,0,child.stderr)
        result=collect(bundle/'results');self.assertEqual(result['report']['counts'],{'completed':2});self.assertTrue(result['report']['headless'])
        self.assertEqual(result['report']['effective_execution']['device'],'cpu')
        before=self.snapshot(bundle/'results');child=self.bash(bundle);self.assertEqual(child.returncode,2,child.stderr)
        self.assertEqual(before,self.snapshot(bundle/'results'));child=self.bash(bundle,'--resume');self.assertEqual(child.returncode,0,child.stderr)
        self.assertEqual(before,self.snapshot(bundle/'results'))
        self.assertFalse((self.root/'literal').exists())

    @unittest.skipUnless(__import__('torch').cuda.is_available(),'CUDA required')
    def test_cuda_bash_request_and_actual_device(self):
        bundle,job=self.export(gpu=True,values='[1]');self.assertIn('#SBATCH --gpus=1',(bundle/'job.sbatch').read_text())
        child=self.bash(bundle);self.assertEqual(child.returncode,0,child.stderr)
        result=collect(bundle/'results');self.assertEqual(result['report']['counts'],{'completed':1})
        self.assertEqual(result['report']['effective_execution']['requested_device'],'cuda');self.assertEqual(result['report']['effective_execution']['device'],'cuda')

    def test_owned_reference_exports_and_native_gui_collection(self):
        from margam.study_results import prepare_open
        import csv
        import xml.etree.ElementTree as ET
        self.w.load_burgers_reference_example()
        for key,value in [('points',8),('boundary_points',4),('boundary_sampling',{}),('layers',(4,))]:self.w.change(key,value)
        source=self.study('[1]');bundle=self.root/'owned-job'
        export_job(source,bundle,cpus=1,memory_mb=1024,minutes=5,exports=True)
        child=self.bash(bundle);self.assertEqual(child.returncode,0,child.stderr)
        result=collect(bundle/'results');self.assertEqual(result['report']['counts'],{'completed':1})
        detail=json.loads((bundle/'results/DP0001/trial.json').read_text());self.assertEqual(len(detail['exports']),2)
        folder=bundle/'results/DP0001/artifacts'
        table=next(row for row in detail['exports'] if row['kind']=='export');series=next(row for row in detail['exports'] if row['kind']=='paraview')
        with (folder/table['path']).open() as stream:rows=list(csv.DictReader(stream))
        frames=ET.parse(folder/series['path']).getroot().findall('.//DataSet')
        self.assertEqual(len(rows),2827);self.assertEqual(len(frames),11);self.assertIn('ref_u',rows[0]);self.assertIn('point_id',rows[0])
        before=self.snapshot(bundle);temporary,path=prepare_open(result,0)
        try:
            with patch.object(self.w,'confirm_discard',return_value=True):self.assertTrue(self.w.open_saved_run(path))
        finally:temporary.cleanup()
        self.assertEqual(self.w.result['steps'],1);self.assertIsNotNone(self.w.loaded_checkpoint);self.assertTrue(self.w.result['outputs'])
        self.assertIsNotNone(self.w.project_workspace._temporary);self.assertEqual(before,self.snapshot(bundle))

    def test_launch_uses_private_verified_input_snapshot(self):
        from margam.execution.study_runner import run_study
        bundle,job=self.export(values='[1]')
        def replace_bundle(path,output,**kwargs):
            self.assertNotEqual(Path(path),bundle/'study.margam-study')
            original=(bundle/'study.margam-study').read_bytes()
            (bundle/'study.margam-study').write_bytes(b'changed after verification')
            self.assertEqual(Path(path).read_bytes(),original)
            return run_study(path,output,**kwargs)
        with patch('margam.execution.study_runner.run_study',side_effect=replace_bundle):report=run_job(bundle)
        self.assertEqual(report['counts'],{'completed':1})
        self.assertEqual(report['study_archive_sha256'],job['files']['study.margam-study']['sha256'])
        self.assertEqual(collect(bundle/'results')['report']['study_id'],job['study_id'])

    def test_resume_requires_matching_bundle_and_failed_retry_is_explicit(self):
        bundle,job=self.export(values='[1]')
        with self.assertRaises(ValueError):run_job(bundle,retry_failed=True)
        from margam.execution.transport import JobTransport,JobFailed
        with patch.object(JobTransport,'run',side_effect=JobFailed('injected failure')):report=run_job(bundle)
        self.assertEqual(report['counts'],{'failed':1});self.assertEqual(run_job(bundle,resume=True)['counts'],{'failed':1})
        self.assertEqual(run_job(bundle,resume=True,retry_failed=True)['counts'],{'completed':1})
        report=json.loads((bundle/'results/summary.json').read_text());report['study_id']='wrong';atomic_json(bundle/'results/summary.json',report)
        with self.assertRaisesRegex(ValueError,'differ'):run_job(bundle,resume=True)


if __name__=='__main__':unittest.main()
