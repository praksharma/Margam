"""Grid continuation preserves completed data, freezes provenance and locks workers."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
try:
    from . import test_study_cli as fixture
except ImportError:
    import test_study_cli as fixture
from margam.execution.study_runner import run_study,resume_study,atomic_json
from margam.execution.study_state import lease
from margam.execution.transport import JobTransport,JobFailed


@unittest.skipUnless(os.name=='posix','POSIX study lease required')
class StudyResumeTests(unittest.TestCase):
    setUpClass=classmethod(fixture.StudyCLITests.setUpClass.__func__)
    setUp=fixture.StudyCLITests.setUp
    tearDown=fixture.StudyCLITests.tearDown
    wait=fixture.StudyCLITests.wait
    study=fixture.StudyCLITests.study
    cli=fixture.StudyCLITests.cli

    def snapshot(self,path):
        return {p.relative_to(path).as_posix():(p.read_bytes(),p.stat().st_mtime_ns) for p in path.rglob('*') if p.is_file()}

    def interrupted(self,*,device='cpu',exports=False):
        path=self.study('[1,2,3]');output=self.root/'results';actual=JobTransport.run;calls=0
        def interrupt(transport,*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2:raise KeyboardInterrupt
            return actual(transport,*args,**kwargs)
        with patch.object(JobTransport,'run',new=interrupt):report=run_study(path,output,device=device,exports=exports)
        self.assertEqual(report['counts'],{'completed':1,'interrupted':1,'not_started':1})
        return output,report

    def check_restart(self,device):
        output,before=self.interrupted(device=device);completed=self.snapshot(output/'DP0001');old=self.snapshot(output/'DP0002')
        report=resume_study(output);self.assertEqual(report['counts'],{'completed':3})
        self.assertEqual(completed,self.snapshot(output/'DP0001'))
        self.assertEqual([row['attempt'] for row in report['trials']],[1,2,1])
        self.assertEqual(report['trials'][1]['directory'],'DP0002/attempts/0002')
        self.assertEqual(report['trials'][1]['history'][0]['status'],'interrupted')
        for name,identity in old.items():self.assertEqual(identity,self.snapshot(output/'DP0002')[name])
        self.assertEqual(report['sessions'][-1]['selected'],[1,2]);self.assertEqual(report['effective_execution']['device'],device)
        from margam.ui.run_document import load_run
        a=load_run(output/'DP0002/attempts/0002/artifacts/run.margam-run.pt')
        b=load_run(output/'DP0003/artifacts/run.margam-run.pt')
        self.assertEqual(a['checkpoint']['iter'],2);self.assertEqual(b['checkpoint']['iter'],3)
        snapshot=self.snapshot(output)
        with patch.object(JobTransport,'run',side_effect=AssertionError('No new job')):self.assertEqual(resume_study(output),report)
        self.assertEqual(snapshot,self.snapshot(output))

    def test_cpu_interrupted_and_unstarted_restart_without_touching_completed(self):self.check_restart('cpu')

    @unittest.skipUnless(__import__('torch').cuda.is_available(),'CUDA required')
    def test_cuda_restart_and_completed_idempotence(self):self.check_restart('cuda')

    def test_owned_reference_and_physical_time_exports_after_resume(self):
        self.w.load_burgers_reference_example()
        for key,value in [('points',8),('boundary_points',4),('boundary_sampling',{}),('layers',(4,))]:self.w.change(key,value)
        output,_=self.interrupted(exports=True);completed=self.snapshot(output/'DP0001')
        report=resume_study(output);self.assertEqual(report['counts'],{'completed':3})
        self.assertEqual(completed,self.snapshot(output/'DP0001'))
        from margam.ui.run_document import load_run
        import csv
        import xml.etree.ElementTree as ET
        for row in report['trials']:
            folder=output/row['directory'];detail=json.loads((folder/'trial.json').read_text())
            saved=load_run(folder/'artifacts/run.margam-run.pt')
            self.assertEqual(len(saved['owned_files']),1);self.assertEqual(saved['result']['steps'],row['index']+1)
            self.assertTrue(row['metrics']);self.assertEqual(len(detail['exports']),2)
            table=next(item for item in detail['exports'] if item['kind']=='export')
            series=next(item for item in detail['exports'] if item['kind']=='paraview')
            with (folder/'artifacts'/table['path']).open() as stream:points=list(csv.DictReader(stream))
            frames=ET.parse(folder/'artifacts'/series['path']).getroot().findall('.//DataSet')
            self.assertEqual(len(points),2827);self.assertEqual(len(frames),11)
            self.assertIn('point_id',points[0]);self.assertIn('ref_u',points[0])
            self.assertEqual([float(frame.attrib['timestep']) for frame in frames],sorted({float(point['t']) for point in points}))
        before=self.snapshot(output);resume_study(output);self.assertEqual(before,self.snapshot(output))

    def test_interrupt_before_attempt_directory_creation_is_recoverable(self):
        path=self.study('[1]');output=self.root/'before-directory';interrupted=False
        def interrupt(path,value):
            nonlocal interrupted
            atomic_json(path,value)
            if Path(path).name=='summary.json' and value['trials'][0]['status']=='preparing' and not interrupted:
                interrupted=True;raise KeyboardInterrupt
        with patch('margam.execution.study_runner.atomic_json',side_effect=interrupt):report=run_study(path,output)
        self.assertEqual(report['state'],'interrupted');self.assertFalse((output/'DP0001').exists())
        result=resume_study(output);self.assertEqual(result['counts'],{'completed':1})
        self.assertEqual(result['trials'][0]['attempt'],2)

    def test_failed_retry_is_explicit_invalid_trials_are_retained(self):
        path=self.study('[[4],[-1],[8]]','layers');output=self.root/'retry';actual=JobTransport.run;calls=0
        def fail_once(transport,*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==1:raise JobFailed('injected retryable failure')
            return actual(transport,*args,**kwargs)
        with patch.object(JobTransport,'run',new=fail_once):report=run_study(path,output)
        self.assertEqual(report['counts'],{'failed':1,'invalid':1,'completed':1})
        before=self.snapshot(output)
        with patch.object(JobTransport,'run',side_effect=AssertionError('No implicit retry')):resume_study(output)
        self.assertEqual(before,self.snapshot(output));invalid=self.snapshot(output/'DP0002')
        later=output/'DP0003/artifacts/run.margam-run.pt';data=later.read_bytes();later.write_bytes(data+b'changed')
        before=self.snapshot(output)
        with patch.object(JobTransport,'run',side_effect=AssertionError('Verify every completed trial first')):
            with self.assertRaisesRegex(ValueError,'checksum'):resume_study(output,retry_failed=True)
        self.assertEqual(before,self.snapshot(output));later.write_bytes(data)
        report=resume_study(output,retry_failed=True)
        self.assertEqual(report['counts'],{'completed':2,'invalid':1});self.assertEqual(invalid,self.snapshot(output/'DP0002'))
        self.assertEqual(report['trials'][0]['attempt'],2)

    def test_fully_published_trial_is_recovered_without_numerical_work(self):
        path=self.study('[1]');output=self.root/'published';report=run_study(path,output)
        completed=self.snapshot(output/'DP0001');row=report['trials'][0]
        row.update(status='running',stage='artifacts',steps=None,metrics=None,publication_sha256=None)
        report.update(state='running',finished_utc=None,counts={'running':1});atomic_json(output/'summary.json',report)
        with patch.object(JobTransport,'run',side_effect=AssertionError('Must adopt already published data')):result=resume_study(output)
        self.assertEqual(result['counts'],{'completed':1});self.assertEqual(result['sessions'][-1]['recovered_published'],[0])
        self.assertEqual(completed,self.snapshot(output/'DP0001'));self.assertEqual(result['trials'][0]['attempt'],1)

    def test_stale_final_summary_is_completed_without_new_numerical_work(self):
        path=self.study('[1]');output=self.root/'final-summary';report=run_study(path,output)
        completed=self.snapshot(output/'DP0001');report.update(state='running',finished_utc=None)
        atomic_json(output/'summary.json',report)
        with patch.object(JobTransport,'run',side_effect=AssertionError('No repeated numerical work')):result=resume_study(output)
        self.assertEqual(result['state'],'completed');self.assertEqual(result['sessions'][-1]['selected'],[])
        self.assertEqual(completed,self.snapshot(output/'DP0001'))

    def test_corruption_and_provenance_refused_before_any_mutation_or_job(self):
        output,report=self.interrupted();summary=output/'summary.json';original=summary.read_bytes()
        changes=[lambda r:r.update(version=1),lambda r:r['runtime'].update(torch='different'),lambda r:r['source'].update(sha256='0'*64),
            lambda r:r.update(study_archive_sha256='0'*64),lambda r:r['trials'][1].update(directory='../escape'),
            lambda r:r['effective_execution'].update(device='cuda'),lambda r:r['effective_execution'].update(exports=True),lambda r:r['effective_execution'].update(wall_time_seconds=1),lambda r:r['trials'][0].update(metrics={'fake':1}),
            lambda r:r['trials'][1].update(attempt=True)]
        for mutate in changes:
            changed=deepcopy(report);mutate(changed);atomic_json(summary,changed);before=self.snapshot(output)
            with patch.object(JobTransport,'run',side_effect=AssertionError('No launch')):
                with self.assertRaises((ValueError,KeyError,TypeError)):resume_study(output)
            self.assertEqual(before,self.snapshot(output));summary.write_bytes(original)
        # Verify a corrupt completed trial before restarting any earlier unfinished row.
        archive=output/'DP0001/artifacts/run.margam-run.pt';archive.write_bytes(archive.read_bytes()+b'changed');before=self.snapshot(output)
        with self.assertRaisesRegex(ValueError,'checksum'):resume_study(output)
        self.assertEqual(before,self.snapshot(output))

    def test_cli_resume_and_exclusive_lock(self):
        output,report=self.interrupted()
        with lease(output):
            before=self.snapshot(output);result=self.cli('study','resume',output)
            self.assertEqual(result.returncode,2,result.stderr);self.assertIn('already in use',result.stderr);self.assertEqual(before,self.snapshot(output))
        result=self.cli('study','resume',output);self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['counts'],{'completed':3})

    def test_inherited_lease_survives_parent_descriptor_close(self):
        output=self.root/'lease';output.mkdir();child=None
        try:
            with lease(output) as fd:
                child=subprocess.Popen([sys.executable,'-c','import sys; print("ready",flush=True); sys.stdin.read()'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,pass_fds=(fd,))
                self.assertEqual(child.stdout.readline().strip(),'ready')
            with self.assertRaisesRegex(RuntimeError,'already in use'):
                with lease(output):pass
            child.communicate('',timeout=10)
            with lease(output):pass
        finally:
            if child is not None and child.poll() is None:child.kill();child.communicate()

    def test_symlink_and_unexpected_attempt_paths_are_refused(self):
        output,report=self.interrupted();target=output/'DP0002/attempts/0002';target.mkdir(parents=True)
        before=self.snapshot(output)
        with self.assertRaisesRegex(ValueError,'already exists'):resume_study(output)
        self.assertEqual(before,self.snapshot(output));target.rmdir();target.parent.rmdir()
        archive=output/'DP0001/artifacts/run.margam-run.pt';original=archive.read_bytes();other=self.root/'archive';other.write_bytes(original);archive.unlink();archive.symlink_to(other)
        with self.assertRaisesRegex(ValueError,'symlink'):resume_study(output)

    def test_real_cli_interrupt_then_resume(self):
        import signal
        path=self.study('[1,200,2]');output=self.root/'real-interrupt';child=None
        process=subprocess.Popen([sys.executable,'-m','margam.cli','study','run',str(path),'--output',str(output)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            deadline=time.monotonic()+45
            while time.monotonic()<deadline:
                events=output/'DP0002/events.jsonl'
                if events.exists():
                    for line in events.read_text().splitlines():
                        try:value=json.loads(line)
                        except json.JSONDecodeError:continue
                        if value['kind']=='ready':child=value['pid'];break
                if child is not None or process.poll() is not None:break
                time.sleep(.01)
            self.assertIsNotNone(child);os.kill(child,signal.SIGSTOP)
            process.send_signal(signal.SIGINT);_,stderr=process.communicate(timeout=20)
            self.assertEqual(process.returncode,130,stderr)
            before=self.snapshot(output/'DP0001');old=self.snapshot(output/'DP0002')
            result=self.cli('study','resume',output);self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((output/'summary.json').read_text());self.assertEqual(report['counts'],{'completed':3})
            self.assertEqual(before,self.snapshot(output/'DP0001'))
            for key,value in old.items():self.assertEqual(value,self.snapshot(output/'DP0002')[key])
            with self.assertRaises(ProcessLookupError):os.kill(child,0)
            self.assertEqual(report['trials'][1]['attempt'],2)
        finally:
            if process.poll() is None:process.kill();process.communicate()


if __name__=='__main__':unittest.main()
