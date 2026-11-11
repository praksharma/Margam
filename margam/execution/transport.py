"""One private, bounded file/pipe session per numerical subprocess; no Qt dependency."""
from collections import deque
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Lock,Thread
from uuid import uuid4
from .codec import load_packet,save_packet


class JobFailed(RuntimeError):
    """A failed/discarded child job, with its original diagnostic preserved."""


class JobTransport:
    def __init__(self):
        self._lock=Lock();self.process=None;self.cancelled=False;self.terminated=False;self.pid=None

    def cancel(self):
        with self._lock:
            self.cancelled=True
            if self.process is not None and self.process.poll() is None:
                try:self.process.stdin.write((json.dumps(dict(job=self.identifier,kind='cancel'))+'\n').encode());self.process.stdin.flush()
                except (OSError,ValueError):pass

    def terminate(self):
        with self._lock:
            self.terminated=True
            if self.process is not None and self.process.poll() is None:self.process.kill()

    def run(self,payload,callback=lambda event:None):
        self.identifier=uuid4().hex;payload=dict(payload,format='margam-job',version=1,id=self.identifier)
        with TemporaryDirectory(prefix='margam-job-') as directory:
            directory=Path(directory);save_packet(directory/'input.pt',payload)
            if self.terminated:raise JobFailed('Numerical job was force-stopped before launch; no result was adopted.')
            process=subprocess.Popen([sys.executable,'-m','margam.execution.worker',str(directory)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            reader=None;reader_started=False
            tail=deque(maxlen=64)
            complete=False;failure=None;sequence=0
            try:
                with self._lock:self.process=process;self.pid=process.pid
                def drain():
                    while True:
                        chunk=process.stderr.read(1024)
                        if not chunk:return
                        tail.append(chunk)
                reader=Thread(target=drain,daemon=True);reader.start();reader_started=True
                if self.cancelled:self.cancel()
                if self.terminated:self.terminate()
                while True:
                    raw=process.stdout.readline(1024*1024+1)
                    if not raw:break
                    if len(raw)>1024*1024 or not raw.endswith(b'\n'):raise ValueError('Oversized/incomplete numerical event.')
                    event=json.loads(raw)
                    if event.get('job')!=self.identifier or event.get('sequence')!=sequence+1:raise ValueError('Numerical event session/sequence mismatch.')
                    sequence+=1;kind=event.get('kind')
                    if kind not in ('ready','progress','expert','scale','sampling','complete','failed'):raise ValueError('Unknown numerical event.')
                    if complete or failure is not None:raise ValueError('Numerical event after terminal event.')
                    if kind=='complete':complete=True
                    elif kind=='failed':failure=event['message']
                    else:callback(event)
                code=process.wait()
                reader.join(1)
                if self.terminated:raise JobFailed('Numerical process was force-stopped. The running job was discarded; previously saved run files are unchanged.')
                if failure is not None:raise JobFailed(failure)
                if code!=0 or not complete:
                    diagnostic=b''.join(tail).decode('utf-8','replace')[-16384:]
                    raise JobFailed(f'Numerical process exited unexpectedly (exit {code}). No partial result/checkpoint was adopted. An OS memory kill or native failure is possible.\n'+diagnostic)
                result=load_packet(directory/'output.pt')
                if result.get('format')!='margam-job-result' or result.get('version')!=1 or result.get('id')!=self.identifier or result.get('pid')!=process.pid or not result.get('headless'):
                    raise ValueError('Numerical result session/schema mismatch.')
                if self.terminated:raise JobFailed('Numerical job was force-stopped during result transfer; no result was adopted.')
                return result
            finally:
                if process.poll() is None:process.kill();process.wait()
                for stream in (process.stdin,process.stdout,process.stderr):stream.close()
                if reader_started:reader.join(1)
                with self._lock:self.process=None
