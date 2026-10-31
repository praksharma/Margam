"""Run-owned data survives source removal; extraction rejects ambiguous paths."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class JorvikWorkflowTests(unittest.TestCase):
    def test_owned_file_validation_and_extraction(self):
        from margam.ui.run_document import validate_owned_files,archive_workspace
        for files in ({'../escape':b'x'},{'/absolute':b'x'},{'.':b'x'},
                      {'a//b':b'x'},{'a/./b':b'x'},{'a\\b':b'x'},
                      {'a':b'x','a/b':b'y'},{'C:/escape':b'x'},{'C:escape':b'x'},{'a':'text'}):
            with self.subTest(files=files),self.assertRaises(ValueError):validate_owned_files(files)
        workspace=archive_workspace({'owned_files':{'nested/sample.nc':b'opaque bytes'}})
        try:
            self.assertEqual(Path(workspace.resolve('margam-owned:nested/sample.nc')).read_bytes(),b'opaque bytes')
        finally:workspace.close()

    @unittest.skipUnless(importlib.util.find_spec('jorvik'),'Optional Jorvik required')
    def test_desktop_workflow_cpu(self):
        with TemporaryDirectory() as directory:
            result=subprocess.run([sys.executable,'-m','examples.jorvik_workflow','--directory',directory],
                cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=100,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1'))
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            self.assertIn('"archive_resumed": true',result.stdout)
