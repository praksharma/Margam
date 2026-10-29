"""Project-owned files survive first save and Save As without touching originals."""
from pathlib import Path
import tempfile
import unittest
from margam.ui.project_workspace import ProjectWorkspace


class ProjectWorkspaceTests(unittest.TestCase):
    def test_first_save_and_save_as(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = ProjectWorkspace()
            temporary = workspace.directory
            (temporary / 'geometry').mkdir()
            (temporary / 'geometry/domain.gsproj').write_bytes(b'geometry')
            first = Path(folder) / 'first.margam.json'
            workspace.save(first, lambda path: path.write_text('model'))
            self.assertFalse(temporary.exists())
            self.assertEqual(workspace.directory.name, 'first_files')
            second = Path(folder) / 'second.margam.json'
            workspace.save(second, lambda path: path.write_text('model'))
            self.assertEqual((workspace.directory/'geometry/domain.gsproj').read_bytes(), b'geometry')
            self.assertTrue((Path(folder)/'first_files/geometry/domain.gsproj').exists())
            reopened = ProjectWorkspace(second)
            self.assertEqual(reopened.directory, workspace.directory)

    def test_failure_conflict_busy_and_discard(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = ProjectWorkspace()
            temporary = workspace.directory
            (temporary/'domain.nc').write_bytes(b'data')
            target = Path(folder)/'model.margam.json'
            def fail(path):
                raise OSError('disk failure')
            with self.assertRaises(OSError): workspace.save(target, fail)
            self.assertTrue(temporary.exists())
            self.assertFalse(workspace.companion(target).exists())
            workspace.companion(target).mkdir()
            with self.assertRaises(ValueError): workspace.save(target, lambda path: None)
            workspace.busy = True
            with self.assertRaises(ValueError): workspace.save(target, lambda path: None)
            workspace.close()
            self.assertFalse(temporary.exists())

    def test_no_unused_folder_on_plain_save(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = ProjectWorkspace()
            target = Path(folder)/'plain.json'
            workspace.save(target, lambda path: path.write_text('model'))
            self.assertFalse(workspace.companion(target).exists())
