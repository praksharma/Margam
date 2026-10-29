"""Lazy document-owned files, independent of the process working directory."""
from pathlib import Path
import shutil
import tempfile


class ProjectWorkspace:
    def __init__(self, model_path=None):
        self.model_path = Path(model_path).absolute() if model_path else None
        self._temporary = None
        self.busy = False

    @staticmethod
    def companion(model_path):
        path = Path(model_path).absolute()
        name = path.name
        suffix = '.margam.json' if name.endswith('.margam.json') else path.suffix
        return path.with_name(name[:-len(suffix)] + '_files' if suffix else name + '_files')

    @property
    def directory(self):
        if self.model_path:
            directory = self.companion(self.model_path)
            directory.mkdir(exist_ok=True)
            return directory
        if self._temporary is None:
            self._temporary = tempfile.TemporaryDirectory(prefix='margam-untitled-')
        return Path(self._temporary.name)

    def save(self, path, writer):
        """Copy owned files before writing; retain the source if either operation fails."""
        path = Path(path).absolute()
        if self.busy:
            raise ValueError('Close the external project editor before saving or relocating the project.')
        source = (self.companion(self.model_path) if self.model_path else
                  Path(self._temporary.name) if self._temporary else None)
        destination = self.companion(path)
        copied = False
        try:
            if source is not None and source.exists() and source != destination:
                if destination.exists():
                    raise ValueError('The destination project files folder already exists. Choose another model name.')
                # Mark ownership before copying so a partial copy is removed on failure.
                copied = True
                shutil.copytree(source, destination)
            writer(path)
        except Exception:
            if copied:
                shutil.rmtree(destination)
            raise
        self.model_path = path
        self.close()

    def close(self):
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
