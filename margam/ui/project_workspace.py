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

    def resolve(self, value, *, create=True):
        if not value.startswith('margam-owned:'):
            return value
        relative = value.removeprefix('margam-owned:')
        if create:directory=self.directory
        elif self.model_path:directory=self.companion(self.model_path)
        elif self._temporary is not None:directory=Path(self._temporary.name)
        else:raise FileNotFoundError('Project-owned working files are missing.')
        root = directory.resolve()
        candidate = (root / relative).resolve()
        if not relative or not candidate.is_relative_to(root) or candidate == root:
            raise ValueError('Invalid project-owned file reference.')
        return str(candidate)

    def portable(self, value):
        if not isinstance(value,str) or value.startswith('margam-owned:'):
            return value
        root = self.companion(self.model_path) if self.model_path else Path(self._temporary.name) if self._temporary else None
        if root is not None and Path(value).is_absolute():
            try:return 'margam-owned:'+Path(value).relative_to(root).as_posix()
            except ValueError:pass
            try:return 'margam-owned:'+Path(value).resolve().relative_to(root.resolve()).as_posix()
            except (ValueError,OSError,RuntimeError):pass
        return value

    def portable_document(self, value):
        if isinstance(value,dict):
            return {key:self.portable(item) if key=='path' else self.portable_document(item) for key,item in value.items()}
        if isinstance(value,list):return [self.portable_document(item) for item in value]
        return value

    def resolve_source(self, source, *, create=True):
        from copy import deepcopy
        result=deepcopy(source)
        if result:result['path']=self.resolve(result['path'],create=create)
        return result

    def store_dataset(self, source, *, copy, validator):
        """Validate a link or make a bounded, independently validated owned snapshot."""
        from uuid import uuid4
        import re
        from ..components.prepared_data import LIMIT
        if self.busy:raise ValueError('Close the external editor before importing data.')
        resolved=self.resolve_source(source)
        portable=self.portable(resolved['path'])
        owned=portable.startswith('margam-owned:')
        if not copy:
            if owned:raise ValueError('Choose an external file to create a link, or keep Copy into project.')
            resolved['path']=str(Path(resolved['path']).expanduser().absolute())
            validator(resolved)
            return resolved
        if owned:
            validator(resolved)
            return dict(resolved,path=portable)
        path=Path(resolved['path'])
        if path.suffix.lower() not in ('.csv','.npz'):raise ValueError('Copy point data from CSV or NPZ files.')
        if not path.is_file() or path.stat().st_size>LIMIT:raise ValueError('Choose a data file no larger than 64 MiB.')
        with path.open('rb') as stream:content=stream.read(LIMIT+1)
        if len(content)>LIMIT:raise ValueError('Data exceeds 64 MiB.')
        folder=self.directory/'datasets'/uuid4().hex
        name=re.sub(r'[^A-Za-z0-9_.-]','_',path.stem)[:100] or 'data'
        created=False
        try:
            folder.mkdir(parents=True);created=True
            destination=folder/(name+path.suffix.lower());destination.write_bytes(content)
            copied=dict(resolved,path=str(destination))
            validator(copied)
            return dict(copied,path=self.portable(str(destination)))
        except Exception:
            if created:shutil.rmtree(folder,ignore_errors=True)
            raise

    def resolve_config(self, config):
        from copy import deepcopy
        config = deepcopy(config)
        if config.interior_data:config.interior_data['path']=self.resolve(config.interior_data['path'])
        for source in config.boundary_data.values():source['path']=self.resolve(source['path'])
        return config

    def snapshot_to(self, path, writer):
        """Recovery copy without changing the active document or its working location."""
        source = self.companion(self.model_path) if self.model_path else Path(self._temporary.name) if self._temporary else None
        destination = self.companion(path)
        with tempfile.TemporaryDirectory(dir=Path(path).parent,prefix='.owned-recovery-') as folder:
            staging=Path(folder)/'new';backup=Path(folder)/'old'
            if source is not None and source.exists():shutil.copytree(source,staging)
            if destination.exists():destination.rename(backup)
            try:
                if staging.exists():staging.rename(destination)
                writer(path)
            except Exception:
                if destination.exists():shutil.rmtree(destination)
                if backup.exists():backup.rename(destination)
                raise
