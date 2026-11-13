"""Bounded, self-contained clipboard snapshots of explicitly referenced owned files."""
import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import stat
from uuid import uuid4
from .dataset_cleanup import existing_root
from .run_document import validate_owned_files

MIME='application/x-margam-owned-fragment'
PREFIX='margam-owned:'
FILE_LIMIT=16*1024*1024
BUNDLE_LIMIT=24*1024*1024


def references(document):
    result=set()
    def visit(item):
        if isinstance(item,dict):
            for key,value in item.items():
                if key in ('label','network_label','annotation_text','data_fingerprints'):continue
                if key in ('path','document','export') and isinstance(value,str) and value.startswith(PREFIX):result.add(value[len(PREFIX):])
                else:visit(value)
        elif isinstance(item,list):
            for value in item:visit(value)
    for node in document['nodes']:
        if node['kind']!='annotation':visit(node['settings'])
    validate_owned_files({name:b'' for name in result})
    return result


def read_owned(root,name):
    """Read only a regular file, without following links in any path component."""
    if not hasattr(os,'O_NOFOLLOW') or os.open not in os.supports_dir_fd:
        raise ValueError('Owned-file clipboard copying is unavailable on this platform.')
    descriptors=[]
    try:
        fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);descriptors.append(fd)
        for part in PurePosixPath(name).parts[:-1]:
            fd=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);descriptors.append(fd)
        fd=os.open(PurePosixPath(name).name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd);descriptors.append(fd)
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size>FILE_LIMIT:raise ValueError('Copy only regular owned files within the 16 MiB clipboard budget.')
        with os.fdopen(os.dup(fd),'rb') as stream:content=stream.read(FILE_LIMIT+1)
        after=os.fstat(fd)
        if len(content)>FILE_LIMIT or (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError('Owned file changed while copying, or exceeds the clipboard budget.')
        return content
    finally:
        for fd in reversed(descriptors):os.close(fd)


def capture_files(document,workspace,*,locked=False):
    # Make implicit Jorvik documents explicit in the copied declaration only.
    from .jorvik_geometry import document_relative
    if (locked or workspace.busy) and any(node['kind']=='jorvik_geometry' for node in document['nodes']):
        raise ValueError('Close the geometry editor or finish the job before copying geometry files.')
    root=existing_root(workspace);optional=set()
    for node in document['nodes']:
        if node['kind']!='jorvik_geometry':continue
        files=node['settings'].get('jorvik_files',{})
        if 'document' not in files and root is not None:
            name=document_relative(node['id'])
            try:read_owned(root,name)
            except FileNotFoundError:
                if node['settings'].get('jorvik'):raise ValueError('The applied geometry document is missing; restore it before copying.')
            else:files=node['settings'].setdefault('jorvik_files',{});files['document']=PREFIX+name
        if 'document' not in files and node['settings'].get('jorvik'):
            raise ValueError('The applied geometry document is missing; restore it before copying.')
        if 'document' in files and 'export' not in files:
            optional.add(str(PurePosixPath(files['document'][len(PREFIX):]).with_suffix('.nc')))
    names=references(document)
    if not names:return {}
    if locked or workspace.busy:raise ValueError('Wait for the numerical job or geometry editor before copying owned files.')
    if root is None:raise ValueError('Project-owned files are missing.')
    files={};total=0
    for name in sorted(names|optional):
        try:content=read_owned(root,name)
        except FileNotFoundError:
            if name in optional and name not in names:continue
            raise
        total+=len(content)
        if total>FILE_LIMIT or len(files)>=128:raise ValueError('Owned clipboard files exceed 16 MiB or 128 files; use Save As for larger projects.')
        files[name]=content
    return files


def encode_files(content,files):
    return json.dumps(dict(version=1,model_sha256=hashlib.sha256(content).hexdigest(),files={name:base64.b64encode(value).decode('ascii') for name,value in files.items()}),allow_nan=False).encode()


def decode_files(document,content,bundle):
    from .model_document import unique_keys
    names=references(document)
    if not bundle:
        if names:raise ValueError('Copied owned-file snapshots are missing; copy the blocks again.')
        return {}
    if len(bundle)>BUNDLE_LIMIT:raise ValueError('Owned clipboard payload exceeds 24 MiB.')
    data=json.loads(bundle,object_pairs_hook=unique_keys)
    if not isinstance(data,dict) or set(data)!={'version','model_sha256','files'} or type(data['version']) is not int or data['version']!=1 or data['model_sha256']!=hashlib.sha256(content).hexdigest():
        raise ValueError('Owned clipboard payload does not match the copied blocks.')
    encoded=data['files']
    if not isinstance(encoded,dict) or len(encoded)>128:raise ValueError('Invalid owned clipboard file count.')
    allowed=names|{str(PurePosixPath(name).with_suffix('.nc')) for name in names if name.endswith('.jorvik')}
    if not names<=set(encoded) or not set(encoded)<=allowed:raise ValueError('Owned clipboard files do not match the selected references.')
    files={};total=0
    for name,value in encoded.items():
        if not isinstance(value,str):raise ValueError('Invalid owned clipboard content.')
        try:decoded=base64.b64decode(value,validate=True)
        except ValueError:raise ValueError('Invalid encoded clipboard file.') from None
        total+=len(decoded)
        if total>FILE_LIMIT:raise ValueError('Owned clipboard files exceed 16 MiB.')
        files[name]=decoded
    validate_owned_files(files)
    return files


def rebase(document,files):
    """Fresh namespace per paste; labels and expressions remain literal text."""
    if not files:return deepcopy(document),None
    prefix='clipboard/'+uuid4().hex+'/'
    def visit(item):
        if isinstance(item,dict):
            result={}
            for key,value in item.items():
                if key in ('label','network_label','annotation_text'):result[key]=value
                elif key in ('path','document','export') and isinstance(value,str) and value.startswith(PREFIX) and value[len(PREFIX):] in files:
                    result[key]=PREFIX+prefix+value[len(PREFIX):]
                else:result[key]=visit(value)
            return result
        if isinstance(item,list):return [visit(value) for value in item]
        return item
    result=deepcopy(document)
    for node in result['nodes']:
        if node['kind']!='annotation':node['settings']=visit(node['settings'])
    return result,prefix


def install(workspace,files,prefix):
    if not files:return None
    if not hasattr(os,'O_NOFOLLOW') or os.open not in os.supports_dir_fd or os.mkdir not in os.supports_dir_fd:
        raise ValueError('Owned-file clipboard pasting is unavailable on this platform.')
    validate_owned_files(files)
    parts=PurePosixPath(prefix).parts
    if len(parts)!=2 or parts[0]!='clipboard' or parts[1] in ('.','..'):
        raise ValueError('Invalid clipboard destination.')
    root=workspace.directory;root_fd=None;parent_fd=None;folder_fd=None
    created=False
    try:
        root_fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:os.mkdir('clipboard',dir_fd=root_fd)
        except FileExistsError:pass
        parent_fd=os.open('clipboard',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=root_fd)
        os.mkdir(parts[1],dir_fd=parent_fd);created=True
        folder_fd=os.open(parts[1],os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent_fd)
        for name,content in files.items():
            descriptors=[];fd=folder_fd
            try:
                for part in PurePosixPath(name).parts[:-1]:
                    try:os.mkdir(part,dir_fd=fd)
                    except FileExistsError:pass
                    fd=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);descriptors.append(fd)
                file_fd=os.open(PurePosixPath(name).name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
                try:stream=os.fdopen(file_fd,'wb')
                except Exception:
                    os.close(file_fd);raise
                with stream:stream.write(content)
            finally:
                for fd in reversed(descriptors):os.close(fd)
    except Exception:
        if created:shutil.rmtree(parts[1],dir_fd=parent_fd)
        raise
    finally:
        for fd in (folder_fd,parent_fd,root_fd):
            if fd is not None:os.close(fd)
    return root/Path(prefix)
