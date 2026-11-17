"""Versioned finite-grid study bundles: data-only ZIP, no code or checkpoints."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path,PurePosixPath
import stat
import tempfile
import zipfile
from .studies import resolve_axes,config_from_document,apply_choices,architecture_defaults,combinations

FILE_LIMIT=64*1024*1024
TOTAL_LIMIT=256*1024*1024
MANIFEST_LIMIT=8*1024*1024
MAX_FILES=256
PREFIX='margam-owned:'


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
def digest(value):return hashlib.sha256(value).hexdigest()


def linked_paths(model):
    result=set()
    def walk(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ('data_fingerprints','label','network_label','annotation_text'):continue
                if key=='path' and isinstance(item,str) and not item.startswith(PREFIX):result.add(item)
                else:walk(item)
        elif isinstance(value,list):
            for item in value:walk(item)
    for node in model['nodes']:
        if node['kind']!='annotation':walk(node['settings'])
    return result


def rewrite_links(value,replacements):
    if isinstance(value,dict):
        return {key:deepcopy(item) if key in ('data_fingerprints','label','network_label','annotation_text') else replacements.get(item,item) if key=='path' and isinstance(item,str) else rewrite_links(item,replacements) for key,item in value.items()}
    if isinstance(value,list):return [rewrite_links(item,replacements) for item in value]
    return deepcopy(value)


def validate(study):
    from .ui.model_document import decode_document
    from .ui.clipboard_files import references
    from .ui.run_document import validate_owned_files
    keys={'format','version','model','axes','execution','files','study_id'}
    if not isinstance(study,dict) or type(study.get('version')) is not int or study['version'] not in (1,2,3) or set(study)!=(keys|{'network_defaults','combination'} if study['version']==3 else keys|{'network_defaults'} if study['version']==2 else keys) or study['format']!='margam-study':raise ValueError('Unsupported study format/version.')
    if len(canonical(study))>MANIFEST_LIMIT:raise ValueError('Study manifest exceeds 8 MiB.')
    combination=study.get('combination','cartesian')
    if study['version']==3 and combination!='paired':raise ValueError('Version 3 requires paired combinations.')
    decode_document(study['model']);resolve_axes(study['model'],study['axes'],combination)
    if study['version']>=2 and not isinstance(study['network_defaults'],dict):raise ValueError('Frozen network defaults must be a family map.')
    families=architecture_defaults(study['model'],study['axes'],study.get('network_defaults'),combination)
    if families:
        if study['version'] not in (2,3) or not isinstance(study['network_defaults'],dict) or set(study['network_defaults'])!=set(families):raise ValueError('Architecture studies require version 2 or 3 with frozen family defaults.')
        from .components.network_config import OPTION_SCHEMAS
        if any(not isinstance(options,dict) or set(options)!=set(OPTION_SCHEMAS[kind]) for kind,options in study['network_defaults'].items()):raise ValueError('Frozen network defaults must include every supported option.')
    elif study['version']==2:raise ValueError('Version 2 is reserved for explicit network-family studies.')
    elif study.get('network_defaults',{}):raise ValueError('Unused network defaults are not allowed.')
    if linked_paths(study['model']):raise ValueError('A portable study cannot contain active linked inputs.')
    execution=study['execution']
    if not isinstance(execution,dict) or set(execution)!={'device','wall_time_seconds'} or execution['device'] not in ('auto','cpu','cuda') or type(execution['wall_time_seconds']) is not int or not 0<=execution['wall_time_seconds']<=604800:raise ValueError('Invalid study execution settings.')
    files=study['files']
    if not isinstance(files,dict) or len(files)>MAX_FILES:raise ValueError('Study has too many input files.')
    validate_owned_files({name:b'' for name in files})
    required=references(study['model'])
    optional={PurePosixPath(name).with_suffix('.nc').as_posix() for name in required if name.endswith('.jorvik')}
    if not required<=set(files) or not set(files)<=required|optional:raise ValueError('Study files do not match its owned input references.')
    total=0
    for row in files.values():
        if not isinstance(row,dict) or set(row)!={'size','sha256'} or type(row['size']) is not int or not 0<=row['size']<=FILE_LIMIT or not isinstance(row['sha256'],str) or len(row['sha256'])!=64 or any(char not in '0123456789abcdef' for char in row['sha256']):raise ValueError('Invalid study input identity.')
        total+=row['size']
    if total>TOTAL_LIMIT:raise ValueError('Study inputs exceed 256 MiB.')
    expected=digest(canonical({key:value for key,value in study.items() if key!='study_id'}))
    if study['study_id']!=expected:raise ValueError('Study identity does not match its contents.')
    return study


def _read_linked_input(source, *, limit):
    import stat
    raw_path = Path(os.path.abspath(source))
    if raw_path.suffix.lower() not in ('.csv', '.npz', '.nc'):
        raise ValueError('Study linked inputs must be CSV, NPZ or NetCDF files.')
    parent = raw_path.parent.resolve(strict=True)
    target = parent / raw_path.name
    if raw_path.is_symlink():
        # Preserve OSError(ELOOP) contract when leaf linked path itself is a symlink
        target = raw_path
    fd = os.open(target, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError('Copy only regular owned files within the selected file budget.')
        with os.fdopen(os.dup(fd), 'rb') as stream:
            content = stream.read(limit + 1)
        after = os.fstat(fd)
        if len(content) > limit or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('Owned file changed while copying, or exceeds the selected file budget.')
        return content, raw_path.suffix.lower()
    finally:
        os.close(fd)


def capture(model,axes,workspace,*,execution,include_links=False,scene=None,network_defaults=None,combination='cartesian'):
    """Freeze current inputs in an independent declaration; never edit the source."""
    from .ui.clipboard_files import capture_files
    if type(include_links) is not bool:raise ValueError('Include linked inputs must be explicitly True or False.')
    if workspace.busy:raise ValueError('Close the geometry editor before saving a study.')
    model=deepcopy(model);resolve_axes(model,axes,combination)
    # Inspector caches and application preferences are not graph source-of-truth.
    config=config_from_document(model,scene);config.device='cpu'
    model['config']=workspace.portable_document(asdict(config))
    model['wires']=sorted(model['wires'],key=lambda row:(row['source'],row['target'],row['input']))
    links=linked_paths(model)
    if links and not include_links:raise ValueError('Linked inputs are present. Select Include linked inputs to copy them into this study, or make project-owned copies first.')
    files=capture_files(model,workspace,file_limit=FILE_LIMIT,total_limit=TOTAL_LIMIT,max_files=MAX_FILES)
    replacements={};total=sum(map(len,files.values()))
    for source in sorted(links):
        content,suffix=_read_linked_input(source,limit=FILE_LIMIT)
        name='study_inputs/'+digest(content)+suffix
        if name in files and files[name]!=content:raise ValueError('Study input name conflict.')
        if name not in files:total+=len(content);files[name]=content
        if len(files)>MAX_FILES or total>TOTAL_LIMIT:raise ValueError('Study inputs exceed 256 files or 256 MiB.')
        replacements[source]=PREFIX+name
    model=rewrite_links(model,replacements)
    study=dict(format='margam-study',version=1,model=model,axes=deepcopy(axes),execution=deepcopy(execution),
        files={name:dict(size=len(content),sha256=digest(content)) for name,content in sorted(files.items())})
    defaults=architecture_defaults(model,axes,network_defaults,combination)
    if defaults:study.update(version=2,network_defaults=defaults)
    if combination=='paired':study.update(version=3,combination='paired',network_defaults=defaults)
    study=json.loads(canonical(study)) # One JSON-normal form, including cached tuple fields.
    study['study_id']=digest(canonical(study));validate(study)
    return study,files


def write(path,study,files,*,overwrite=False):
    if type(overwrite) is not bool:raise ValueError('Overwrite must be explicitly True or False.')
    validate(study)
    if set(files)!=set(study['files']) or any(not isinstance(content,bytes) or len(content)!=study['files'][name]['size'] or digest(content)!=study['files'][name]['sha256'] for name,content in files.items()):raise ValueError('Input bytes do not match the study manifest.')
    path=Path(path)
    if not path.name.endswith('.margam-study'):raise ValueError('Study files use the .margam-study extension.')
    if path.exists() and not overwrite:raise FileExistsError('Study already exists; choose a new filename or explicitly replace it.')
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.margam-study-',delete=False) as stream:temporary=Path(stream.name)
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('study.json',canonical(study))
            for name,content in sorted(files.items()):archive.writestr('files/'+name,content)
        with temporary.open('rb') as stream:os.fsync(stream.fileno())
        if overwrite:os.replace(temporary,path)
        else:os.link(temporary,path) # Atomic no-clobber publication on the same filesystem.
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def read(path):
    try:return _read(path)
    except (zipfile.BadZipFile,EOFError) as error:raise ValueError('Invalid or damaged study archive.') from error


def _read(path):
    from .ui.model_document import unique_keys
    path=Path(path)
    if path.stat().st_size>TOTAL_LIMIT+MANIFEST_LIMIT+4*1024*1024:raise ValueError('Study archive exceeds the byte budget.')
    with zipfile.ZipFile(path) as archive:
        entries=archive.infolist();names=[entry.filename for entry in entries]
        if len(entries)>MAX_FILES+1 or len(names)!=len(set(names)) or 'study.json' not in names:raise ValueError('Invalid or duplicate study archive entries.')
        for entry in entries:
            if entry.compress_type not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED):raise ValueError('Study entries require stored or deflate compression.')
            if entry.flag_bits&1 or stat.S_IFMT(entry.external_attr>>16) not in (0,stat.S_IFREG) or entry.is_dir():raise ValueError('Study entries must be ordinary unencrypted files.')
            if entry.file_size>(MANIFEST_LIMIT if entry.filename=='study.json' else FILE_LIMIT):raise ValueError('Study entry exceeds its byte budget.')
        if sum(entry.file_size for entry in entries)>TOTAL_LIMIT+MANIFEST_LIMIT:raise ValueError('Expanded study exceeds the byte budget.')
        study=json.loads(archive.read('study.json'),object_pairs_hook=unique_keys);validate(study)
        if set(names)!={'study.json',*('files/'+name for name in study['files'])}:raise ValueError('Study has missing or unexpected entries.')
        files={}
        for name,identity in study['files'].items():
            if archive.getinfo('files/'+name).file_size!=identity['size']:raise ValueError('Study input size mismatch.')
            content=archive.read('files/'+name)
            if digest(content)!=identity['sha256']:raise ValueError('Study input checksum mismatch.')
            files[name]=content
    return study,files


def restore(files):
    from .ui.run_document import archive_workspace
    if not isinstance(files,dict) or len(files)>MAX_FILES or any(not isinstance(content,bytes) or len(content)>FILE_LIMIT for content in files.values()):raise ValueError('Invalid study input restoration payload.')
    return archive_workspace({'owned_files':files})


def trial_model(study,index):
    """Resolve one deterministic trial, without opening data or executing it."""
    validate(study);choices,total=resolve_axes(study['model'],study['axes'],study.get('combination','cartesian'))
    if type(index) is not int or not 0<=index<total:raise ValueError('Trial index is outside this study.')
    values=next(itertools.islice(combinations(choices,study.get('combination','cartesian')),index,None))
    model,_,_=apply_choices(study['model'],choices,values,network_defaults=study.get('network_defaults'))
    return f"DP{index+1:04d}-{study['study_id']}",model
