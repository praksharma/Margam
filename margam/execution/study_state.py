"""Data-only verification and an OS-owned lease for serial study continuation."""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
from .. import study_document as sd
from ..studies import resolve_axes


STATUSES={'not_started','preparing','running','completed','invalid','incompatible','failed','interrupted'}


def attempt_directory(index,attempt):
    base=f'DP{index+1:04d}'
    return base if attempt<=1 else f'{base}/attempts/{attempt:04d}'


def safe_path(root,name):
    from ..ui.run_document import validate_owned_files
    validate_owned_files({name:b''})
    root=Path(root)
    if root.is_symlink() or not root.is_dir():raise ValueError('Choose a regular results directory, without a symlink.')
    target=root
    for part in Path(name).parts:
        target=target/part
        if target.is_symlink():raise ValueError('Study paths must not contain symlinks.')
    return target


@contextmanager
def lease(root):
    """Children inherit the same descriptor: parent death cannot release a live job."""
    if os.name!='posix':raise ValueError('Study resume currently requires a POSIX system with process-owned file locking.')
    import fcntl
    root=Path(root)
    if root.is_symlink() or not root.is_dir():raise ValueError('Choose an existing regular results directory.')
    directory=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);fd=None
    try:
        fd=os.open('.study.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600,dir_fd=directory)
        if not stat.S_ISREG(os.fstat(fd).st_mode):raise ValueError('Study lock must be a regular file.')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This results directory is already in use by a study process.') from None
        yield fd
    finally:
        # Do not LOCK_UN: an orphan worker may still own this open file description.
        if fd is not None:os.close(fd)
        os.close(directory)


def read_json(root,name):
    path=safe_path(root,name)
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW),'rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size>64*1024*1024:raise ValueError('Study JSON exceeds its regular-file budget.')
        data=stream.read(64*1024*1024+1)
    if len(data)>64*1024*1024:raise ValueError('Study JSON exceeds its byte budget.')
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate study record key.')
            result[key]=value
        return result
    value=json.loads(data,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite study JSON.')))
    sd.canonical(value)
    return value


def file_identity(root,name,*,check=None):
    path=safe_path(root,name);hasher=hashlib.sha256();size=0
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW),'rb') as stream:
        before=os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):raise ValueError('Study artifact must be a regular file.')
        while chunk:=stream.read(1024*1024):
            if check is not None and check():raise InterruptedError('Study reading cancelled.')
            size+=len(chunk);hasher.update(chunk)
        after=os.fstat(stream.fileno())
    if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('Study file changed during verification.')
    return dict(size=size,sha256=hasher.hexdigest())


def regular_tree(root,name):
    path=safe_path(root,name);result=set()
    for directory,dirs,files in os.walk(path,followlinks=False):
        for item in dirs+files:
            target=Path(directory)/item;relative=target.relative_to(root).as_posix();safe_path(root,relative)
            if not (target.is_dir() or target.is_file()):raise ValueError('Study contains a non-regular entry.')
            if target.is_file():result.add(target.relative_to(path).as_posix())
    return result


def verify_publication(root,report,study,row,*,check=None):
    prefix=row['directory']+'/artifacts/';name=prefix+'publication.json'
    publication=read_json(root,name)
    if set(publication)!={'format','version','study_id','study_archive_sha256','source_sha256','execution','trial'}:raise ValueError('Invalid publication record.')
    bindings=dict(format='margam-trial-publication',version=1,study_id=report['study_id'],study_archive_sha256=report['study_archive_sha256'],source_sha256=report['source']['sha256'],execution=report['effective_execution'])
    if any(publication[key]!=value for key,value in bindings.items()):raise ValueError('Published trial provenance differs from the frozen study.')
    detail=publication['trial']
    for key in ('index','trial_id','directory','attempt','history'):
        if detail[key]!=row[key]:raise ValueError('Published trial identity differs from the study record.')
    if detail['status']!='completed' or detail['stage']!='complete' or type(detail['steps']) is not int or not 0<=detail['steps']<=100000 or not isinstance(detail['metrics'],dict):raise ValueError('Invalid completed trial record.')
    model=read_json(root,row['directory']+'/model.margam.json');config=read_json(root,row['directory']+'/config.json')
    _,expected=sd.trial_model(study,row['index']);expected['config']=config
    if sd.canonical(model)!=sd.canonical(expected) or sd.digest(sd.canonical(model))!=detail['model_sha256'] or config!=detail['config']:raise ValueError('Published model/config differs from its frozen trial.')
    artifacts=detail['artifacts']
    if not isinstance(artifacts,list) or not 1<=len(artifacts)<=100000:raise ValueError('Invalid artifact manifest.')
    names=set()
    for artifact in artifacts:
        if check is not None and check():raise InterruptedError('Study reading cancelled.')
        if set(artifact)!={'path','size','sha256'} or artifact['path'] in names or artifact['path']=='publication.json':raise ValueError('Invalid artifact manifest entry.')
        names.add(artifact['path'])
        if file_identity(root,prefix+artifact['path'],check=check)!={key:artifact[key] for key in ('size','sha256')}:raise ValueError('Published artifact checksum changed.')
    if 'run.margam-run.pt' not in names or regular_tree(root,row['directory']+'/artifacts')!=names|{'publication.json'}:raise ValueError('Published artifact set differs from its manifest.')
    digest=file_identity(root,name,check=check)['sha256']
    if row['status']=='completed' and (row['publication_sha256']!=digest or row['steps']!=detail['steps'] or row['metrics']!=detail['metrics']):raise ValueError('Completed trial summary differs from its publication.')
    return dict(row,status='completed',stage='complete',reason=None,steps=detail['steps'],metrics=detail['metrics'],publication_sha256=digest)


def execution_record(report):
    return dict(format='margam-study-execution',version=1,**{key:deepcopy(report[key]) for key in
        ('study_id','study_archive_sha256','saved_execution','effective_execution','runtime','source')})


def verify_resume(root,report,*,retry_failed):
    """Verify every completed trial before allowing any execution or summary mutation."""
    from .study_runner import source_identity,runtime
    if report.get('format')!='margam-study-results' or report.get('version')!=2:raise ValueError('Resume requires version-2 results; older run archives remain independently usable.')
    if read_json(root,'execution.json')!=execution_record(report):raise ValueError('Frozen execution provenance differs from the summary.')
    if file_identity(root,'study.margam-study')['sha256']!=report['study_archive_sha256']:raise ValueError('Frozen study archive checksum changed.')
    study,files=sd.read(safe_path(root,'study.margam-study'))
    if report['study_id']!=study['study_id'] or report['saved_execution']!=study['execution']:raise ValueError('Frozen study identity changed.')
    if source_identity()!=report['source']:raise ValueError('Margam source changed; use a fresh results directory.')
    execution=report['effective_execution']
    if set(execution)!={'requested_device','device','wall_time_seconds','exports'} or execution['requested_device'] not in ('auto','cpu','cuda') or type(execution['exports']) is not bool:raise ValueError('Invalid execution settings.')
    from .transport import validate_wall_time
    validate_wall_time(execution['wall_time_seconds'])
    device,_,versions=runtime(execution['requested_device'])
    if device!=execution['device'] or versions!=report['runtime']:raise ValueError('Recorded runtime/device changed; use a fresh results directory.')
    _,total=resolve_axes(study['model'],study['axes']);rows=report['trials']
    if not isinstance(rows,list) or len(rows)!=total or not isinstance(report['sessions'],list) or not 1<=len(report['sessions'])<=4096:raise ValueError('Invalid study trial/session records.')
    if len(report['sessions'])==4096:raise ValueError('Study session budget reached; use a fresh results directory.')
    if report['state'] not in {'running','completed','interrupted','aborted','finished_with_errors'}:raise ValueError('Invalid study execution state.')
    if report['state']=='completed' and not all(row['status']=='completed' for row in rows):raise ValueError('Completed study contains unfinished trials.')
    if report['counts']!=dict(Counter(row['status'] for row in rows)):raise ValueError('Study status counts differ.')
    selected=[];recovered=[];result=deepcopy(report)
    for index,row in enumerate(rows):
        attempt=row['attempt']
        if type(row['index']) is not int or row['index']!=index or row['trial_id']!=f"DP{index+1:04d}-{study['study_id']}" or type(attempt) is not int or not 0<=attempt<=1000000:raise ValueError('Invalid trial identity/attempt.')
        required={'index','trial_id','directory','status','stage','reason','steps','metrics','attempt','history','publication_sha256'}
        if not required<=row.keys() or row.keys()-required-{'started_utc'}:raise ValueError('Invalid trial record fields.')
        if row['status']!='completed' and (row['metrics'] is not None or row['steps'] is not None):raise ValueError('Unfinished trial cannot have completed metrics/steps.')
        if attempt==0 and row['status']!='not_started':raise ValueError('Unstarted attempt identity differs.')
        if row['status'] not in STATUSES or row['directory']!=attempt_directory(index,attempt):raise ValueError('Invalid trial status/path.')
        if not isinstance(row['history'],list) or len(row['history'])!=max(0,attempt-1):raise ValueError('Invalid attempt history.')
        for number,old in enumerate(row['history'],1):
            if old['attempt']!=number or old['directory']!=attempt_directory(index,number) or old['trial_id']!=row['trial_id'] or old['status'] not in STATUSES-{'completed'}:raise ValueError('Invalid prior attempt record.')
        folder=safe_path(root,row['directory']);marker=safe_path(root,row['directory']+'/artifacts/publication.json')
        if folder.exists():regular_tree(root,row['directory'])
        if safe_path(root,row['directory']+'/artifacts').exists() and not marker.exists():raise ValueError('Published artifacts are missing their completion record.')
        if row['status']=='completed' or marker.exists():
            result['trials'][index]=verify_publication(root,report,study,row)
            if row['status']!='completed':recovered.append(index)
        elif row['status'] in {'not_started','preparing','running','interrupted'} or (retry_failed and row['status']=='failed'):
            if safe_path(root,attempt_directory(index,attempt+1)).exists():raise ValueError('Next attempt directory already exists; refusing to overwrite it.')
            selected.append(index)
    return result,study,files,selected,recovered
