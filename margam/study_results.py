"""Read-only collection of native version-2 study results; no numerical execution."""
from collections import Counter
from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
from . import study_document as sd
from .studies import resolve_axes
from .execution.study_state import (read_json,safe_path,file_identity,execution_record,
    attempt_directory,verify_publication,STATUSES)


def cancelled(check):
    if check is not None and check():raise InterruptedError('Study reading cancelled.')


def collect(root,*,check=None):
    """Read reported outcomes, without verifying/loading numerical payloads yet."""
    root=Path(root).absolute();cancelled(check)
    summary_sha=file_identity(root,'summary.json',check=check)['sha256'];report=read_json(root,'summary.json')
    if not isinstance(report,dict) or report.get('format')!='margam-study-results' or type(report.get('version')) is not int or report['version']!=2:raise ValueError('Collection requires version-2 study results. Open older native run archives individually.')
    if read_json(root,'execution.json')!=execution_record(report):raise ValueError('Frozen execution provenance differs from the summary.')
    execution=report['effective_execution'];runtime=report['runtime'];source=report['source']
    if not isinstance(execution,dict) or set(execution)!={'requested_device','device','wall_time_seconds','exports'} or execution['requested_device'] not in ('auto','cpu','cuda') or execution['device'] not in ('cpu','cuda') or type(execution['exports']) is not bool:raise ValueError('Invalid recorded execution settings.')
    limit=execution['wall_time_seconds']
    if type(limit) not in (int,float) or not math.isfinite(limit) or not 0<=limit<=604800:raise ValueError('Invalid recorded job limit.')
    if not isinstance(runtime,dict) or not {'python','torch','system','machine'}<=runtime.keys() or any(not isinstance(runtime[key],str) or not runtime[key] or len(runtime[key])>1024 for key in ('python','torch','system','machine')):raise ValueError('Invalid recorded runtime metadata.')
    if not isinstance(source,dict) or not isinstance(source.get('sha256'),str) or len(source['sha256'])!=64 or any(c not in '0123456789abcdef' for c in source['sha256']):raise ValueError('Invalid recorded source identity.')

    if file_identity(root,'study.margam-study',check=check)['sha256']!=report['study_archive_sha256']:raise ValueError('Frozen study checksum changed.')
    study,_=sd.read(safe_path(root,'study.margam-study'));cancelled(check)
    if study['study_id']!=report['study_id'] or study['execution']!=report['saved_execution']:raise ValueError('Study identity differs from the results.')
    _,total=resolve_axes(study['model'],study['axes']);rows=report['trials']
    if not isinstance(rows,list) or len(rows)!=total:raise ValueError('Study trial count differs.')
    if report['state'] not in {'running','completed','interrupted','aborted','finished_with_errors'}:raise ValueError('Invalid study state.')
    if report['counts']!=dict(Counter(row['status'] for row in rows)):raise ValueError('Study status counts differ.')
    if report['state']=='completed' and any(row['status']!='completed' for row in rows):raise ValueError('Completed study contains unfinished trials.')
    records=[]
    for index,row in enumerate(rows):
        cancelled(check);attempt=row['attempt']
        if type(row['index']) is not int or row['index']!=index or row['trial_id']!=f"DP{index+1:04d}-{study['study_id']}" or type(attempt) is not int or not 0<=attempt<=1000000:raise ValueError('Invalid trial identity/attempt.')
        if row['directory']!=attempt_directory(index,attempt) or row['status'] not in STATUSES:raise ValueError('Invalid trial path/status.')
        if (attempt==0)!=(row['status']=='not_started'):raise ValueError('Invalid unstarted attempt.')
        safe_path(root,row['directory'])
        if row['reason'] is not None and not isinstance(row['reason'],str):raise ValueError('Invalid trial reason.')
        if not isinstance(row['history'],list) or len(row['history'])!=max(0,attempt-1):raise ValueError('Invalid attempt history.')
        for number,old in enumerate(row['history'],1):
            if old['attempt']!=number or old['directory']!=attempt_directory(index,number) or old['trial_id']!=row['trial_id'] or old['status'] not in STATUSES-{'completed'}:raise ValueError('Invalid prior attempt record.')
        if row['status']=='completed':
            if type(row['steps']) is not int or not 0<=row['steps']<=100000 or not isinstance(row['metrics'],dict):raise ValueError('Invalid completed trial diagnostics.')
            for identifier,metric in row['metrics'].items():
                if not isinstance(identifier,str) or not isinstance(metric,dict) or 'value' not in metric or metric.get('norm')!='relative_l2' or not isinstance(metric.get('status'),str) or not isinstance(metric.get('fields'),list) or type(metric.get('rows')) is not int or metric['rows']<1:raise ValueError('Invalid reference diagnostic.')
                value=metric.get('value')
                if value is not None and (type(value) not in (int,float) or value<0):raise ValueError('Invalid reference error value.')
        elif row['steps'] is not None or row['metrics'] is not None:raise ValueError('Unfinished trials cannot have completed diagnostics.')
        _,model=sd.trial_model(study,index);by_id={node['id']:node for node in model['nodes']}
        choices=[dict(block=by_id[axis['node_id']]['title'],node_id=axis['node_id'],parameter=axis['parameter'],
            value=deepcopy(by_id[axis['node_id']]['settings']['parameters'][axis['parameter']])) for axis in study['axes']]
        records.append(dict(row=deepcopy(row),choices=choices))
    if file_identity(root,'summary.json')['sha256']!=summary_sha:raise ValueError('Study progress changed while reading. Refresh the collection.')
    return dict(root=str(root),summary_sha256=summary_sha,report=report,study=study,records=records)


def prepare_open(collection,index,*,check=None):
    """Verify selected publication and return a private checksum-verified archive copy.

    Caller owns/cleans the returned TemporaryDirectory after native archive opening.
    No network/checkpoint deserialization occurs here.
    """
    if type(index) is not int or not 0<=index<len(collection['records']):raise ValueError('Choose a recorded trial.')
    root=Path(collection['root']);fresh=collect(root,check=check)
    if fresh['summary_sha256']!=collection['summary_sha256']:raise ValueError('Study results changed. Refresh before opening a trial.')
    collection=fresh;row=collection['records'][index]['row'];cancelled(check)
    if file_identity(root,'summary.json')['sha256']!=collection['summary_sha256']:raise ValueError('Study results changed. Refresh before opening a trial.')
    if row['status']!='completed':raise ValueError('Only a reported completed trial can be opened.')
    verify_publication(root,collection['report'],collection['study'],row,check=check)
    publication=read_json(root,row['directory']+'/artifacts/publication.json')
    expected=next(item for item in publication['trial']['artifacts'] if item['path']=='run.margam-run.pt')
    from .ui.run_document import LIMIT
    if expected['size']>LIMIT:raise ValueError('Run exceeds native archive byte budget.')
    source=safe_path(root,row['directory']+'/artifacts/run.margam-run.pt');temporary=TemporaryDirectory(prefix='margam-study-open-')
    try:
        target=Path(temporary.name)/'run.margam-run.pt';digest=hashlib.sha256();size=0
        with os.fdopen(os.open(source,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as stream,target.open('xb') as output:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):raise ValueError('Run archive must be a regular file.')
            while chunk:=stream.read(1024*1024):
                cancelled(check);size+=len(chunk)
                if size>LIMIT:raise ValueError('Run exceeds native archive byte budget.')
                digest.update(chunk);output.write(chunk)
        if size!=expected['size'] or digest.hexdigest()!=expected['sha256']:raise ValueError('Run changed during the verified copy.')
        if file_identity(root,'summary.json')['sha256']!=collection['summary_sha256']:raise ValueError('Study results changed. Refresh before opening a trial.')
        if file_identity(root,row['directory']+'/artifacts/publication.json',check=check)['sha256']!=row['publication_sha256']:raise ValueError('Publication changed during the verified copy.')
        cancelled(check);return temporary,target
    except BaseException:temporary.cleanup();raise


def export_csv(collection,path):
    """Export the collected metadata snapshot; no payload reads, ranking or refresh.

    Arbitrary text is JSON-encoded in cells, preserving it without spreadsheet
    formula interpretation. Empty metric cells mean absent, never zero.
    """
    import csv
    import io
    import json
    from tempfile import NamedTemporaryFile
    report=collection['report'];records=collection['records']
    identifiers=sorted({key for record in records for key in (record['row']['metrics'] or {})})
    fields=['csv_version','metadata_scope','study_id','summary_sha256','study_state','design_point','trial_id','reported_status','attempt','actual_steps','choices_json','reason_json','attempt_history_json','device','requested_device','source_sha256','runtime_json']
    for key in identifiers:fields.extend(f'metric:{key}:{name}' for name in ('value','status_json','norm','fields_json','rows'))
    def encode(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(',',':'))
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for index,record in enumerate(records):
        row=record['row'];execution=report['effective_execution']
        result=dict(csv_version=1,metadata_scope='reported_snapshot_payloads_unverified',study_id=report['study_id'],summary_sha256=collection['summary_sha256'],study_state=report['state'],design_point=f'DP{index+1:04d}',trial_id=row['trial_id'],reported_status=row['status'],attempt=row['attempt'],actual_steps='' if row['steps'] is None else row['steps'],choices_json=encode(record['choices']),reason_json=encode(row['reason']),attempt_history_json=encode(row['history']),device=execution['device'],requested_device=execution['requested_device'],source_sha256=report['source']['sha256'],runtime_json=encode(report['runtime']))
        for key,metric in (row['metrics'] or {}).items():
            for name in ('value','norm','rows'):result[f'metric:{key}:{name}']='' if metric[name] is None else metric[name]
            result[f'metric:{key}:status_json']=encode(metric['status'])
            result[f'metric:{key}:fields_json']=encode(metric['fields'])
        writer.writerow(result)
        if stream.tell()>32*1024*1024:raise ValueError('Study CSV exceeds the 32 MiB export budget.')
    content=stream.getvalue().encode('utf-8')
    if len(content)>32*1024*1024:raise ValueError('Study CSV exceeds the 32 MiB export budget.')
    path=Path(path).absolute();temporary=None
    try:
        with NamedTemporaryFile(prefix='.margam-study-csv-',dir=path.parent,delete=False) as output:
            temporary=Path(output.name);output.write(content);output.flush();os.fsync(output.fileno())
        os.link(temporary,path) # Atomic no-clobber; existing files/symlinks remain intact.
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    return len(records)
