"""Data-only invariants for Expert window resource measurements."""
import math
import re


def validate_window_report(value):
    """Reject partial, inconsistent or misleading phase accounting without mutation."""
    try:_validate(value)
    except (KeyError,TypeError,IndexError) as error:raise ValueError('Incomplete Expert window probe report.') from error
    return value


def _validate(v):
    def require(condition,message):
        if not condition:raise ValueError('Invalid Expert window probe: '+message)
    def integer(value,minimum=0,maximum=100000):return type(value) is int and minimum<=value<=maximum
    def real(value):
        try:return type(value) in (int,float) and math.isfinite(value)
        except OverflowError:return False
    def memory(value):return value is None or type(value) is int and 0<=value<=2**63-1
    require(isinstance(v,dict) and v.get('format')=='margam-resource-probe' and type(v.get('version')) is int and v['version']==2,'format/version')
    n,steps,total=v['configured_windows'],v['updates_per_window'],v['completed_updates']
    require(integer(n,2,32) and integer(steps,1) and integer(total,1) and integer(v['requested_updates'],1) and total==v['requested_updates'],'schedule/work count')
    transfers=min(n-1,(total-1)//steps)
    require(all(integer(v[key]) for key in ('observed_transfers','observed_windows','teacher_count')) and v['observed_transfers']==transfers and v['observed_windows']==transfers+1 and v['teacher_count']==transfers,'observed window/teacher counts')
    require(type(v['next_transfer_pending']) is bool and v['next_transfer_pending']==(transfers<n-1 and total%steps==0),'pending transfer')
    require(v['internal_window_snapshots_included'] is True and v['measurement_overhead_included'] is True and v['checkpoint_retained'] is False and v['capacity_prediction'] is False,'measurement scope flags')
    require(integer(v['cached_teacher_copies'],0,transfers),'teacher cache count')
    phases=v['phases'];require(isinstance(phases,list) and len(phases)==2+2*transfers and len(phases)<=64,'phase count')
    require(isinstance(v['device'],str) and re.fullmatch(r'cpu|cuda(?::[0-9]+)?',v['device']) is not None,'device')
    cuda=v['device'].startswith('cuda')
    required={'kind','window','start_update','end_update','time_interval','duration_seconds','process_peak_rss_bytes','cuda_start_allocated_bytes','cuda_start_reserved_bytes','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes','cuda_end_allocated_bytes','cuda_end_reserved_bytes','completed_window_snapshots','teacher_count','cached_teacher_copies','pde_rows'}
    end=0;rss=0;intervals={}
    for i,p in enumerate(phases):
        require(isinstance(p,dict) and set(p)==required,'phase fields')
        kind='construction' if i==0 else 'window_training' if i%2 else 'teacher_transfer'
        window=1 if i==0 else 1+i//2
        require(p['kind']==kind and integer(p['window'],1,32) and p['window']==window,'phase order/window')
        require(integer(p['start_update']) and integer(p['end_update']) and p['start_update']==end,'phase continuity')
        if kind=='construction':require(p['end_update']==0,'construction work')
        elif kind=='teacher_transfer':require(p['start_update']==p['end_update']==(window-1)*steps,'handoff work')
        else:
            expected=min(window*steps,total) if window<n else total
            require(p['start_update']==(window-1)*steps and p['end_update']==expected and expected>p['start_update'],'window update range')
        end=p['end_update'];require(real(p['duration_seconds']) and p['duration_seconds']>=0,'duration')
        interval=p['time_interval'];require(isinstance(interval,list) and len(interval)==2 and all(real(x) for x in interval) and interval[0]<interval[1],'physical time interval')
        if window in intervals:require(interval==intervals[window],'window interval changed')
        else:
            if window>1:require(interval[0]==intervals[window-1][1],'window time gap/overlap')
            intervals[window]=interval
        require(integer(p['completed_window_snapshots'],0,31) and integer(p['teacher_count'],0,31) and p['completed_window_snapshots']==p['teacher_count']==window-1 and integer(p['cached_teacher_copies'],0,window-1),'phase teacher/snapshot counts')
        require(integer(p['pde_rows'],1,10000000),'PDE row count')
        r=p['process_peak_rss_bytes'];require(memory(r) and (r is None or r>=rss),'cumulative RSS')
        if r is not None:rss=r
        for counter in ('allocated','reserved'):
            start,peak,stop=(p[f'cuda_{name}_{counter}_bytes'] for name in ('start','peak','end'))
            require(all(memory(x) for x in (start,peak,stop)),'allocator byte values')
            require((all(x is not None for x in (start,peak,stop)) and peak>=max(start,stop)) if cuda else all(x is None for x in (start,peak,stop)),'allocator peak/baseline scope')
        if cuda:require(p['cuda_peak_reserved_bytes']>=p['cuda_peak_allocated_bytes'],'reserved/allocated peaks')
    require(integer(v['initial_pde_rows'],1,10000000) and integer(v['final_pde_rows'],1,10000000),'row count types')
    require(v['initial_pde_rows']==phases[0]['pde_rows'] and v['final_pde_rows']==phases[-1]['pde_rows'],'initial/final row counts')
    require(end==total and v['cached_teacher_copies']==phases[-1]['cached_teacher_copies'],'final phase state')
    require(memory(v['process_peak_rss_bytes']) and (v['process_peak_rss_bytes'] is None or v['process_peak_rss_bytes']>=rss),'overall RSS')
    for counter in ('allocated','reserved'):
        peak=v[f'cuda_peak_{counter}_bytes']
        require((type(peak) is int and memory(peak) and peak==max(p[f'cuda_peak_{counter}_bytes'] for p in phases)) if cuda else peak is None,'overall CUDA maximum')
