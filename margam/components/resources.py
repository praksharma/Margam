"""Allocation-free architecture/workload arithmetic and advisory memory snapshots.

No Torch or Qt imports at module load. These figures are not a peak-memory model.
"""
from .network_config import validate_network


def architecture_size(config):
    options=validate_network(config.network_type,config.network_options,config.layers)
    d=len(config.bounds[0]);out=len(config.field_names);layers=config.layers;kind=config.network_type
    width=layers[0];depth=len(layers);parameters=0;buffers=2*d;expanded=d
    def linear(a,b,bias=True):
        nonlocal parameters
        parameters+=a*b+(b if bias else 0)
        if options.get('weight_factorization',False):parameters+=b
    def trunk(inputs):
        sizes=[inputs,*layers]
        for a,b in zip(sizes,sizes[1:]):linear(a,b)
    if kind in ('fourier','modified_fourier','highway_fourier'):
        encoding=options['encoding'];n=options['features'] if encoding=='gaussian' else len(options['frequencies'])**d if encoding=='full' else d*len(options['frequencies'])
        expanded=d+2*n;buffers+=d*n
    elif kind=='fourier_mlp':expanded=d+2*d*options['modes'];buffers+=options['modes']
    if kind in ('fcnn','siren','fourier','fourier_mlp'):
        trunk(expanded);linear(layers[-1],out)
    elif kind=='modified_fourier':
        linear(expanded,width);linear(expanded,width);trunk(expanded);linear(width,out)
    elif kind=='highway_fourier':
        linear(expanded if options['transform_fourier'] else d,width)
        linear(expanded if options['project_fourier'] else d,width)
        trunk(expanded);linear(width,out)
    elif kind=='multiscale_fourier':
        n=options['features'];branches=len(options['scales']);expanded=d+2*n;buffers+=branches*d*n
        trunk(expanded);linear(layers[-1]*branches,out)
    elif kind=='affine_fourier':
        from .affine_inputs import affine_map
        _,offset=affine_map(options['input_transform'],config.coordinate_names)
        buffers+=len(offset)*(d+1);expanded=options['features']
        linear(len(offset),expanded);trunk(expanded);linear(layers[-1],out)
    elif kind=='dgm':
        linear(d,width)
        for _ in range(depth-1):
            for _ in range(4):linear(d,width);linear(width,width,False)
        linear(width,out)
    elif kind in ('fourier_filter','gabor_filter'):
        expanded=width
        parameters+=(depth+1)*(d*width+width)*(2 if kind=='gabor_filter' else 1)
        for _ in range(depth):linear(width,width)
        linear(width,out)
    else:raise ValueError('No resource arithmetic for this architecture.')
    if config.method=='inverse_burgers':parameters+=2
    return dict(parameters=parameters,buffer_elements=buffers,expanded_inputs=expanded)


def memory_snapshot(device):
    """Read host/device availability; no allocation, sampling or capacity probe.

    CUDA's query may initialize its context. Host figures do not include container
    or scheduler quotas and available memory is never a reservation.
    """
    import os
    info=dict(host_available=None,host_total=None,gpu_available=None,gpu_total=None,dtype='unknown',scalar_bytes=None)
    try:
        with open('/proc/meminfo') as stream:
            lines=stream.read(65536).splitlines()
        values={row.split(':',1)[0]:int(row.split()[1])*1024 for row in lines if row.startswith(('MemTotal:','MemAvailable:'))}
        info.update(host_available=values.get('MemAvailable'),host_total=values.get('MemTotal'))
    except (OSError,ValueError,IndexError):
        try:info.update(host_available=os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE'),host_total=os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE'))
        except (OSError,ValueError,AttributeError):pass
    try:
        import torch
        dtype=str(torch.get_default_dtype());info.update(dtype=dtype,scalar_bytes={'torch.float16':2,'torch.bfloat16':2,'torch.float32':4,'torch.float64':8}.get(dtype))
        if str(device).startswith('cuda') and torch.cuda.is_available():
            free,total=torch.cuda.mem_get_info(device);info.update(gpu_available=free,gpu_total=total)
    except (ImportError,RuntimeError,ValueError,AssertionError):pass
    for key in ('host_available','host_total','gpu_available','gpu_total'):
        if info[key] is not None and info[key]<0:info[key]=None
    return info


def resource_report(config,counts,requests,*,memory,monitor_limits=(),output_names=None):
    size=architecture_size(config);rows=counts['pde'];adaptive=config.adaptive_sampling
    maximum=adaptive['maximum'] if adaptive.get('method')=='rar' else rows
    outputs=[]
    for request in requests:
        if request.kind=='monitor':continue
        supplied=request.point_data is not None
        outputs.append(dict(identifier=request.identifier,name=(output_names or {}).get(request.identifier,request.kind),rows=len(request.point_data.points) if supplied else request.resolution**len(config.bounds[0]),upper_bound=not supplied))
    scalar=memory.get('scalar_bytes')
    weights=(size['parameters']+size['buffer_elements'])*scalar if scalar else None
    adam=(4*size['parameters']+size['buffer_elements'])*scalar if scalar else None
    activations=maximum*sum(config.layers)*scalar if scalar else None
    encoded=maximum*size['expanded_inputs']*scalar if scalar else None
    available=memory.get('gpu_available') if str(config.device).startswith('cuda') else memory.get('host_available')
    warnings=[]
    for label,value in (('Weight/buffer storage',weights),('Adam/gradient reference subtotal',adam),('PDE hidden-output scale',activations),('PDE encoded-input scale',encoded)):
        if available is not None and available>=0 and value is not None and value>available/2:
            warnings.append(label+' exceeds half of the currently reported available target memory. Review the workload; this is not a fit prediction.')
    return dict(**size,memory=memory,weights_bytes=weights,adam_reference_bytes=adam,hidden_output_scale_bytes=activations,encoded_input_scale_bytes=encoded,
        initial_pde_rows=rows,maximum_pde_rows=maximum,named_set_rows=sum(counts['sets'].values()),
        candidate_rows=adaptive.get('candidates',0),scoring_chunk=min(adaptive.get('candidates',0),adaptive.get('chunk',0)),
        outputs=outputs,output_rows=sum(row['rows'] for row in outputs),monitor_record_budget=sum(monitor_limits),warnings=warnings)


def byte_text(value):
    if value is None:return 'unavailable'
    for unit in ('B','KiB','MiB','GiB','TiB'):
        if value<1024 or unit=='TiB':return f'{value:,.2f} {unit}'
        value/=1024


def report_text(config,report):
    m=report['memory'];lines=['','Memory and workload (advisory)',
        f"Trainable network parameters: {report['parameters']:,} (including inverse coefficients where present)",
        f"Network buffer elements: {report['buffer_elements']:,}; encoded input width: {report['expanded_inputs']:,}",
        f"Scalar dtype: {m['dtype']}",f"Weight/buffer storage: {byte_text(report['weights_bytes'])}",
        f"Adam/gradient reference subtotal: {byte_text(report['adam_reference_bytes'])}",
        'Reference subtotal assumes full-size gradients and two Adam moments for every parameter; it is NOT total training memory.',
        f"PDE rows: {report['initial_pde_rows']:,} initially, {report['maximum_pde_rows']:,} at the configured maximum (including connected PDE boundary anchors)",
        f"Named geometric set rows (sum): {report['named_set_rows']:,}",
        f"PDE hidden-output scale: {byte_text(report['hidden_output_scale_bytes'])} (maximum PDE rows × sum of hidden widths × scalar bytes; NOT a peak estimate)",
        f"PDE encoded-input scale: {byte_text(report['encoded_input_scale_bytes'])} (one input/feature array; NOT a peak estimate, multi-branch copies excluded)",
        f"Adaptive candidates: {report['candidate_rows']:,}; scoring chunk: {report['scoring_chunk']:,}",
        f"Requested prediction rows across output blocks: up to {report['output_rows']:,}; repeated requests are counted separately",
        f"Monitor retained-record budgets (sum): {report['monitor_record_budget']:,}; trainer histories also retain update records",
        f"Host RAM available / total: {byte_text(m['host_available'])} / {byte_text(m['host_total'])} (host snapshot; container/scheduler limits excluded)"]
    if str(config.device).startswith('cuda'):
        lines.append(f"GPU driver-free / total: {byte_text(m['gpu_available'])} / {byte_text(m['gpu_total'])} (snapshot, not reserved)")
        lines.append('PyTorch may also reuse its own cached memory; driver-free memory is not a complete allocation budget.')
    for row in report['outputs']:lines.append(f"  Output {row['name']}: {'up to ' if row['upper_bound'] else ''}{row['rows']:,} prediction rows")
    lines+=['Derivative/autograd graphs, Fourier feature activations, optimizer temporaries, BC graphs, data copies and retained method states are not estimated.']
    if config.method=='scale':lines.append('Scale-PINN retains additional frozen network iterates.')
    if config.method=='expert':lines.append('Expert balancing/NTK workspace and retained time-window teachers add memory beyond the subtotal.')
    if config.method=='self_adaptive':lines.append('SA-PINN point masks and their optimizer state add memory beyond the network subtotal.')
    if config.method=='gpinn':lines.append('gPINN differentiates residuals again; derivative graphs can dominate network storage.')
    lines+=['WARNING: '+warning for warning in report['warnings']]
    lines+=['This review does not reserve memory, run a probe or guarantee that training fits. Resource advice never changes samples, architecture or precision automatically.']
    return '\n'.join(lines)


def worker_failure(error,stage,device):
    """Preserve ordinary diagnostics; explain recognised allocator failures locally."""
    text=str(error);lower=text.lower();name=type(error).__name__;module=type(error).__module__
    host=isinstance(error,MemoryError) or isinstance(error,RuntimeError) and 'defaultcpuallocator' in lower and any(s in lower for s in ("can't allocate",'not enough memory','out of memory'))
    cuda=isinstance(error,RuntimeError) and lower.startswith(('cuda out of memory','hip out of memory'))
    typed=name=='OutOfMemoryError' and module.startswith('torch')
    if not (host or cuda or typed):return f'{name}: {text}'
    location='host RAM' if host or not cuda and not str(device).startswith('cuda') else 'GPU memory'
    advice='For a new run, reduce network width/depth or Fourier feature counts.' if stage=='model construction' else 'For a new run, reduce active PDE/BC rows or network size; review RAR maximum points and residual scoring chunk size.'
    if stage=='output evaluation':advice='Reduce requested output resolution/rows or the number of output requests, then evaluate again. Training samples are separate from output points.'
    if stage=='checkpoint restore':advice='Continuation needs the saved architecture, data and objective. Free memory or choose a suitable device; changing the experiment requires a new run.'
    return f'Memory allocation failed during {stage} on {device} ({location}).\n{advice}\nCheck other memory-heavy jobs and current RAM/VRAM availability. Samples, architecture and precision were not changed automatically. Previously saved run files are unchanged.\nBackend diagnostic: {name}: {text}'
