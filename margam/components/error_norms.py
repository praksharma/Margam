"""Discrete, unweighted scalar/vector diagnostics on already aligned result rows."""
import math
import numpy as np

NORMS={
    'relative_l2':'Relative L2',
    'relative_max':'Relative max magnitude',
    'rms':'RMS magnitude',
    'mean':'Mean magnitude',
    'max':'Max magnitude',
    'l2':'Discrete L2',
}
FORMULAS={
    'relative_l2':'sqrt(sum_i ||prediction_i-reference_i||²) / sqrt(sum_i ||reference_i||²)',
    'relative_max':'max_i ||prediction_i-reference_i|| / max_i ||reference_i||',
    'rms':'sqrt(mean_i ||prediction_i-reference_i||²)',
    'mean':'mean_i ||prediction_i-reference_i||',
    'max':'max_i ||prediction_i-reference_i||',
    'l2':'sqrt(sum_i ||prediction_i-reference_i||²)',
}
ZERO_POLICIES={'undefined':'Undefined for a zero reference',
               'zero_if_exact':'Zero only when prediction and reference are exactly zero'}


def array(value):
    if hasattr(value,'detach'):
        if value.is_complex():raise ValueError('Error norms require real values.')
        value=value.detach().cpu()
        try:value=value.numpy()
        except TypeError:value=value.double().numpy()
    if np.ma.isMaskedArray(value) and np.any(value.mask):raise ValueError('Filter masked rows explicitly before computing error norms.')
    try:
        result=np.asarray(value)
        if result.dtype.kind not in 'iuf':raise ValueError('Error norms require real numeric arrays.')
        with np.errstate(over='ignore',invalid='ignore'):result=result.astype(np.float64,copy=False)
    except (TypeError,ValueError,OverflowError) as error:
        raise ValueError('Error norms require finite real float64-compatible arrays.') from error
    if result.ndim==1:result=result[:,None]
    if result.ndim!=2 or not result.size or not np.isfinite(result).all():
        raise ValueError('Error norms require finite nonempty (N,C) arrays, or (N,) scalar fields.')
    return result


def scaled_error(prediction,reference):
    with np.errstate(over='ignore',invalid='ignore'):difference=prediction-reference
    if np.isfinite(difference).all():
        scale=float(np.max(np.abs(difference)))
        return scale,difference/scale if scale else np.zeros_like(difference)
    scale=max(float(np.max(np.abs(prediction))),float(np.max(np.abs(reference))))
    return scale,prediction/scale-reference/scale


def statistic(values,norm):
    squared=np.sum(values*values,axis=1)
    if norm in ('l2','relative_l2'):return math.sqrt(float(np.sum(squared)))
    if norm=='rms':return math.sqrt(float(np.mean(squared)))
    magnitudes=np.sqrt(squared)
    return float(np.mean(magnitudes)) if norm=='mean' else float(np.max(magnitudes))


def scaled_value(scale,factor,denominator_scale=1.,denominator_factor=1.):
    if not scale or not factor:return 0.,'defined'
    numerator,exponent=math.frexp(scale);denominator,other_exponent=math.frexp(denominator_scale)
    try:value=math.ldexp((numerator/denominator)*(factor/denominator_factor),exponent-other_exponent)
    except OverflowError:return None,'overflow'
    if not math.isfinite(value):return None,'overflow'
    if value==0:return None,'underflow'
    return value,'defined'


def error_norms(prediction,reference,*,fields=None,selected=None,norm='relative_l2',zero_reference='undefined'):
    """Return one finite/None diagnostic with explicit field/reduction metadata.

    Vector errors use each row's Euclidean magnitude. Relative L2 divides global
    flattened L2 norms; relative max divides the maxima of pointwise magnitudes.
    There is no quadrature, interpolation, field normalization or gradient graph.
    """
    if not isinstance(norm,str) or norm not in NORMS:raise ValueError('Choose a supported error norm.')
    if not isinstance(zero_reference,str) or zero_reference not in ZERO_POLICIES:raise ValueError('Choose a supported zero-reference policy.')
    prediction,reference=array(prediction),array(reference)
    if prediction.shape!=reference.shape:raise ValueError('Prediction and reference shapes must match exactly; broadcasting is not supported.')
    count=prediction.shape[1]
    if fields is None:fields=['u'] if count==1 else ['field_'+str(i) for i in range(count)]
    if not isinstance(fields,(list,tuple)) or len(fields)!=count or any(not isinstance(name,str) or not name.strip() or len(name)>256 for name in fields) or len(set(fields))!=count:
        raise ValueError('Provide one unique field name per component.')
    selected=list(fields) if selected is None else selected
    if not isinstance(selected,(list,tuple)) or not selected or any(not isinstance(name,str) or name not in fields for name in selected) or len(set(selected))!=len(selected):
        raise ValueError('Select one or more unique existing fields.')
    indices=[fields.index(name) for name in selected]
    prediction,reference=prediction[:,indices],reference[:,indices]
    scale,error=scaled_error(prediction,reference);factor=statistic(error,norm)
    ref_scale=float(np.max(np.abs(reference)));relative=norm.startswith('relative_')
    if relative and ref_scale==0:
        value,status=(0.,'defined_zero_reference') if zero_reference=='zero_if_exact' and scale==0 else (None,'undefined_zero_reference')
    elif relative:
        value,status=scaled_value(scale,factor,ref_scale,statistic(reference/ref_scale,norm))
    else:value,status=scaled_value(scale,factor)
    return dict(norm=norm,value=value,status=status,fields=list(selected),rows=len(prediction),
        components=len(selected),reduction='pointwise_euclidean',relative=relative,
        zero_reference_policy=zero_reference,reference_zero=ref_scale==0)
