"""Explanations of the concrete payloads carried by Margam arrows."""
PAYLOADS={
 'geometry':'Domain definition and sample provenance. This is not a mesh or a stream of tensor values.',
 'interior_points':'Coordinates supplied by an interior sampler or imported data. The destination chooses how they are used.',
 'boundary_points':'Named boundary/initial coordinate sets. Their labels do not assign boundary-condition physics.',
 'collocation_points':'Coordinates where the PDE residual is enforced. Interior or boundary sources are accepted; only connected sources contribute.',
 'residual':'PDE residual definition and its selected collocation points, passed to the trainer.',
 'constraints':'Boundary/initial operators, targets and named point sets, passed to the trainer separately from the PDE loss.',
 'network':'The neural-network architecture and initialization settings used by the trainer.',
 'pinn_run':'A PINN run handle. Outputs request retained losses or evaluated predictions; this is not a signal at every optimizer update.',
 'evaluation_points':'Prediction coordinates and optional point IDs, independent of training points. Row order is preserved.',
 'reference':'Explicit reference fields, expressions or mapped values. No analytical solution is assumed.',
 'comparison_data':'Predictions paired with explicitly aligned reference fields and errors for export.',
}


def port_text(port):
    return ('Output: ' if port.output else 'Input: ')+port.label+'\n'+PAYLOADS.get(port.data_type,port.data_type.replace('_',' '))+('\nDrag to a compatible input.' if port.output else '\nAccepts one connected source.')


def wire_text(wire):
    if wire.target is None:return 'Choose a compatible input to complete this connection.'
    source,target=wire.source.parentItem(),wire.target.parentItem()
    text=f'{source.title} → {target.title}\n{wire.source.label} → {wire.target.label}\n\n'
    if wire.target.data_type=='collocation_points' or wire.target.name=='boundary_points':
        text+='These coordinates contribute to the PDE loss. Boundary/initial rows are included only because this arrow exists. This does not create a boundary-condition operator.'
    elif target.kind=='bc':
        text+='These named point sets supply boundary/initial conditions. The BC block assigns operators, targets and coefficients. This arrow alone does not include them in the PDE loss.'
    else:text+=PAYLOADS.get(wire.target.data_type,wire.target.data_type.replace('_',' '))
    return text+'\n\nArrows describe problem dependencies, not physical-time signals or the PyTorch differentiation graph.'
