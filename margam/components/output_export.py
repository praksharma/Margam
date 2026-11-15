"""Shared explicit export dispatch, without dialogs or implicit output requests."""

def extension(data,kind):
    if kind=='export':return '.csv'
    if kind!='paraview':raise ValueError('Choose a CSV Export or ParaView Export block.')
    config=data['config']
    return '.pvd' if config.is_time_dependent else '.vtu' if config.has_obstacle or 'evaluation' in data else '.vti'


def export_data(path,data,kind):
    if kind=='export':
        from .export import export_csv
        return export_csv(path,data)
    if kind!='paraview':raise ValueError('Choose a CSV Export or ParaView Export block.')
    from .paraview_export import export_vti,export_spacetime_pvd
    if 'evaluation' in data:
        from .point_export import export_supplied
        return export_supplied(path,data)
    if data['config'].is_time_dependent:return export_spacetime_pvd(path,data)
    if data['config'].has_obstacle:
        from .point_export import export_vtu
        return export_vtu(path,data)
    return export_vti(path,data)
