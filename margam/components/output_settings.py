"""Optional-output settings shared by desktop and headless execution."""
OUTPUT_KINDS = {'monitor', 'solution', 'comparison', 'export', 'paraview'}


def output_enabled(node):
    value=node.settings.get('enabled',True)
    if type(value) is not bool:raise ValueError('Output enabled must be boolean.')
    return value
