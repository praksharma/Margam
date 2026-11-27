"""Finite Basis PINN (FBPINN) configuration and validation, shared by scripts and UI.

No Torch or Qt imports at module load.
"""
import ast
import math

WINDOW_TYPES = ('cosine_sq', 'smooth_sigmoid', 'cosine')
SCHEDULES = ('simultaneous', 'progressive')
SCHEDULE_AXES = ('auto', 'x', 'y', 'z', 't')
MAX_SUBDOMAINS_PER_AXIS = 32
MAX_TOTAL_SUBDOMAINS = 64
SUPPORTED_SHAPES = (
    'Interval',
    'Rectangle',
    'SpaceTime1D',
    'SpaceTime2D',
    'Box3D',
    'SpaceTime3D',
    'SphereObstacle3D',
)


def fbpinn_defaults():
    return dict(
        subdomains_x=4,
        subdomains_y=1,
        subdomains_z=1,
        subdomains_t=1,
        overlap=1.5,
        window='cosine_sq',
        window_sigma=0.05,
        multilevel=False,
        schedule='simultaneous',
        schedule_axis='auto',
        active_steps=100,
        freeze_completed=True,
    )


def validate_fbpinn_settings(settings):
    if not isinstance(settings, dict) or set(settings) != set(fbpinn_defaults()):
        raise ValueError('Incomplete FBPINN settings.')
    for axis in ('x', 'y', 'z', 't'):
        key = f'subdomains_{axis}'
        value = settings[key]
        if type(value) is not int or not 1 <= value <= MAX_SUBDOMAINS_PER_AXIS:
            raise ValueError(f'FBPINN {key} must be an integer between 1 and {MAX_SUBDOMAINS_PER_AXIS}.')
    for key, lo, hi in (('overlap', 1.05, 3.0), ('window_sigma', 0.01, 0.5)):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'FBPINN {key} must be a finite number between {lo} and {hi}.')
    if settings['window'] not in WINDOW_TYPES:
        raise ValueError('FBPINN window must be cosine_sq, smooth_sigmoid, or cosine.')
    if type(settings['multilevel']) is not bool:
        raise ValueError('FBPINN multilevel option must be boolean.')
    if settings['schedule'] not in SCHEDULES:
        raise ValueError('FBPINN schedule must be simultaneous or progressive.')
    if settings['schedule_axis'] not in SCHEDULE_AXES:
        raise ValueError('FBPINN schedule_axis must be auto, x, y, z, or t.')
    steps = settings['active_steps']
    if type(steps) is not int or not 1 <= steps <= 100000:
        raise ValueError('FBPINN active_steps must be an integer between 1 and 100000.')
    if type(settings['freeze_completed']) is not bool:
        raise ValueError('FBPINN freeze_completed option must be boolean.')


def spatial_time_axes(config):
    names = tuple(config.coordinate_names)
    if hasattr(config, 'spatial_time_bounds'):
        return names[:len(config.spatial_time_bounds[0])]
    return names


def active_subdomain_counts(config):
    settings = config.fbpinn_settings
    return tuple(settings[f'subdomains_{name}'] for name in spatial_time_axes(config))


def total_subdomains(config):
    total = 1
    for count in active_subdomain_counts(config):
        total *= count
    return total


def total_networks(config):
    return total_subdomains(config) + (1 if config.fbpinn_settings['multilevel'] else 0)


def resolve_schedule_axis(config):
    axes = spatial_time_axes(config)
    counts = active_subdomain_counts(config)
    choice = config.fbpinn_settings['schedule_axis']
    if choice == 'auto':
        if 't' in axes:
            idx = axes.index('t')
            if counts[idx] >= 2:
                return idx, 't'
        for idx, (name, count) in enumerate(zip(axes, counts)):
            if count >= 2:
                return idx, name
        raise ValueError('Progressive FBPINN marching requires at least 2 subdomains along an active domain axis.')
    if choice not in axes:
        raise ValueError(f'FBPINN schedule axis {choice!r} is not present in this domain ({", ".join(axes)}).')
    idx = axes.index(choice)
    if counts[idx] < 2:
        raise ValueError(f'Progressive FBPINN marching along {choice} requires at least 2 subdomains on that axis.')
    return idx, choice


def _expression_max_derivative_order(text):
    if not text or not isinstance(text, str):
        return 0
    try:
        tree = ast.parse(text, mode='eval')
    except SyntaxError:
        return 0
    max_order = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == 'laplace':
                max_order = max(max_order, 2)
            elif node.func.id == 'dn':
                max_order = max(max_order, 1)
            elif node.func.id == 'D':
                order = 1
                if len(node.args) >= 3 and isinstance(node.args[2], ast.Constant) and type(node.args[2].value) is int:
                    order = node.args[2].value
                max_order = max(max_order, order)
    return max_order


def _max_problem_derivative_order(config):
    if config.pde_preset in ('kdv_1d', 'ks_1d'):
        return 4 if config.pde_preset == 'ks_1d' else 3
    order = _expression_max_derivative_order(config.pde_expression)
    if config.physics == 'incompressible_2d':
        for expr in config.flow_residuals:
            order = max(order, _expression_max_derivative_order(expr))
        for side_specs in config.flow_boundary.values():
            for spec in side_specs.values():
                if spec.get('type') == 'Custom':
                    order = max(order, _expression_max_derivative_order(spec.get('operator', '')))
    else:
        for expr in config.bc_operators.values():
            order = max(order, _expression_max_derivative_order(expr))
    return order


def validate_fbpinn_problem(config):
    validate_fbpinn_settings(config.fbpinn_settings)
    if config.shape not in SUPPORTED_SHAPES:
        raise ValueError('FBPINN supports built-in interval, rectangle, 1D/2D/3D space–time, 3D box and sphere-obstacle domains.')
    if config.physics not in ('scalar', 'incompressible_2d') or config.is_paper_case or config.pde_preset == 'kdv_1d':
        raise ValueError('FBPINN supports scalar PDEs and steady 2D rectangular incompressible flow; paper cases and KdV are not supported.')
    if config.physics == 'incompressible_2d' and config.shape != 'Rectangle':
        raise ValueError('FBPINN 2D flow requires a Rectangle domain.')
    if config.loss_formulation != 'mean' or config.adaptive_sampling or config.interior_data or config.boundary_data:
        raise ValueError('FBPINN currently requires mean MSE loss and fixed built-in points; imported or adaptive samples are not supported.')
    count = total_subdomains(config)
    if count > MAX_TOTAL_SUBDOMAINS:
        raise ValueError(f'FBPINN supports at most {MAX_TOTAL_SUBDOMAINS} fine subdomains in total (requested {count}).')
    settings = config.fbpinn_settings
    if settings['multilevel']:
        if count < 2:
            raise ValueError('Multilevel FBPINN requires at least 2 fine subdomains.')
        if settings['schedule'] == 'progressive':
            raise ValueError('Multilevel coarse-to-fine FBPINN currently requires simultaneous training, not progressive marching.')
    if settings['schedule'] == 'progressive':
        resolve_schedule_axis(config)
    max_order = _max_problem_derivative_order(config)
    if max_order >= 3 and settings['window'] != 'smooth_sigmoid':
        raise ValueError(f'FBPINN window {settings["window"]!r} is not C-infinity smooth; use smooth_sigmoid for 3rd- or higher-order PDE derivatives.')
