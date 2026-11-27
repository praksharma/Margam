"""Concrete Finite Basis PINN (FBPINN) example presets with explicit references."""
import math
from .wave import INITIAL, REFERENCE as WAVE_REFERENCE

MULTISCALE_1D_REF = 'sin(2*pi*x)+0.5*sin(16*pi*x)'
MULTISCALE_1D_FORCING = f'{4 * math.pi ** 2!r}*sin({2 * math.pi!r}*x)+{128 * math.pi ** 2!r}*sin({16 * math.pi!r}*x)'

MULTISCALE_2D_REF = 'sin(2*pi*x)*sin(2*pi*y)+0.5*sin(8*pi*x)*sin(8*pi*y)'
MULTISCALE_2D_FORCING = (
    f'{8 * math.pi ** 2!r}*sin({2 * math.pi!r}*x)*sin({2 * math.pi!r}*y)'
    f'+{64 * math.pi ** 2!r}*sin({8 * math.pi!r}*x)*sin({8 * math.pi!r}*y)'
)

PARAMETRIC_1D_REF = 'sin(2*pi*x)+mu*sin(8*pi*x)'
POISSON_3D_REF = 'sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z)'
POISSON_3D_FORCING = f'{12 * math.pi ** 2!r}*sin({2 * math.pi!r}*x)*sin({2 * math.pi!r}*y)*sin({2 * math.pi!r}*z)'

FBPINN_EXAMPLES = {
    'fbpinn_multiscale_1d': (
        'Multiscale Poisson · 1D · multilevel FBPINN',
        'Two-frequency solution sin(2πx) + 0.5 sin(16πx) on [0,1]; 12 overlapping C2 subdomains plus a global coarse network; analytical reference.',
        MULTISCALE_1D_REF,
    ),
    'fbpinn_multiscale_2d': (
        'Multiscale Poisson · 2D · multilevel FBPINN',
        'Two-frequency solution on [0,1]² with 4×4 overlapping C2 subdomains and coarse-to-fine multilevel hierarchy; analytical reference.',
        MULTISCALE_2D_REF,
    ),
    'fbpinn_wave_progressive': (
        'Standing wave · 1D + time · progressive FBPINN',
        'Two-mode standing wave (c=2) with progressive active-subdomain marching along physical time t; analytical reference.',
        WAVE_REFERENCE,
    ),
    'fbpinn_parametric_1d': (
        'Parametric multiscale Poisson · 1D · u(x; μ)',
        'Continuous high-frequency amplitude μ ∈ [0.2, 1.0] across 6 overlapping subdomains; scrub μ live in Solution Explorer; analytical reference.',
        PARAMETRIC_1D_REF,
    ),
    'fbpinn_poisson_3d': (
        'Multiscale Poisson · 3D · 2×2×2 FBPINN',
        'Manufactured 3D Poisson sin(2πx)sin(2πy)sin(2πz) on [0,1]³ using 2×2×2 overlapping subdomains and multilevel coarse network.',
        POISSON_3D_REF,
    ),
}


def fbpinn_example_config(key):
    from .poisson import PoissonConfig
    if key not in FBPINN_EXAMPLES:
        raise ValueError('Unknown FBPINN example.')
    if key == 'fbpinn_multiscale_1d':
        c = PoissonConfig(
            shape='Interval',
            points=320,
            boundary_points=16,
            sampling='grid',
            seed=7,
            network_seed=7,
            pde_preset='poisson',
            source=0.0,
            pde_forcing=MULTISCALE_1D_FORCING,
            layers=(16, 16),
            iterations=2000,
            lr=0.01,
            pde_weight=1e-5,
            boundary_weight=1.0,
            method='fbpinn',
        )
        c.fbpinn_settings.update(subdomains_x=12, overlap=1.5, window='cosine_sq', multilevel=True, schedule='simultaneous')
        return c
    if key == 'fbpinn_multiscale_2d':
        c = PoissonConfig(
            shape='Rectangle',
            points=1024,
            boundary_points=64,
            sampling='hammersley',
            seed=7,
            network_seed=7,
            pde_preset='poisson',
            source=0.0,
            pde_forcing=MULTISCALE_2D_FORCING,
            layers=(16, 16),
            iterations=2000,
            lr=0.01,
            pde_weight=1e-4,
            boundary_weight=1.0,
            method='fbpinn',
        )
        c.fbpinn_settings.update(subdomains_x=4, subdomains_y=4, overlap=1.5, window='cosine_sq', multilevel=True, schedule='simultaneous')
        return c
    if key == 'fbpinn_wave_progressive':
        c = PoissonConfig(
            shape='SpaceTime1D',
            points=512,
            boundary_points=48,
            sampling='hammersley',
            seed=7,
            network_seed=7,
            pde_preset='wave_1d',
            source=2.0,
            layers=(16, 16),
            iterations=1600,
            lr=0.005,
            pde_weight=1e-3,
            boundary_weight=3.0,
            method='fbpinn',
        )
        c.bc_targets = {'bottom': INITIAL}
        c.fbpinn_settings.update(
            subdomains_x=2,
            subdomains_t=4,
            overlap=1.5,
            window='cosine_sq',
            multilevel=False,
            schedule='progressive',
            schedule_axis='t',
            active_steps=400,
            freeze_completed=True,
        )
        return c
    if key == 'fbpinn_parametric_1d':
        c = PoissonConfig(
            shape='Interval',
            parametric_axes=({'name': 'mu', 'min': 0.2, 'max': 1.0, 'default': 0.5},),
            points=512,
            boundary_points=32,
            sampling='hammersley',
            seed=7,
            network_seed=7,
            pde_preset='',
            pde_expression='laplace(u) + 4*pi*pi*sin(2*pi*x) + 64*pi*pi*mu*sin(8*pi*x)',
            layers=(16, 16),
            iterations=1500,
            lr=0.005,
            pde_weight=1e-4,
            boundary_weight=1.0,
            method='fbpinn',
        )
        c.fbpinn_settings.update(subdomains_x=6, overlap=1.5, window='cosine_sq', multilevel=True, schedule='simultaneous')
        return c
    # fbpinn_poisson_3d
    c = PoissonConfig(
        shape='Box3D',
        points=512,
        boundary_points=36,
        sampling='hammersley',
        seed=7,
        network_seed=7,
        pde_preset='poisson',
        source=0.0,
        pde_forcing=POISSON_3D_FORCING,
        layers=(16, 16),
        iterations=1200,
        lr=0.005,
        pde_weight=1e-3,
        boundary_weight=1.0,
        method='fbpinn',
    )
    c.bc_types = dict.fromkeys(c.sides, 'Dirichlet')
    c.bc_values = dict.fromkeys(c.sides, 0.0)
    c.fbpinn_settings.update(subdomains_x=2, subdomains_y=2, subdomains_z=2, overlap=1.5, window='cosine_sq', multilevel=True, schedule='simultaneous')
    return c
