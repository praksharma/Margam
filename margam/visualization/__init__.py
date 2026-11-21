"""Unified 1D/2D/3D/transient/parametric solution visualization data layer."""
from .colormaps import COLORMAPS, COLORMAP_LABELS, colormap_lut, map_scalars_to_rgba
from .dataset import (
    DERIVED_SPEED,
    DERIVED_VORTICITY_2D,
    DERIVED_VORTICITY_3D,
    DISPLAY_MODES,
    SolutionDataset,
)
from .loaders import (
    StudyCollection,
    from_arrays,
    from_file,
    from_output_result,
    from_run_archive,
)


def explore(source=None, *, block=True, **kwargs):
    """Launch the dedicated Solution Explorer window for a dataset, file, or model."""
    from ..ui.solution_explorer import launch_solution_explorer

    return launch_solution_explorer(source, block=block, **kwargs)


__all__ = [
    "COLORMAPS",
    "COLORMAP_LABELS",
    "colormap_lut",
    "map_scalars_to_rgba",
    "DERIVED_SPEED",
    "DERIVED_VORTICITY_2D",
    "DERIVED_VORTICITY_3D",
    "DISPLAY_MODES",
    "SolutionDataset",
    "StudyCollection",
    "from_arrays",
    "from_file",
    "from_output_result",
    "from_run_archive",
    "explore",
]
