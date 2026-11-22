# Solution Explorer (2D, 3D, and parametric)

Margam includes a dedicated **Solution Explorer** window for interactive post-processing of 1D, 2D, 3D, transient, parameter-study, and continuous parametric PINN solutions. It stays closed until you open it explicitly and shares your current Light or Dark desktop theme.

## Open Solution Explorer

You can launch Solution Explorer in four ways:

1. **From the main desktop window**: choose **View → Open Solution Explorer…** (`Ctrl+Shift+V`), click **Solution Explorer** on the toolbar, or right-click any evaluated **Solution Plot** or **Comparison** block and pick **Open in Solution Explorer…**.
2. **From a completed Parameter Study**: open **Run → Inspect saved study results…** and click **Explore in Solution Explorer…**.
3. **Directly on saved data**: use **Open file / run…** inside Solution Explorer to load `.margam-run.pt` archives, `.npz`, `.csv`, or `.nc` datasets, or **Open study…** to load a saved study directory.
4. **From Python / `margam.research`**: call `margam.visualization.explore(...)` on a dataset, file path, study folder, or live model/callable.

## 1D and 2D field visualization

For 1D domains, Solution Explorer displays prediction, reference, and error profiles in a synchronized curve view.

For 2D domains (regular grids, obstacle domains such as cylinder wakes, and unstructured or prepared-domain point clouds via Delaunay projection), the **2D Field** viewport provides:

- **Field and derived quantities**: choose any predicted scalar component (`u`, `v`, `w`, `p`), plus automatic **Velocity magnitude |u|** and **Vorticity ω_z** when vector velocity components are present.
- **Display modes**: **Prediction**, **Reference**, **Signed error (pred − ref)**, **Absolute error |pred − ref|**, and **Relative error |pred − ref| / |ref|**.
- **3-Panel Comparison**: switch **Mode** to **3-Panel Comparison** to inspect Prediction, Reference, and Error side by side with synchronized pan/zoom.
- **Colormaps and value limits**: 8 scientific colormaps (`Viridis`, `Plasma`, `Inferno`, `Magma`, `Cividis`, `Turbo`, `Coolwarm`, `RdBu`), automatic or manual `vmin` / `vmax` limits, and **Lock color scale** (recommended when scrubbing time frames or parameter axes).
- **Contours and vector overlays**: configurable contour levels, strided velocity quiver arrows, and RK4 streamlines integrated over the 2D velocity field.
- **Interactive probing**: hover anywhere over the 2D field for interpolated coordinates and values; click **Locate max |error|** to highlight the peak error location; enable **Drag line-cut probe** to sample a 1D cross-section into the bottom **Probe & diagnostics** dock and export it via **Export probe CSV…**.
- **Snapshot export**: click **Snapshot PNG…** to save the current viewport and colorbar as a PNG image.

## 3D viewport and native 3D domains

Margam supports three built-in 3D domains directly on the canvas (**Domain** block) and in `PoissonConfig`:

- **`Box3D`**: steady 3D rectangular prism `[x_min, x_max] × [y_min, y_max] × [z_min, z_max]` with six named faces (`left`, `right`, `bottom`, `top`, `front`, `back`).
- **`SpaceTime3D`**: transient 3D box `[x_min, x_max] × [y_min, y_max] × [z_min, z_max] × [t_min, t_max]` with six spatial faces plus an `initial` constraint at `t = t_min`.
- **`SphereObstacle3D`**: 3D box excluding a solid sphere `(sphere_x, sphere_y, sphere_z, sphere_radius)` with an explicit `sphere` boundary set and analytical outward unit normals.

Try the built-in examples under **File → Examples → 3D examples**:

- **3D Poisson — unit cube** (`poisson_3d`): manufactured 3D problem with exact analytical reference `sin(pi*x)*sin(pi*y)*sin(pi*z)` and `HardDirichletPINN`.
- **3D heat conduction — spherical cavity** (`sphere_heat_3d`): 3D Laplace equation around a hot sphere (`u = 1`) inside cold walls (`u = 0`).

In Solution Explorer's **3D Viewport** (selected automatically for 3D datasets, or via **Mode → 3D Viewport** to view a 2D field as a warped heightfield surface):

- **Camera controls**: left-drag to orbit, middle-drag or `Shift`+left-drag to pan, wheel to zoom, camera presets (`Isometric`, `XY (+Z)`, `XZ (-Y)`, `YZ (+X)`), and an **Orthographic projection** toggle.
- **Orthogonal X / Y / Z slices**: toggle individual slice planes and scrub their normalized position across `[0, 1]`.
- **3D isosurfaces**: enable **Isosurface** and set the scalar iso-value to extract a shaded 3D mesh via marching tetrahedra.
- **Thresholded 3D point clouds & cutaways**: inspect unstructured 3D clouds (such as obstacle domains or 3D research trajectories) with scalar threshold filtering, point-size scaling, and a cutaway X-clip slider.
- **2D heightfield warping**: switch any 2D field to **3D Viewport** and adjust **2D height warp** to inspect surface topography.

## Transient playback and frame export

For any dataset with a physical time axis `t` (1D space–time, 2D space–time, or 3D space–time):

- Use the bottom time scrubber bar with **Play / Pause**, **FPS** selector (`1–30 FPS`), and **Loop**.
- Click **Time trace** to plot the time evolution at the probed location (or spatial mean/max) in the **Probe & diagnostics** dock.
- Click **Export frames…** to export every physical time frame as numbered PNG images (`frame_0000.png`, …) using your active colormap, overlays, and locked scale.

## Discrete parameter-study explorer

When you open a completed parameter study folder via **Open study…** (or from **Study Results → Explore in Solution Explorer…**):

- One slider/selector appears automatically for each swept parameter axis in `study.json`.
- Toggle **Compare difference vs baseline DP** to subtract any chosen baseline design point from the active trial.
- Choose a **Response axis** and **Response metric** (`Spatial max`, `Spatial min`, `Spatial mean`, `Spatial L2 RMS`) and click **Plot parametric response** to chart how the field responds across the sweep.

## Native continuous parametric PINNs

Instead of training separate networks per design point, you can train a **single continuous parametric PINN** $u_\theta(\mathbf{x}, t, \boldsymbol{\mu})$ over up to four continuous parameter axes:

1. Select the **Domain** block and open **Continuous parametric axes** in Properties.
2. Click **Add parameter axis…** and define a symbol name (for example `mu`), `min`, `max`, and `default`.
3. Reference that symbol directly in **PDE residual**, **Boundary condition**, and **Reference** expressions (for example `mu * laplace(u) + 2*pi*pi*sin(pi*x)*sin(pi*y)` and reference `sin(pi*x)*sin(pi*y)/mu`).
4. Try the runnable preset under **File → Examples → Parametric examples → 2D parametric Poisson — u(x,y; μ)**.

During training, interior and boundary samplers jointly sample space, time, and the configured parameter intervals; `HardDirichletPINN` applies its vanishing wall factor strictly to the spatial axes so exact Dirichlet walls hold across every parameter value. Initial evaluation grids fix parameters at their `default` values to keep output size bounded.

When you open the trained output (or reopen its saved `.margam-run.pt` run archive) in Solution Explorer:

- Continuous sliders and spinboxes appear in the **Parametric** panel.
- Dragging any slider runs debounced background-thread neural inference on the active model or saved checkpoint and refreshes the 1D, 2D, or 3D viewport live while preserving locked colorbars.
- Click **Plot parametric response** to sweep the chosen parameter smoothly across its interval and inspect the response curve.

## Launch from Python (`margam.research`)

From any Python script or notebook, call `margam.visualization.explore`:

```python
import numpy as np
from margam.visualization import explore

# 1) Open a saved run archive, NPZ/CSV/NetCDF file, or study folder
explore("path/to/run.margam-run.pt")
explore("path/to/study_folder")

# 2) Explore a live 3D transient parametric model or callable directly
def velocity_model(coords):
    x, y, z, t, re = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3], coords[:, 4]
    u = np.sin(np.pi * x) * np.cos(t) * (re / 100.0)
    v = np.cos(np.pi * y) * np.sin(t)
    w = np.sin(np.pi * z) * 0.5
    p = u * u + v * v
    return {"u": u, "v": v, "w": w, "p": p}

explore(
    velocity_model,
    bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    resolution=13,
    times=[0.0, 0.25, 0.5, 0.75, 1.0],
    param_specs=[{"name": "Re", "min": 50.0, "max": 300.0, "default": 100.0}],
    obstacle={"kind": "sphere", "center": (0.5, 0.5, 0.5), "radius": 0.2},
    title="3D Sphere Flow",
)
```

For headless or offscreen testing, pass `block=False` to receive the `SolutionExplorerWindow` instance directly without entering a blocking Qt event loop.
