# Margam desktop layout prototype

PySide6 is the proposed desktop toolkit. Use Qt Widgets for the application
shell and Graphics View for the 2D block canvas. No 3D viewport is planned.

## Layout

- Left: block library; search and method/template catalogues are planned.
- Centre: movable blocks with ports and connections.
- Right: properties of the selected block.
- Bottom: 2D domain/sample preview, followed later by results and loss plots;
  a separate messages tab explains validation and execution issues.

For a rectangle, the eventual properties panel will provide bounds, sampling
settings, and side selection. This prototype only displays a static domain
preview; it does not generate a training dataset or assign boundary conditions.

## Run

Use an environment with PySide6 installed:

```sh
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m margam.ui
```

Double-click a library entry to add a block. Drag blocks to move them. Drag a
right-hand output port onto another block's left-hand input port to connect.
Select a block to see its name and summary. Panels can be resized and docked.

## Deliberate prototype limits

This is a layout and interaction study, not a PINN implementation. Connections
have no physics semantics yet, and the initial graph illustrates a workflow
rather than an executable dependency specification. Ports are generic and allow
multiple incoming connections; typed ports and connection validation need design.
Properties are read-only. Saving, undo/redo, deletion, zoom controls, background
training, catalogue discovery, and backend execution are not implemented.

The prototype uses the repository-root `.venv` with Python 3.14.8 and PySide6 6.12.0.
Launch from the repository root with:

```sh
.venv/bin/python -m margam.ui
```

Offscreen startup, initial connections, block movement, selection, and screenshot
generation passed. The rendered layout was inspected. Mouse-driven interaction
still needs a desktop check.

## Appearance

Use Appearance → Light or Dark to switch themes. The palette is adapted from
GeoSampler/PyniteGUI with attribution in `../THIRD_PARTY.md`. Both themes also apply
to blocks, ports, wires and the domain preview; both passed offscreen rendering
checks. Theme choice is not yet persisted across launches.
