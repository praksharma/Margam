# Margam repository instructions

## Working location and environment

- Work in this repository. On the current machine it is
  `/home/hell/Desktop/Margam`. Do not put project code or documentation in a
  separate ChatGPT workspace, even if that is the chat's initial directory.
- Reuse the single repository-root `.venv`. Check its Python version and installed
  dependencies before changing it. Current development uses Python 3.14.
- Do not create virtual environments inside feature folders or invent additional
  environment/backup directories. A Python upgrade should be a deliberate,
  user-authorized change to the root environment.
- Use `uv pip ... --python .venv/bin/python` for dependency management. Do not assume
  the environment has its own pip module. Follow README.md for running commands.

## Repository organisation

- Source: `margam/`; numerical methods: `margam/methods/`; shared components:
  `margam/components/`; configured examples: `margam/templates/`; desktop: `margam/ui/`.
- Keep one file per PINN method. Reuse shared sampling, optimisation, networks and
  loss components, while allowing a method to own its distinctive behaviour.
- Tests belong in `tests/`, runnable examples in `examples/`, documentation and
  plans in `docs/`, static resources in `assets/`, screenshots in
  `assets/screenshots/`. Keep the root limited to standard project metadata,
  instructions and setup files. Do not leave scratch code or images there.
- Keep implementation status, component capabilities and limitations documented.
  Update README.md running commands and docs/todo.md as work changes.

## Scope and interaction design

- Build concrete supported workflows first. Do not invent generic registries,
  extension frameworks, universal plot actions or broad abstractions without a
  demonstrated need. Share actual common behaviour rather than hypothetical cases.
- Follow the documented Simulink interaction reference in docs/simulink-evidence.md,
  distinguishing verified behaviours from deliberate Margam choices.
- The project is a PINN builder across physics, not a heat-transfer-only tool.
  Keep demo-specific assumptions in templates rather than shared UI contracts.
- Basic interval/rectangle sampling must work directly. Advanced CAD/geometry
  workflows belong to future GeoSampler integration. Do not add a 3D viewport.
- Views are optional and owned by their blocks. Do not show global default figure
  panes or assume an analytical solution. Only expose supported context actions.
- Local sample/geometry previews currently belong to Rectangle blocks only.
  Keep operational progress/errors separate from numerical output views.
- Do not continue polishing or expanding in-app field heatmaps without a concrete
  user request. Preserve existing behaviour unless the user requests removal.
- ParaView integration and export are deferred. Record them in TODO; do not start
  implementing them during unrelated tasks.

## Validation and commits

- Check the current branch and working tree before editing; preserve user changes.
- Keep `dev` untouched unless the user explicitly authorizes changes to it.
- Run checks appropriate to the change. Verify UI interactions and lifecycle where
  relevant; store screenshots under assets/screenshots. Label CPU-only validation
  and CUDA skips accurately. Do not claim numerical accuracy from UI startup alone.
- Make regular, coherent commits at verified milestones. Do not leave completed
  work uncommitted or push changes unless the user requests it.
- Every commit created or amended by Codex must include this exact trailer,
  separated from the commit body by a blank line:

  ```text
  Co-authored-by: Codex <codex@openai.com>
  ```

  This includes documentation-only commits. Preserve the human author's identity.
  Do not rewrite existing history unless explicitly authorized for that purpose.
