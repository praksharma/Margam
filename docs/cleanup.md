# Package rename and staged legacy cleanup

26 October 2026. Work remains on the current main checkout; dev was not changed.

1. Renamed DeepINN to margam and updated maintained examples and tests.
2. Preserved the baseline method, problem/constraint components, loss aggregation,
   fully connected network, differentiation operators and benchmark runner.
3. Retired Model/domain/constraint compatibility API and its dedicated compatibility
   test. Numerical behaviour tests remain. Removed borrowed geometry, unfinished
   geometry_refactor, old data/plotting/callback helpers, display and config modules
   only after the modern engine passed independently. Deleted source was checked
   against recoverable Git history to avoid discarding local modifications.
4. Removed old tutorials and smoke scripts targeting retired APIs, Sphinx/Jupyter
   Book deployment scaffolding, UML exports, legacy dependency snapshots, Docker
   and release pipelines targeting the old package. Git retains their history.
5. Moved UI source into margam/ui and planning documentation into docs. Kept the
   single active environment at the repository root and screenshots under assets.
6. Replaced placeholder setup.py packaging with pyproject.toml and modernised
   dependencies and CI for Python 3.14. Historical saved benchmark outputs were
   preserved locally; their recorded package names/versions are historical data.

The current environment has CPU PyTorch. CUDA checks skip; this migration does
not claim GPU validation. There is no legacy import shim. The UI remains a layout
prototype, not a connected numerical model. GeoSampler integration is future work.
