# Current TODO

See roadmap.md for the agreed design and stages. Completed historical DeepINN
checkboxes are not claims about current Margam support.

- [ ] Catalogue implemented methods, sampling and optimisation components with limitations.
- [ ] Reuse GeoSampler unit-domain samplers and expose supported settings.
- [ ] Add built-in intervals, squares and rectangles with meshless sampling.
- [ ] Select rectangle sides, create named boundary point sets, and assign BCs later.
- [ ] Define PDE residual and PDE-point interfaces; defer pointwise forcing inputs.
- [ ] Add BC type/operator/target settings, normal requirements and corner policy.
- [ ] Add prepared-data adapters with explicit coordinate/label mapping and previews.
- [ ] Define typed ports, connection validation and executable graph semantics.
- [ ] Add saved models, undo/redo, deletion and convenient canvas navigation.
- [ ] Connect the baseline engine through background training with progress and cancellation.
- [ ] Add a complete Poisson visual template and compare with the Python example.
- [ ] Reuse GeoSampler dock persistence and recovery patterns where appropriate.
- [ ] Integrate GeoSampler for complex geometry later; no 3D viewport in this project.
- [ ] Validate a compatible GPU PyTorch build in the Python 3.14 environment.
