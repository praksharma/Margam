# Margam

Margam is a visual builder for physics-informed neural networks. Connect reusable
components to define a problem, train a PINN, inspect results and export predictions.

![Margam block editor](assets/screenshots/manual-expanded.png)

## Quick start

Use Python 3.14 and [uv](https://docs.astral.sh/uv/). Run commands from the repository
root and reuse its single `.venv`.

For a fresh checkout without an environment:

```sh
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -e .
```

Launch Margam:

```sh
.venv/bin/python -m margam.ui
```

See [installation help](docs/start/install.md) for setup details.

## Documentation

- [Illustrated user manual](docs/index.md)
- [Your first PINN](docs/start/first-pinn.md)
- [Geometry documents with Jorvik](docs/jorvik.md)
- [Technical reference](docs/README.md)
- [Development and experiment commands](docs/commands.md)
- [Roadmap and planned work](docs/todo.md)

**F1 → User manual** opens the locally built guide.
[Build the manual](docs/documentation.md) once to enable it.

## Contributing

Read the [repository instructions](AGENTS.md) and [extension guide](docs/adding-a-pinn.md).

[Licence](LICENSE) · [Third-party attribution](THIRD_PARTY.md)
