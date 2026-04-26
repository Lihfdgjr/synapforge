# Contributing to synapforge

Thanks for your interest. synapforge is small, focused, and OSS — PRs are
welcome but please open an issue first for non-trivial changes so we can align
on direction.

## Quick start

```bash
git clone https://github.com/Lihfdgjr/synapforge
cd synapforge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## Code style

- **Formatter / linter**: `ruff check . && ruff format .`
- **Type-checker**: `mypy synapforge/` (strict on new code)
- **Line length**: 100 (set in `pyproject.toml`)

## Testing

```bash
pytest tests/ -v                          # CPU + GPU when available
pytest tests/ --cov=synapforge            # with coverage
pytest tests/test_correctness.py -v       # fast smoke
```

GPU / Triton tests skip gracefully on machines without CUDA.

## PR checklist

- [ ] `ruff check .` is clean
- [ ] `pytest tests/ -v` is green
- [ ] If adding a public API, update `synapforge/__init__.py.__all__`
- [ ] If changing user-facing behavior, update `CHANGELOG.md` (Unreleased section)
- [ ] If touching a backend, add a correctness test under `tests/`
- [ ] PR description explains the *why*, not just the *what*

## Scope

We are picky about what lands. The framework is purpose-built for liquid +
spiking + plastic networks. PRs that add general-purpose deep-learning utilities
will be politely redirected to PyTorch.

We **do** want:

- New surrogate gradients
- New plasticity rules (with cited paper)
- New backends (analog, FPGA, custom ASIC)
- Bug fixes with regression tests
- Performance wins with benchmark numbers

We **don't** want:

- Generic transformer layers
- Reinventing PyTorch primitives
- Drive-by code-style PRs without a behavior change

## Releasing (maintainers only)

```bash
# 1. Bump version in pyproject.toml + synapforge/__init__.py
# 2. Update CHANGELOG.md
git tag v0.X.Y
git push --tags
# release.yml will build sdist+wheel and upload to PyPI
```

## License

By submitting a PR you agree your contribution is licensed under the project's
[MIT license](LICENSE).
