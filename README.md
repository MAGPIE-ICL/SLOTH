**SLOTH** stands for **Scalable Laser geometric Optics Tracer for HEDP**.

This repository currently contains a refactored geometric-optics ray tracing stack alongside legacy reference solvers.

## Scope

Primary focus:
- Refraction-driven ray tracing through scalar refractive/electron-density fields.
- Beam generation, propagation, and diagnostics utilities.
- Retention of legacy solver implementations for reference and parity checks.

## Repository layout

- `src/core/` — main simulation building blocks (`domain`, `beam`, `propagator`, interpolator, config).
- `src/shared/` — shared math/helpers and propagation transforms.
- `src/legacy/` — legacy solver implementations used as behavioral reference.
- `src/utils/` — I/O and analysis helpers.
- `src/processing/` — diagnostic post-processing.
- `examples/` — exploratory notebooks/scripts.

## Current project status

- The repository does **not** yet include packaging metadata (`pyproject.toml`/`setup.py`).
- The previous `pip install -e .` quick-start instructions are therefore not valid yet.

## Running locally (current)

Use the repository root on `PYTHONPATH` while iterating locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy jax equinox
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
```

Then import modules such as `core.domain`, `core.beam`, and `core.propagator`.

## Audit and improvement plan

See `AUDIT.md` for current inconsistencies, risks, and prioritized improvements.
