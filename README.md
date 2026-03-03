**SLOTH** stands for **Scalable Laser geometric Optics Tracer for HEDP**.

This repository is a refactored, reduced-scope variant of `synthPy`, focused only on **geometric optics with refraction**.

## Scope

SLOTH keeps:
- Refraction-driven ray tracing (`n(x, y, z)` based media)
- Minimal ray/state models
- A small simulation pipeline for HEDP-focused use cases

SLOTH intentionally removes:
- Phase-contrast imaging physics
- Faraday rotation / polarization effects
- Non-essential diagnostics and post-processing modules unrelated to refraction
- Monolithic multiphysics coupling interfaces

## Repository layout

- `src/sloth/models.py` — core data models (`Ray`, `RayBundle`, `Medium`)
- `src/sloth/tracer.py` — refraction-only ray marcher
- `src/sloth/api.py` — simple high-level simulation API
- `tests/` — focused tests for refraction-only behavior

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest
```

## Design notes

- The code path is intentionally narrow and explicit.
- New features should align with the refraction-only mission.
- If additional physics are needed, they should live in optional downstream extensions, not in SLOTH core.