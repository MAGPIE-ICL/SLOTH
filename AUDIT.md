# Repository audit (legacy as ground truth)

This audit is based on static inspection of the current repository and quick command checks.

## High-impact inconsistencies

1. **README does not match the code layout**
   - `README.md` references `src/sloth/models.py`, `src/sloth/tracer.py`, `src/sloth/api.py`, and `tests/`.
   - Those paths do not exist in this repository; instead the code lives under `src/core`, `src/shared`, `src/legacy`, `src/utils`, and `src/processing`.

2. **Packaging instructions are currently broken**
   - `pip install -e .` fails because there is no `pyproject.toml` or `setup.py`.
   - This is a blocking issue for users following Quick Start literally.

3. **Legacy + refactor coexist, but boundaries are unclear**
   - Legacy solvers are retained under `src/legacy`, but there is no explicit compatibility matrix or parity test documenting what behavior must match legacy.
   - This makes "legacy is ground truth" hard to enforce automatically.

## Potential bug risks (code-level)

1. **Import fragility in package context**
   - Internal modules frequently import with top-level names (`from shared...`, `from core...`) instead of package-relative imports.
   - This can work only when `src` is manually on `PYTHONPATH`, but is fragile for normal packaging and imports.

2. **Typoed API name with compatibility cost**
   - `back_propogate` appears in `src/shared/propagation.py` and is imported elsewhere.
   - The misspelling is likely accidental; renaming without alias could break callers, but keeping it indefinitely also spreads inconsistency.

3. **Docstring/terminology drift**
   - Multiple docstrings describe removed/legacy capabilities (polarisation/RTM wording) in active modules.
   - This can cause misuse and maintenance confusion relative to the reduced project scope.

## Recommended improvements (prioritized)

### P0
- Add minimal packaging metadata (`pyproject.toml`) and expose a canonical import path.
- Add smoke tests that assert:
  - core modules import,
  - one tiny domain + beam + propagator step runs,
  - legacy reference case remains numerically close for a fixed seed.

### P1
- Introduce a `compat` policy file:
  - what legacy behavior is canonical,
  - acceptable numerical tolerances,
  - supported probing directions and units.
- Normalize imports to one style (prefer explicit package-relative).

### P2
- Add API cleanup aliases (e.g. `back_propagate` -> alias to `back_propogate`) with deprecation warning.
- Remove stale mentions of out-of-scope physics from docs and comments.

## Commands used

- `rg --files | head -n 200`
- `find . -maxdepth 2 -type f | sort`
- `python -m compileall -q src`
- `python -m pip install -e .`
- targeted `sed`/`rg` inspections of `README.md` and `src/` modules.
