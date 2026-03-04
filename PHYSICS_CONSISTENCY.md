# Physics consistency review (legacy `full_solver` as ground truth)

This review focuses on the **refraction path only** and treats `src/legacy/full_solver.py` as the canonical legacy reference.

## Scope checked

- Refraction equation form (`dx/dt`, `dv/dt`) in legacy and refactored solver paths.
- Plasma critical-density normalization constant usage.
- Beam initial phase-space sampling, because `Np`-dependent differences are usually sampling + solver statistics, not changes in governing physics.

## Refraction equation parity

### Legacy ground truth (`full_solver`)

- `calc_dndr` computes:
  - `omega = 2*pi*c/lambda`
  - `nc = 3.14207787e-4 * omega**2`
  - `dndx,dndy,dndz = -0.5*c**2*grad(ne/nc)`
- `dsdt` computes:
  - `dx/dt = v`
  - `dv/dt = dndr(x)`

This is the reference refraction model.

### Refactored (`core/propagator.py`)

- For `edensity=True`, it uses:
  - `gradient_term = -0.5*c**2 * ne / (3.14207787e-4 * omega**2)`
  - then `dv/dt = grad(gradient_term)` and `dx/dt = v`.

That is equivalent in form to legacy for the refraction-only path.

## `Np`-sensitivity bug fixed in this PR

### Root cause

`src/core/beam.py` previously called helper RNG functions that each could reseed NumPy with the same seed inside one beam build, creating correlated `t/u/phi/chi` streams. This can produce artificial differences when changing `Np`.

### Fix now applied

- Beam construction now uses a **single local RNG stream** with legacy-compatible algorithm:
  - `rng = np.random.RandomState(seed)`
  - sequential draws via `rng.rand`, `rng.randn`, `rng.power`.

Why `RandomState` (not `default_rng`):
- Legacy uses `np.random.rand/randn` (Mersenne Twister family stream semantics).
- `RandomState` keeps seeded behavior closer to legacy expectations while still avoiding global reseed side effects.

## What this means before pushing

- The governing refraction physics remains consistent with legacy `full_solver`.
- The most likely non-physical source of `Np`-dependent drift in beam initialisation has been removed.
- Remaining differences (if any) should now primarily come from numerical solver/interpolation differences (Diffrax/JAX path vs SciPy legacy path), which require runtime parity sweeps to quantify.

## Recommended pre-push runtime checks (when JAX runtime available)

1. Use one identical density field (e.g. slab / Gaussian lens) and same probe direction.
2. Compare legacy `full_solver` vs refactor for `Np = [1e3, 3e3, 1e4, 3e4]`.
3. Compare statistics (not ray-by-ray identity):
   - mean/std of final plane positions,
   - mean/std of deflection angles,
   - 5/50/95% quantiles.
4. Confirm convergence with increasing `Np` and stable legacy-vs-refactor gaps.


## Diagnostic-processing mismatch fixed

A second mismatch source was identified in `src/processing/diagnostics.py`:
- input rays were being pre-filtered by `lens_cutoff(...)` during object construction, before the legacy-equivalent diagnostic solve chain.
- legacy `rtm_solver` applies optical cutoffs during solve stages, not as a mandatory constructor prefilter.

This PR changes processing diagnostics to preserve legacy flow by default (`prefilter_input=False`) and only apply early prefiltering when explicitly requested.

Compatibility helpers were also added so `solve()` defaults match legacy expectations for `Shadowgraphy`, `Schlieren`, and `Refractometry`.
