"""
Analytic quadratic-trough benchmark for the JAX ray tracer.

The test domain has a parabolic electron-density profile in the transverse (x)
direction and is uniform in y and z::

    ne(x, y, z) = ne_0 * (x / a)²

where *ne_0* is the peak density at the domain edge and *a* is the half-width.

Physics
-------
The ODE driving term in the propagator is::

    gradient_term = coeff * ne,   coeff = -0.5 * c² / (α * ω²)

Differentiating with respect to x gives a restoring force toward x = 0::

    d²x/dt² = coeff * 2 * ne_0 * x / a² = -(ω_osc)² * x

where the oscillation frequency is::

    ω_osc = c * √(ne_0) / (a * √α * ω_laser)

Collimated rays (vx₀ = 0 at entry) therefore undergo simple harmonic motion::

    x(t)  = x₀ · cos(ω_osc · t)
    vx(t) = -x₀ · ω_osc · sin(ω_osc · t)

The integration time used by ``trace_and_save_depths`` for a ray at speed *c*
to cover ``depth_max`` is::

    t_trace = depth_max / c

(the sqrt(8) factor in the ODE normalisation ensures the solver has enough
head-room for angled rays, but the save-point timing is set by depth/c).

Test design
-----------
``_NE_0`` is chosen so that ``ω_osc · t_trace = π/4`` (one eighth of a full
period), giving an observable but not too large oscillation amplitude::

    x_final = x₀ · cos(π/4) = x₀ / √2

Benchmark
---------
``TestBenchmark`` measures wall-clock time with and without inverse
bremsstrahlung on the same domain.  No timing failure is asserted on CI (CI
machines vary), but a 10× ceiling guards against catastrophic regressions such
as the pre-fix behaviour of recomputing full-grid ``jnp.gradient`` calls inside
the ODE body at every adaptive time step.

Usage::

    python -m pytest tests/test_quadratic_trough.py -v

"""

import os
import sys
import time

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Source path
# ---------------------------------------------------------------------------
_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import jax
import jax.numpy as jnp

jax.config.update('jax_platform_name', 'cpu')

from scipy.constants import c
from core.propagator import trace_and_save_depths, kappa_inv_brems

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
# α = mₑ ε₀ / e²  (matches the 3.14207787e-4 constant in propagator.py)
_ALPHA = 3.14207787e-4


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
_LWL   = 1064e-9
_OMEGA = 2 * np.pi * c / _LWL   # laser angular frequency (rad/s)

# Domain geometry
_A      = 5e-3    # x/y half-width (m) — domain spans ±5 mm in x and y
_HALF_Z = 11e-3   # z half-depth (m)   — domain spans ±11 mm in z (22 mm total)
_DEPTH_MAX = 20e-3  # trace depth (m) — kept < domain z length to avoid float32 issues

# Design: rays complete exactly π/4 of an oscillation over _DEPTH_MAX.
# This fixes ne_0 uniquely for the given geometry and wavelength.
_ANGLE     = np.pi / 4.0
_T_TRACE   = _DEPTH_MAX / c                           # 6.67e-11 s
_OMEGA_OSC = _ANGLE / _T_TRACE                        # SHM frequency (rad/s)

# Back-solve ne_0 from ω_osc = c · √ne_0 / (a · √α · ω_laser)
_NE_0 = (_OMEGA_OSC * _A * np.sqrt(_ALPHA) * _OMEGA / c) ** 2
# ≈ 3.8e25 m⁻³  (< 4 % of critical density — well sub-critical, no float32 overflow)

# Initial off-axis positions (all well inside the domain)
_X0_VALS = np.array([0.5e-3, 1.0e-3, 1.5e-3, 2.0e-3])  # (m)

# Position tolerance: 0.2 mm — well above ODE/float32 noise (~µm) and below
# the expected signal (0.29 – 0.59 mm change over the trace).
_TOL_X_M    = 0.2e-3
_TOL_VX_REL = 0.10    # 10 % relative velocity tolerance


# ---------------------------------------------------------------------------
# Analytic solution helpers
# ---------------------------------------------------------------------------

def _analytic_x(x0):
    """Analytic x position after one SHM period fraction _ANGLE."""
    return x0 * np.cos(_ANGLE)         # = x0 / √2


def _analytic_vx(x0):
    """Analytic x velocity at the same time."""
    return -x0 * _OMEGA_OSC * np.sin(_ANGLE)


# ---------------------------------------------------------------------------
# Domain / beam factories
# ---------------------------------------------------------------------------

class _QuadraticTroughDomain:
    """
    Minimal domain with parabolic density  ne(x) = ne_0 · (x/a)².

    Only the attributes read by ``trace_and_save_depths`` are set.
    """

    def __init__(self, ne_0, a, half_z, n_xy=24, n_z=24,
                 inv_brems=False, Te=None, Z=None):
        x = np.linspace(-a, a, n_xy)
        y = np.linspace(-a, a, n_xy)
        z = np.linspace(-half_z, half_z, n_z)

        XX, _, _ = np.meshgrid(x, y, z, indexing='ij')
        ne = (ne_0 * (XX / a) ** 2).astype(np.float32)

        self.ne  = jnp.array(ne)
        self.x   = jnp.array(x, dtype=jnp.float32)
        self.y   = jnp.array(y, dtype=jnp.float32)
        self.z   = jnp.array(z, dtype=jnp.float32)
        self.probing_direction = 'z'
        self.lengths = jnp.array(
            [x[-1] - x[0], y[-1] - y[0], z[-1] - z[0]], dtype=jnp.float32
        )
        self.dims    = jnp.array([len(x), len(y), len(z)], dtype=jnp.int32)
        self.Np_total = None

        self.inv_brems = inv_brems
        self.Te = jnp.asarray(Te, dtype=jnp.float32) if Te is not None else None
        self.Z  = jnp.asarray(Z,  dtype=jnp.float32) if Z  is not None else None


def _collimated_rays(x0_vals, z_start):
    """
    Return a (6, N) array of collimated rays in +z at speed c.
    Each ray starts at (x0, 0, z_start) with velocity (0, 0, c).
    """
    Np = len(x0_vals)
    s0 = np.zeros((6, Np))
    s0[0, :] = x0_vals
    s0[2, :] = z_start
    s0[5, :] = c
    return jnp.array(s0, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# TestQuadraticTroughNoBrems
# ---------------------------------------------------------------------------

class TestQuadraticTroughNoBrems:
    """
    Rays in a parabolic density trough follow SHM.
    Check analytic vs. numerical position and angle at the final depth.
    """

    def _run(self, n_xy=24, n_z=24):
        domain = _QuadraticTroughDomain(
            _NE_0, _A, _HALF_Z, n_xy=n_xy, n_z=n_z,
        )
        s0 = _collimated_rays(_X0_VALS, z_start=-_HALF_Z)
        return trace_and_save_depths(
            s0, domain,
            step=_DEPTH_MAX,
            depth_max=_DEPTH_MAX,
            output_path=None,
            lwl=_LWL,
            jones_components=[0, 1],   # x-position and x-angle
            jitted=True,
            verbose=False,
        )

    def test_x_position_matches_analytic(self):
        """Final x position must match the SHM analytic formula (atol 0.2 mm)."""
        result   = self._run()
        x_final  = np.asarray(result['jvec'][-1][0])
        x_expect = _analytic_x(_X0_VALS)

        np.testing.assert_allclose(
            x_final, x_expect,
            atol=_TOL_X_M,
            err_msg=(
                f"x position mismatch (atol={_TOL_X_M*1e3:.1f} mm).\n"
                f"  numerical : {x_final*1e3} mm\n"
                f"  analytic  : {x_expect*1e3} mm\n"
                f"  diff      : {np.abs(x_final-x_expect)*1e3} mm"
            ),
        )

    def test_x_velocity_matches_analytic(self):
        """Final x velocity must match the SHM analytic formula (rtol 10 %)."""
        result    = self._run()
        phi_final = np.asarray(result['jvec'][-1][1])   # angle = vx / vz ≈ vx/c
        vx_final  = phi_final * c
        vx_expect = _analytic_vx(_X0_VALS)

        nz = np.abs(vx_expect) > 1e-3 * c
        if nz.any():
            np.testing.assert_allclose(
                vx_final[nz], vx_expect[nz],
                rtol=_TOL_VX_REL,
                err_msg=f"x velocity mismatch (rtol={_TOL_VX_REL*100:.0f}%)",
            )

    def test_y_position_unchanged(self):
        """Rays starting at y=0 with vy=0 must remain at y≈0."""
        domain = _QuadraticTroughDomain(_NE_0, _A, _HALF_Z, n_xy=20, n_z=20)
        s0 = _collimated_rays(_X0_VALS, z_start=-_HALF_Z)
        result = trace_and_save_depths(
            s0, domain,
            step=_DEPTH_MAX, depth_max=_DEPTH_MAX,
            output_path=None, lwl=_LWL,
            jones_components=[2, 3],   # y-position and y-angle
            jitted=True, verbose=False,
        )
        y_final = np.asarray(result['jvec'][-1][0])
        np.testing.assert_allclose(
            y_final, 0.0, atol=0.1e-3,
            err_msg="y position must remain zero in a 1-D parabolic trough",
        )

    def test_vacuum_straight_rays(self):
        """Vacuum (ne=0): rays travel straight — x position must not change."""
        n = 16; half = 2e-3
        coords = np.linspace(-half, half, n)

        class _Vac:
            ne = jnp.zeros((n, n, n), dtype=jnp.float32)
            x  = jnp.array(coords, dtype=jnp.float32)
            y  = jnp.array(coords, dtype=jnp.float32)
            z  = jnp.array(coords, dtype=jnp.float32)
            probing_direction = 'z'
            lengths = jnp.array([2*half, 2*half, 2*half], dtype=jnp.float32)
            dims    = jnp.array([n, n, n], dtype=jnp.int32)
            Np_total = None
            inv_brems = False

        x0  = np.array([0.5e-3, 1.0e-3])
        s0  = _collimated_rays(x0, z_start=-half)
        result = trace_and_save_depths(
            s0, _Vac(),
            step=2*half, depth_max=2*half,
            output_path=None, lwl=_LWL,
            jones_components=[0], jitted=True, verbose=False,
        )
        x_final = np.asarray(result['jvec'][-1][0])
        np.testing.assert_allclose(
            x_final, x0, atol=1e-6,
            err_msg="Vacuum: x position must not change",
        )


# ---------------------------------------------------------------------------
# TestQuadraticTroughWithBrems
# ---------------------------------------------------------------------------

class TestQuadraticTroughWithBrems:
    """
    Same parabolic trough but with inv_brems=True.

    The amplitude ODE must not corrupt ray trajectories; amplitude must decay
    monotonically; weighted jvec must be smaller than unweighted jvec.
    """

    _TE = 100.0   # eV
    _Z  = 1.0

    def _domain(self, n_xy=20, n_z=20):
        return _QuadraticTroughDomain(
            _NE_0, _A, _HALF_Z, n_xy=n_xy, n_z=n_z,
            inv_brems=True, Te=self._TE, Z=self._Z,
        )

    def _run(self):
        domain = self._domain()
        s0 = _collimated_rays(_X0_VALS, z_start=-_HALF_Z)
        return trace_and_save_depths(
            s0, domain,
            step=_DEPTH_MAX / 4,        # a few intermediate snapshots
            depth_max=_DEPTH_MAX,
            output_path=None, lwl=_LWL,
            jones_components=[0, 1],
            jitted=True, verbose=False,
        )

    def test_x_position_matches_analytic_with_brems(self):
        """The amplitude ODE must not corrupt the position trajectory."""
        result   = self._run()
        x_geo    = np.asarray(result['jvec_unweighted'][-1][0])
        x_expect = _analytic_x(_X0_VALS)
        np.testing.assert_allclose(
            x_geo, x_expect, atol=_TOL_X_M,
            err_msg=(
                "x position mismatch when inv_brems=True\n"
                f"  numerical (geo) : {x_geo*1e3} mm\n"
                f"  analytic        : {x_expect*1e3} mm"
            ),
        )

    def test_amplitude_decays(self):
        """Amplitude at the final depth must be strictly less than 1."""
        result    = self._run()
        amp_final = float(np.asarray(result['amplitude'][-1]).mean())
        assert amp_final < 1.0, (
            f"Expected amplitude < 1 in absorbing plasma; got {amp_final:.6f}"
        )

    def test_amplitude_monotonically_decreasing(self):
        """Mean amplitude must be non-increasing at successive depth saves."""
        result = self._run()
        amps = [float(np.asarray(a).mean()) for a in result['amplitude']]
        for i in range(1, len(amps)):
            assert amps[i] <= amps[i - 1] + 1e-6, (
                f"Amplitude increased at depth index {i}: "
                f"{amps[i - 1]:.8f} → {amps[i]:.8f}"
            )

    def test_weighted_jvec_smaller_than_unweighted(self):
        """Amplitude weighting must reduce |jvec| in an absorbing medium."""
        result = self._run()
        jv_w   = np.asarray(result['jvec'][-1])
        jv_u   = np.asarray(result['jvec_unweighted'][-1])
        rms_w  = float(np.sqrt(np.nanmean(jv_w ** 2)))
        rms_u  = float(np.sqrt(np.nanmean(jv_u ** 2)))
        assert rms_w <= rms_u + 1e-10, (
            f"Weighted RMS {rms_w:.8f} should be ≤ unweighted {rms_u:.8f}"
        )


# ---------------------------------------------------------------------------
# TestBenchmark
# ---------------------------------------------------------------------------

class TestBenchmark:
    """
    Wall-clock timing comparison: plain solve vs. inv_brems solve.

    The test does **not** enforce a hard timing budget (CI machines vary widely).
    It does assert that the inv_brems overhead is less than 10×, guarding
    against catastrophic regressions such as reintroducing full-grid
    ``jnp.gradient`` calls inside the ODE body at every adaptive step.
    """

    _NP  = 30          # small enough for CI
    _TE  = 100.0
    _Z   = 1.0
    _N   = 16          # grid points per axis

    def _s0(self):
        rng = np.random.RandomState(0)
        x0  = _A * 0.3 * (rng.rand(self._NP) - 0.5)
        return _collimated_rays(x0, z_start=-_HALF_Z)

    def _domain_plain(self):
        return _QuadraticTroughDomain(
            _NE_0, _A, _HALF_Z, n_xy=self._N, n_z=self._N,
        )

    def _domain_ib(self):
        return _QuadraticTroughDomain(
            _NE_0, _A, _HALF_Z, n_xy=self._N, n_z=self._N,
            inv_brems=True, Te=self._TE, Z=self._Z,
        )

    def _run(self, domain, s0):
        return trace_and_save_depths(
            s0, domain,
            step=_DEPTH_MAX, depth_max=_DEPTH_MAX,
            output_path=None, lwl=_LWL,
            jones_components=[0, 1],
            jitted=True, verbose=False,
        )

    def test_inv_brems_overhead_within_10x(self):
        """
        inv_brems must not be more than 10× slower than a plain solve on the
        same domain.  A larger ratio indicates a performance regression (e.g.
        gradient recomputation inside the ODE body).
        """
        s0 = self._s0()
        # Warm up both paths (JIT compilation)
        self._run(self._domain_plain(), s0)
        self._run(self._domain_ib(),    s0)

        t0 = time.perf_counter()
        self._run(self._domain_plain(), s0)
        t_plain = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._run(self._domain_ib(), s0)
        t_ib = time.perf_counter() - t0

        ratio = t_ib / max(t_plain, 1e-9)
        print(
            f"\n[Benchmark] Np={self._NP}  "
            f"plain: {t_plain:.3f}s  inv_brems: {t_ib:.3f}s  ratio: {ratio:.2f}×"
        )
        assert ratio < 10.0, (
            f"inv_brems is {ratio:.1f}× slower than plain (limit 10×). "
            "Likely a performance regression — check that gradient grids are "
            "pre-computed outside the ODE body."
        )

    def test_geometry_identical_with_and_without_brems(self):
        """
        The geometric (unweighted) ray positions from inv_brems must match the
        plain solve — the amplitude ODE must not perturb ray trajectories.
        """
        s0 = self._s0()
        res_plain = self._run(self._domain_plain(), s0)
        res_ib    = self._run(self._domain_ib(),    s0)

        x_plain  = np.asarray(res_plain['jvec'][-1][0])
        x_ib_geo = np.asarray(res_ib['jvec_unweighted'][-1][0])

        np.testing.assert_allclose(
            x_ib_geo, x_plain, atol=1e-5,
            err_msg=(
                "Geometric x positions with and without inv_brems must agree. "
                "A mismatch suggests the amplitude ODE corrupts trajectory state."
            ),
        )
