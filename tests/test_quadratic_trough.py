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
import pickle
import tempfile
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from core.propagator import trace_and_save_depths, kappa_inv_brems, precompute_gradients
from core.domain import ScalarDomain
from processing.diagnostics import (Diagnostic, Shadowgraphy, Schlieren,
                                    Refractometry,
                                    plot_amplitude_diagnostics,
                                    compare_diagnostics, transmission_map,
                                    apply_ccd_mask, absorption_sanity_check,
                                    compute_arrangement_rtm,
                                    compute_ccd_bounds_spatial,
                                    compute_ccd_bounds_angular,
                                    build_ccd_acceptance_mask)

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
        x_geo    = np.asarray(result['jvec'][-1][0])
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

    def test_amplitude_weighted_smaller_than_geometric(self):
        """Manually amplitude-weighting jvec must reduce its RMS in an absorbing medium."""
        result = self._run()
        jv_geo = np.asarray(result['jvec'][-1])
        amp    = np.asarray(result['amplitude'][-1])
        jv_weighted = jv_geo * amp[np.newaxis, :]
        rms_geo = float(np.sqrt(np.nanmean(jv_geo ** 2)))
        rms_wt  = float(np.sqrt(np.nanmean(jv_weighted ** 2)))
        assert rms_wt <= rms_geo + 1e-10, (
            f"Weighted RMS {rms_wt:.8f} should be ≤ geometric {rms_geo:.8f}"
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
        The geometric ray positions from inv_brems must match the plain solve —
        the amplitude ODE must not perturb ray trajectories.
        """
        s0 = self._s0()
        res_plain = self._run(self._domain_plain(), s0)
        res_ib    = self._run(self._domain_ib(),    s0)

        x_plain = np.asarray(res_plain['jvec'][-1][0])
        x_ib    = np.asarray(res_ib['jvec'][-1][0])

        np.testing.assert_allclose(
            x_ib, x_plain, atol=1e-5,
            err_msg=(
                "Geometric x positions with and without inv_brems must agree. "
                "A mismatch suggests the amplitude ODE corrupts trajectory state."
            ),
        )


# ---------------------------------------------------------------------------
# Helpers shared by the new test classes below
# ---------------------------------------------------------------------------

def _vacuum_domain(half=2e-3, n=12):
    """Tiny vacuum (ne=0) domain — fast for API/contract tests."""
    coords = np.linspace(-half, half, n)
    ne = np.zeros((n, n, n), dtype=np.float32)

    class _MinimalDomain:
        pass

    d = _MinimalDomain()
    d.ne  = jnp.array(ne)
    d.x   = jnp.array(coords, dtype=jnp.float32)
    d.y   = jnp.array(coords, dtype=jnp.float32)
    d.z   = jnp.array(coords, dtype=jnp.float32)
    d.probing_direction = 'z'
    d.lengths = jnp.array([2*half, 2*half, 2*half], dtype=jnp.float32)
    d.dims    = jnp.array([n, n, n], dtype=jnp.int32)
    d.Np_total = None
    d.inv_brems = False
    return d


def _uniform_plasma_domain(ne_val=1e25, Te_val=100.0, Z_val=1.0, half=3e-3, n=16):
    """Uniform absorbing plasma domain for absorption formula tests."""
    coords = np.linspace(-half, half, n)
    ne = np.full((n, n, n), ne_val, dtype=np.float32)

    class _MinimalDomain:
        pass

    d = _MinimalDomain()
    d.ne  = jnp.array(ne)
    d.x   = jnp.array(coords, dtype=jnp.float32)
    d.y   = jnp.array(coords, dtype=jnp.float32)
    d.z   = jnp.array(coords, dtype=jnp.float32)
    d.probing_direction = 'z'
    d.lengths = jnp.array([2*half, 2*half, 2*half], dtype=jnp.float32)
    d.dims    = jnp.array([n, n, n], dtype=jnp.int32)
    d.Np_total = None
    d.inv_brems = True
    d.Te = jnp.float32(Te_val)
    d.Z  = jnp.float32(Z_val)
    return d


def _small_rays(Np=8, z_start=-2e-3):
    s0 = np.zeros((6, Np))
    s0[0] = np.linspace(-5e-4, 5e-4, Np)
    s0[2] = z_start
    s0[5] = c
    return jnp.array(s0, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# TestAPIContracts
# ---------------------------------------------------------------------------

class TestAPIContracts:
    """
    Minimal API contract tests for trace_and_save_depths.

    These protect the output structure and input-validation behaviour that
    downstream code depends on.  The quadratic-trough classes are the physics
    benchmark; this class protects the interface.
    """

    def _run(self, domain, **kw):
        defaults = dict(step=500e-6, depth_max=1e-3, output_path=None,
                        lwl=1064e-9, jitted=True, verbose=False)
        defaults.update(kw)
        return trace_and_save_depths(_small_rays(), domain, **defaults)

    def test_output_keys_present(self):
        """Result must have depth_saves, jvec, jones_components."""
        result = self._run(_vacuum_domain())
        assert 'depth_saves' in result
        assert 'jvec' in result
        assert 'jones_components' in result

    def test_depth_saves_spacing(self):
        """depth_saves must be uniformly spaced at the requested step."""
        step = 200e-6
        result = self._run(_vacuum_domain(), step=step, depth_max=1e-3)
        diffs = np.diff(result['depth_saves'])
        np.testing.assert_allclose(diffs, step, rtol=1e-6,
                                   err_msg="depth_saves not uniformly spaced")

    def test_domain_too_small_raises(self):
        """depth_max larger than domain extent → ValueError."""
        with pytest.raises(ValueError, match="smaller than the requested depth_max"):
            self._run(_vacuum_domain(half=2e-3), depth_max=10e-3)

    def test_jones_components_partial(self):
        """'position' save must equal rows 0 and 2 of the full save."""
        full = self._run(_vacuum_domain())
        pos  = self._run(_vacuum_domain(), jones_components='position')
        for j in range(len(full['jvec'])):
            np.testing.assert_array_equal(
                pos['jvec'][j], full['jvec'][j][[0, 2], :],
                err_msg=f"position rows at depth {j} do not match rows 0,2 of full save",
            )

    def test_pickle_roundtrip(self):
        """Data written to disk must reload identically."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        try:
            result = self._run(_vacuum_domain(), output_path=path)
            with open(path, 'rb') as fh:
                loaded = pickle.load(fh)
            np.testing.assert_array_equal(loaded['depth_saves'], result['depth_saves'])
            np.testing.assert_array_equal(loaded['jvec'][0], result['jvec'][0])
        finally:
            os.remove(path)

    def test_no_amplitude_key_without_inv_brems(self):
        """amplitude key must NOT appear when inv_brems=False."""
        result = self._run(_vacuum_domain())
        assert 'amplitude' not in result
        assert 'jvec_unweighted' not in result


# ---------------------------------------------------------------------------
# TestAbsorptionFormula
# ---------------------------------------------------------------------------

class TestAbsorptionFormula:
    """
    Unit tests for the inverse-bremsstrahlung absorption formula and the
    domain-level input validation.  These do not require the quadratic-trough
    physics — they test the κ formula and the ODE amplitude bookkeeping.
    """

    LWL   = 1064e-9
    HALF  = 3e-3
    DEPTH = 2e-3

    def _omega(self):
        return 2 * np.pi * c / self.LWL

    def test_analytic_amplitude_match(self):
        """Amplitude at DEPTH must match exp(-κ·DEPTH/c) to 5 %."""
        ne_val, Te_val, Z_val = 1e25, 100.0, 1.0
        omega  = self._omega()
        kappa  = float(kappa_inv_brems(jnp.float32(ne_val), jnp.float32(Te_val), Z_val, omega))
        expected = np.exp(-kappa * self.DEPTH / c)

        domain = _uniform_plasma_domain(ne_val, Te_val, Z_val, self.HALF)
        s0     = _small_rays(Np=10, z_start=-self.HALF)
        result = trace_and_save_depths(s0, domain,
                                       step=500e-6, depth_max=self.DEPTH,
                                       output_path=None, lwl=self.LWL,
                                       jitted=True, verbose=False)
        amp = float(np.asarray(result['amplitude'][-1]).mean())
        np.testing.assert_allclose(amp, expected, rtol=0.05,
                                   err_msg=f"amplitude {amp:.6f} vs analytic {expected:.6f}")

    @pytest.mark.parametrize("Te_eV, Z_val", [(100.0, 1.0), (10.0, 4.0)])
    def test_kappa_coulomb_log_formula(self, Te_eV, Z_val):
        """kappa_inv_brems must match the reference classical-b_min formula."""
        from scipy.constants import e as e_charge, epsilon_0 as eps0
        ne_val = 1e25
        omega  = self._omega()
        ne_cc  = ne_val * 1e-6
        v_the  = 4.19e5 * np.sqrt(Te_eV)
        o_pe   = 5.64e4 * np.sqrt(ne_cc)
        o_max  = max(o_pe, omega)
        b_min  = Z_val * e_charge / (4.0 * np.pi * eps0 * Te_eV)
        CL     = max(2.0, np.log(v_the / (o_max * b_min)))
        kappa_ref = 3.1e-5 * Z_val * c * (ne_cc / omega) ** 2 * CL * Te_eV ** (-1.5)
        kappa_got = float(kappa_inv_brems(jnp.float32(ne_val), jnp.float32(Te_eV),
                                          float(Z_val), omega))
        np.testing.assert_allclose(kappa_got, kappa_ref, rtol=1e-4,
                                   err_msg=f"Te={Te_eV} eV, Z={Z_val}: kappa mismatch")

    def test_negative_Te_raises(self):
        """ScalarDomain must reject negative Te."""
        with pytest.raises(AssertionError, match="eV"):
            ScalarDomain([4e-3, 4e-3, 4e-3], [8, 8, 8], inv_brems=True, Te=-10.0, Z=1.0)

    def test_overcritical_warning(self):
        """precompute_gradients must warn when ne ≥ ncr."""
        omega = self._omega()
        ncr   = 3.14207787e-4 * omega ** 2
        coords = np.linspace(-self.HALF, self.HALF, 8)
        x = y = z = jnp.array(coords, dtype=jnp.float32)
        ne_over = jnp.full((8, 8, 8), 1.5 * ncr, dtype=jnp.float32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            precompute_gradients(ne_over, x, y, z, omega)
        assert len(caught) == 1 and issubclass(caught[0].category, UserWarning)


# ---------------------------------------------------------------------------
# TestDiagnosticWeighting
# ---------------------------------------------------------------------------

class TestDiagnosticWeighting:
    """
    Tests for per-ray amplitude weights in Diagnostic.histogram().

    When inv_brems=True, the caller can pass result['amplitude'] as the
    'weights' argument to Diagnostic to produce an intensity-weighted
    histogram rather than a raw ray-count histogram.

    Key contract:
    * geometry (ray positions, bin edges) is unchanged by weights
    * histogram values (bin intensities) differ when weights differ from 1
    * default (weights=None) is backward-compatible — produces count histogram
    """

    # Use the quadratic-trough domain so rays actually deflect (more realistic)
    _NP = 30

    def _rays_and_domains(self):
        """Return (s0, domain_no_ib, domain_ib) using the global parabolic trough."""
        rng = np.random.RandomState(7)
        x0  = _A * 0.4 * (rng.rand(self._NP) - 0.5)
        s0  = _collimated_rays(x0, z_start=-_HALF_Z)
        dom_plain = _QuadraticTroughDomain(_NE_0, _A, _HALF_Z)
        dom_ib    = _QuadraticTroughDomain(_NE_0, _A, _HALF_Z,
                                           inv_brems=True, Te=100.0, Z=1.0)
        return s0, dom_plain, dom_ib

    def _run(self, domain, s0):
        return trace_and_save_depths(
            s0, domain,
            step=_DEPTH_MAX, depth_max=_DEPTH_MAX,
            output_path=None, lwl=_LWL,
            jones_components=[0, 1, 2, 3],
            jitted=True, verbose=False,
        )

    def _build_diagnostic(self, jvec, amplitude=None):
        """Construct a small Diagnostic object using the final jvec."""
        rf = np.asarray(jvec)   # (4, N) geometric Jones vector
        weights = np.asarray(amplitude) if amplitude is not None else None
        # Use a small detector to keep the test fast
        return Diagnostic(rf, weights=weights, L=100, R=50, Lx=50, Ly=50)

    def test_unweighted_histogram_backward_compatible(self):
        """Diagnostic without weights gives a count histogram (sum = Np)."""
        s0, dom_plain, _ = self._rays_and_domains()
        res = self._run(dom_plain, s0)
        jvec = res['jvec'][-1]
        diag = self._build_diagnostic(jvec)
        diag.histogram(pix_x=50, pix_y=50)
        assert diag.H.sum() == pytest.approx(self._NP, abs=2), \
            "Unweighted histogram total should equal Np (modulo rays outside range)"

    def test_weighted_histogram_total_less_than_count(self):
        """With amplitude < 1 everywhere, weighted sum < Np."""
        s0, dom_plain, dom_ib = self._rays_and_domains()
        res_plain = self._run(dom_plain, s0)
        res_ib    = self._run(dom_ib,    s0)
        jvec = res_plain['jvec'][-1]   # geometry is identical
        amp  = np.asarray(res_ib['amplitude'][-1])
        diag_plain = self._build_diagnostic(jvec)
        diag_wt    = self._build_diagnostic(jvec, amplitude=amp)
        diag_plain.histogram(pix_x=50, pix_y=50)
        diag_wt.histogram(pix_x=50, pix_y=50)
        assert diag_wt.H.sum() < diag_plain.H.sum(), (
            "Amplitude-weighted sum must be < unweighted count "
            f"(got {diag_wt.H.sum():.4f} vs {diag_plain.H.sum():.4f})"
        )

    def test_bin_edges_identical_regardless_of_weights(self):
        """Bin edges must be identical with and without weights — geometry unchanged."""
        s0, dom_plain, dom_ib = self._rays_and_domains()
        res_plain = self._run(dom_plain, s0)
        res_ib    = self._run(dom_ib,    s0)
        jvec = res_plain['jvec'][-1]
        amp  = np.asarray(res_ib['amplitude'][-1])
        diag_plain = self._build_diagnostic(jvec)
        diag_wt    = self._build_diagnostic(jvec, amplitude=amp)
        diag_plain.histogram(pix_x=50, pix_y=50)
        diag_wt.histogram(pix_x=50, pix_y=50)
        np.testing.assert_array_equal(diag_plain.xedges, diag_wt.xedges,
                                      err_msg="x bin edges changed with amplitude weights")
        np.testing.assert_array_equal(diag_plain.yedges, diag_wt.yedges,
                                      err_msg="y bin edges changed with amplitude weights")


# ---------------------------------------------------------------------------
# TestPrefilterAndAmplitudeHelpers
# ---------------------------------------------------------------------------

class TestPrefilterAndAmplitudeHelpers:
    """
    Regression tests covering:
      1. prefilter_input preserves alignment between rf and weights
      2. unit weights produce identical histograms to no weights
      3. artificially strong attenuation produces a measurable histogram reduction
      4. amplitude inspection helpers run without error
    """

    _NP = 40

    def _rays_and_domain(self, inv_brems=False):
        rng = np.random.RandomState(99)
        x0  = _A * 0.5 * (rng.rand(self._NP) - 0.5)
        s0  = _collimated_rays(x0, z_start=-_HALF_Z)
        dom = _QuadraticTroughDomain(_NE_0, _A, _HALF_Z,
                                     inv_brems=inv_brems,
                                     Te=100.0 if inv_brems else None,
                                     Z=1.0   if inv_brems else None)
        return s0, dom

    def _run(self, domain, s0):
        return trace_and_save_depths(
            s0, domain,
            step=_DEPTH_MAX, depth_max=_DEPTH_MAX,
            output_path=None, lwl=_LWL,
            jones_components=[0, 1, 2, 3],
            jitted=True, verbose=False,
        )

    # ------------------------------------------------------------------
    # 1.  prefilter_input alignment
    # ------------------------------------------------------------------

    def test_prefilter_no_weights_runs(self):
        """prefilter_input=True without weights must not raise."""
        s0, dom = self._rays_and_domain()
        res = self._run(dom, s0)
        jvec = res['jvec'][-1]
        # Use a tight lens radius so some rays are actually cut
        diag = Diagnostic(jvec, L=100, R=1, prefilter_input=True)
        diag.histogram(pix_x=20, pix_y=20)
        assert diag.H.sum() <= self._NP

    def test_prefilter_weights_aligned(self):
        """
        After prefilter_input, self.rf.shape[-1] must equal len(self.weights).
        The bug: weights were taken from the original (unfiltered) array.
        """
        s0, dom = self._rays_and_domain(inv_brems=True)
        res = self._run(dom, s0)
        jvec = res['jvec'][-1]
        amp  = np.asarray(res['amplitude'][-1])

        # Tight lens so at least some rays are dropped
        diag = Diagnostic(jvec, weights=amp, L=100, R=1, prefilter_input=True)

        assert diag.rf.shape[-1] == len(diag.weights), (
            f"rf has {diag.rf.shape[-1]} rays but weights has {len(diag.weights)} entries; "
            "prefilter_input must apply the same mask to both."
        )

    def test_prefilter_histogram_runs_with_weights(self):
        """Histogram must complete without shape mismatch after prefilter + weights."""
        s0, dom = self._rays_and_domain(inv_brems=True)
        res = self._run(dom, s0)
        jvec = res['jvec'][-1]
        amp  = np.asarray(res['amplitude'][-1])

        diag = Diagnostic(jvec, weights=amp, L=100, R=1, prefilter_input=True)
        # Would raise IndexError before the fix if sizes differ
        diag.histogram(pix_x=20, pix_y=20)

    # ------------------------------------------------------------------
    # 2.  unit weights are identical to no weights
    # ------------------------------------------------------------------

    def test_unit_weights_match_unweighted(self):
        """weights = array of 1.0 must give exactly the same histogram as weights=None."""
        s0, dom = self._rays_and_domain()
        res = self._run(dom, s0)
        jvec = res['jvec'][-1]

        ones = np.ones(np.asarray(jvec).shape[-1], dtype=np.float64)

        diag_plain = Diagnostic(jvec, L=100, R=50, Lx=50, Ly=50)
        diag_ones  = Diagnostic(jvec, weights=ones, L=100, R=50, Lx=50, Ly=50)
        diag_plain.histogram(pix_x=30, pix_y=30)
        diag_ones.histogram(pix_x=30, pix_y=30)

        np.testing.assert_array_equal(
            diag_plain.H, diag_ones.H,
            err_msg="Unit weights must produce identical histogram to no weights",
        )

    # ------------------------------------------------------------------
    # 3.  strong attenuation is visible
    # ------------------------------------------------------------------

    def test_strong_attenuation_measurable(self):
        """
        Artificially setting all weights to 0.1 must reduce the histogram sum
        to roughly 10 % of the unweighted sum.
        """
        s0, dom = self._rays_and_domain()
        res = self._run(dom, s0)
        jvec = res['jvec'][-1]
        N    = np.asarray(jvec).shape[-1]

        low_amp = np.full(N, 0.1, dtype=np.float64)

        diag_plain = Diagnostic(jvec, L=100, R=50, Lx=50, Ly=50)
        diag_low   = Diagnostic(jvec, weights=low_amp, L=100, R=50, Lx=50, Ly=50)
        diag_plain.histogram(pix_x=30, pix_y=30)
        diag_low.histogram(pix_x=30, pix_y=30)

        ratio = diag_low.H.sum() / diag_plain.H.sum()
        assert abs(ratio - 0.1) < 0.01, (
            f"Expected weighted sum ≈ 10 % of unweighted (ratio={ratio:.4f})"
        )

    # ------------------------------------------------------------------
    # 4.  amplitude inspection helpers
    # ------------------------------------------------------------------

    def test_plot_amplitude_diagnostics_runs(self):
        """plot_amplitude_diagnostics must return (fig, axes) without error."""
        s0, dom = self._rays_and_domain(inv_brems=True)
        res = self._run(dom, s0)
        jvec = res['jvec'][-1]
        amp  = np.asarray(res['amplitude'][-1])

        fig, axes = plot_amplitude_diagnostics(amp, jvec)
        assert len(axes) == 2
        plt.close(fig)

    def test_compare_diagnostics_runs(self):
        """compare_diagnostics must return (fig, axes) and the ratio image must be in [0, 1]."""

        s0, dom_plain = self._rays_and_domain(inv_brems=False)
        s0, dom_ib = self._rays_and_domain(inv_brems=True)

        res_plain = self._run(dom_plain, s0)
        res_ib    = self._run(dom_ib,    s0)
        jvec = res_plain['jvec'][-1]
        amp  = np.asarray(res_ib['amplitude'][-1])

        diag_plain = Diagnostic(jvec, L=100, R=50, Lx=50, Ly=50)
        diag_wt    = Diagnostic(jvec, weights=amp, L=100, R=50, Lx=50, Ly=50)
        diag_plain.histogram(pix_x=20, pix_y=20)
        diag_wt.histogram(pix_x=20, pix_y=20)

        fig, axes = compare_diagnostics(
            diag_plain.H, diag_wt.H,
            diag_plain.xedges, diag_plain.yedges,
        )
        assert len(axes) == 3
        plt.close(fig)

    # ------------------------------------------------------------------
    # 5.  transmission_map recovers uniform weight correctly
    # ------------------------------------------------------------------

    def test_transmission_map_uniform_weights_recovers_amplitude(self):
        """
        For spatially uniform weights w, transmission_map must equal w at every
        occupied pixel.  This verifies that H_wt/H_plain cancels the ray-density
        variation and leaves only the absorption factor.
        """
        s0, dom = self._rays_and_domain()
        res  = self._run(dom, s0)
        jvec = res['jvec'][-1]
        N    = np.asarray(jvec).shape[-1]
        w    = 0.76  # matches user-reported mean for 1064 nm

        uniform_amp = np.full(N, w, dtype=np.float64)

        diag_plain = Diagnostic(jvec, L=100, R=50, Lx=50, Ly=50)
        diag_wt    = Diagnostic(jvec, weights=uniform_amp, L=100, R=50, Lx=50, Ly=50)
        diag_plain.histogram(pix_x=30, pix_y=30)
        diag_wt.histogram(pix_x=30, pix_y=30)

        T = transmission_map(diag_wt.H, diag_plain.H)
        occupied = ~np.isnan(T)
        assert occupied.any(), "No occupied pixels — increase Lx/Ly or bin count"

        np.testing.assert_allclose(
            T[occupied], w, atol=1e-9,
            err_msg=(
                f"transmission_map must equal {w} everywhere for uniform weights; "
                f"got min={T[occupied].min():.6f}, max={T[occupied].max():.6f}"
            ),
        )

    def test_transmission_map_nan_for_empty_pixels(self):
        """Empty bins (H_plain == 0) must be nan in the transmission map."""
        H_plain = np.array([[1.0, 0.0], [2.0, 0.0]])
        H_wt    = np.array([[0.8, 0.0], [1.6, 0.0]])
        T = transmission_map(H_wt, H_plain)
        assert np.isnan(T[0, 1]) and np.isnan(T[1, 1]), \
            "Empty bins must map to nan in transmission_map"
        np.testing.assert_allclose(T[0, 0], 0.8, atol=1e-12)
        np.testing.assert_allclose(T[1, 0], 0.8, atol=1e-12)

    def test_compare_diagnostics_shared_scale(self):
        """
        compare_diagnostics must use a shared color scale for the first two panels
        so a uniform attenuation is visible rather than hidden by auto-normalization.
        The weighted panel's image vmax must equal the unweighted panel's vmax.
        """
        s0, dom = self._rays_and_domain()
        res  = self._run(dom, s0)
        jvec = res['jvec'][-1]
        N    = np.asarray(jvec).shape[-1]

        amp = np.full(N, 0.76, dtype=np.float64)
        diag_plain = Diagnostic(jvec, L=100, R=50, Lx=50, Ly=50)
        diag_wt    = Diagnostic(jvec, weights=amp, L=100, R=50, Lx=50, Ly=50)
        diag_plain.histogram(pix_x=20, pix_y=20)
        diag_wt.histogram(pix_x=20, pix_y=20)

        fig, axes = compare_diagnostics(
            diag_plain.H, diag_wt.H,
            diag_plain.xedges, diag_plain.yedges,
        )
        ax0, ax1, _ = axes
        # Both image panels must share the same clim so the attenuation magnitude
        # is preserved in the visual comparison.
        vmax0 = ax0.get_images()[0].norm.vmax
        vmax1 = ax1.get_images()[0].norm.vmax
        assert vmax0 == vmax1, (
            f"Panel 0 vmax={vmax0} != panel 1 vmax={vmax1}; "
            "shared scale is required so absorption is not hidden by auto-normalization"
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# TestCCDMask
# ---------------------------------------------------------------------------

class TestCCDMask:
    """
    Tests for CCD rectangular mask applied at the diagnostic stage.

    The CCD mask in Diagnostic.histogram() is intended for post-RTM coordinates
    (mm).  The standalone apply_ccd_mask helper works in metres.
    """

    def _make_rf_metres(self, N=100):
        """Return a (4, N) Jones vector with rays in metres (for apply_ccd_mask)."""
        rng = np.random.RandomState(42)
        rf = np.zeros((4, N))
        rf[0] = (rng.rand(N) - 0.5) * 40e-3      # x ±20 mm = ±0.02 m
        rf[2] = (rng.rand(N) - 0.5) * 40e-3      # y ±20 mm
        rf[1] = rng.randn(N) * 1e-4
        rf[3] = rng.randn(N) * 1e-4
        return rf

    def _make_rf_mm(self, N=100):
        """Return a (4, N) Jones vector in mm (simulating post-RTM output)."""
        rng = np.random.RandomState(42)
        rf = np.zeros((4, N))
        rf[0] = (rng.rand(N) - 0.5) * 40      # x ±20 mm
        rf[2] = (rng.rand(N) - 0.5) * 40      # y ±20 mm
        rf[1] = rng.randn(N) * 1e-4
        rf[3] = rng.randn(N) * 1e-4
        return rf

    # ── apply_ccd_mask standalone helper (works in metres) ──

    def test_apply_ccd_mask_excludes_outside(self):
        """Rays outside the CCD footprint must be NaN'd."""
        rf = self._make_rf_metres(200)
        ccd = (10e-3, 12e-3)   # 10 mm × 12 mm
        rf_out, _, mask = apply_ccd_mask(rf, ccd_shape_m=ccd)
        outside = ~mask
        assert outside.any(), "Some rays should be outside a small CCD"
        assert np.all(np.isnan(rf_out[0, outside]))
        assert np.all(np.isnan(rf_out[2, outside]))

    def test_apply_ccd_mask_preserves_inside(self):
        """Rays inside the CCD footprint must be unchanged."""
        rf = self._make_rf_metres(200)
        ccd = (10e-3, 12e-3)
        rf_out, _, mask = apply_ccd_mask(rf, ccd_shape_m=ccd)
        inside = mask
        np.testing.assert_array_equal(rf_out[:, inside], rf[:, inside])

    def test_apply_ccd_mask_weights_zeroed(self):
        """Weights outside the CCD must be set to zero."""
        rf = self._make_rf_metres(200)
        weights = np.ones(200)
        ccd = (10e-3, 12e-3)
        _, w_out, mask = apply_ccd_mask(rf, weights=weights, ccd_shape_m=ccd)
        assert np.all(w_out[~mask] == 0.0)
        assert np.all(w_out[mask] == 1.0)

    def test_apply_ccd_mask_no_weights(self):
        """apply_ccd_mask with weights=None must return None for weights."""
        rf = self._make_rf_metres(50)
        _, w_out, _ = apply_ccd_mask(rf, weights=None, ccd_shape_m=(10e-3, 12e-3))
        assert w_out is None

    # ── Diagnostic with ccd_shape_m (post-RTM mm coordinates) ──

    def test_diagnostic_ccd_reduces_histogram(self):
        """CCD mask must reduce the histogram sum compared to no mask."""
        # Simulate post-RTM output: construct Diagnostic from mm-scale rf
        # by manually assigning self.rf in mm after init.
        rf_m = self._make_rf_metres(500)
        kw = dict(L=100, R=500, Lx=50, Ly=50)

        diag_no_ccd = Diagnostic(rf_m, **kw)
        diag_ccd    = Diagnostic(rf_m, ccd_shape_m=(13.5e-3, 18e-3), **kw)

        # Simulate RTM solve output: overwrite self.rf with mm coordinates.
        rf_mm = self._make_rf_mm(500)
        diag_no_ccd.rf = rf_mm.copy()
        diag_ccd.rf    = rf_mm.copy()

        diag_no_ccd.histogram(pix_x=20, pix_y=20)
        diag_ccd.histogram(pix_x=20, pix_y=20)
        assert diag_ccd.H.sum() < diag_no_ccd.H.sum(), (
            "CCD mask must exclude some rays and reduce histogram sum"
        )

    def test_diagnostic_ccd_with_weights(self):
        """CCD mask must apply consistently to both geometry and weights."""
        rf_m = self._make_rf_metres(300)
        weights = np.full(300, 0.8)
        kw = dict(L=100, R=500, Lx=50, Ly=50)

        diag = Diagnostic(rf_m, weights=weights, ccd_shape_m=(13.5e-3, 18e-3), **kw)
        rf_mm = self._make_rf_mm(300)
        diag.rf = rf_mm.copy()
        diag.histogram(pix_x=20, pix_y=20)
        assert diag.H.sum() > 0

        diag_plain = Diagnostic(rf_m, ccd_shape_m=(13.5e-3, 18e-3), **kw)
        diag_plain.rf = rf_mm.copy()
        diag_plain.histogram(pix_x=20, pix_y=20)
        T = transmission_map(diag.H, diag_plain.H)
        occupied = ~np.isnan(T) & (T > 0)
        if occupied.any():
            np.testing.assert_allclose(T[occupied], 0.8, atol=1e-9)

    def test_diagnostic_ccd_none_is_noop(self):
        """ccd_shape_m=None must produce identical histogram to default."""
        rf = self._make_rf_metres(100)
        kw = dict(L=100, R=500, Lx=50, Ly=50)
        diag_default = Diagnostic(rf, **kw)
        diag_none    = Diagnostic(rf, ccd_shape_m=None, **kw)
        diag_default.histogram(pix_x=20, pix_y=20)
        diag_none.histogram(pix_x=20, pix_y=20)
        np.testing.assert_array_equal(diag_default.H, diag_none.H)

    def test_ccd_default_shape(self):
        """CCD shape (Lx=13.5mm, Ly=18mm) must clip correctly."""
        # Standalone helper works in metres
        rf = np.zeros((4, 4))
        rf[0] = [0, 5e-3, 8e-3, 10e-3]   # x positions in metres
        rf[2] = [0, 5e-3, 8e-3, 10e-3]   # y positions
        ccd = (13.5e-3, 18e-3)   # full size (Lx, Ly); half: 6.75mm, 9mm
        _, _, mask = apply_ccd_mask(rf, ccd_shape_m=ccd)
        # Ray 0: (0,0) → inside
        # Ray 1: (5mm, 5mm) → inside (|5e-3| < 6.75e-3 and |5e-3| < 9e-3)
        # Ray 2: (8mm, 8mm) → outside (|8e-3| > 6.75e-3)
        # Ray 3: (10mm, 10mm) → outside
        np.testing.assert_array_equal(mask, [True, True, False, False])


# ---------------------------------------------------------------------------
# TestAbsorptionSanityCheck
# ---------------------------------------------------------------------------

class TestAbsorptionSanityCheck:
    """
    Tests for the absorption_sanity_check diagnostic utility.
    """

    def test_returns_expected_keys(self):
        """absorption_sanity_check must return all expected info keys."""
        info, fig, _ = absorption_sanity_check(
            ne=1e25, Te=100.0, Z=1.0, lwl=1064e-9, depth=2e-3)
        assert 'kappa' in info
        assert 'tau' in info
        assert 'amplitude' in info
        assert 'coulomb_log' in info
        plt.close(fig)

    def test_uniform_plasma_matches_formula(self):
        """Sanity check output must match direct kappa calculation."""
        ne, Te, Z = 1e25, 100.0, 1.0
        lwl = 1064e-9
        depth = 2e-3
        omega = 2 * np.pi * c / lwl
        kappa_ref = float(kappa_inv_brems(
            jnp.float32(ne), jnp.float32(Te), float(Z), omega))
        tau_ref = kappa_ref * depth / c
        info, fig, _ = absorption_sanity_check(ne, Te, Z, lwl, depth)
        np.testing.assert_allclose(info['kappa'], kappa_ref, rtol=1e-3)
        np.testing.assert_allclose(info['tau'], tau_ref, rtol=1e-3)
        plt.close(fig)

    def test_coulomb_log_clamped_at_low_Te(self):
        """At low Te, Coulomb log must be clamped to 2."""
        info, fig, _ = absorption_sanity_check(
            ne=1e25, Te=5.0, Z=1.0, lwl=1064e-9, depth=2e-3)
        assert info['coulomb_log'] == 2.0
        plt.close(fig)

    def test_high_density_strong_absorption(self):
        """At ne=1e26, tau must be >> 1 for moderate Te."""
        info, fig, _ = absorption_sanity_check(
            ne=1e26, Te=10.0, Z=1.0, lwl=1064e-9, depth=10e-3)
        assert info['tau'] > 1.0, f"Expected strong absorption at ne=1e26, got tau={info['tau']}"
        plt.close(fig)


# ---------------------------------------------------------------------------
# TestCCDAcceptance — arrangement-derived CCD acceptance
# ---------------------------------------------------------------------------

class TestCCDAcceptance:
    """
    Tests for arrangement-derived CCD acceptance bounds.

    The angular acceptance of a CCD is not a generic global constant — it
    depends on the specific optical arrangement (RTM) of each diagnostic.

    For an imaging axis (B ≈ 0): position maps to position with magnification
    M = A, and the CCD does not constrain the angle.

    For an angular axis (A ≈ 0): angle maps to position through the scale B,
    so the CCD constrains  |angle| ≤ CCD_half / |B|.

    Different diagnostics (shadowgraphy vs refractometry) have different RTMs
    and therefore different angular acceptances for the same CCD.
    """

    # -- RTM computation helpers --

    def test_rtm_travel_identity(self):
        """Travel(0) must be the identity matrix."""
        rtm_x, rtm_y = compute_arrangement_rtm([('travel', 0)])
        np.testing.assert_array_almost_equal(rtm_x, np.eye(2))
        np.testing.assert_array_almost_equal(rtm_y, np.eye(2))

    def test_rtm_single_lens_imaging(self):
        """Single lens at 2f-2f must give magnification -1, B=0."""
        f = 200
        ops = [('travel', 2 * f), ('lens', f), ('travel', 2 * f)]
        rtm_x, _ = compute_arrangement_rtm(ops)
        np.testing.assert_allclose(rtm_x[0, 0], -1.0, atol=1e-12)
        np.testing.assert_allclose(rtm_x[0, 1],  0.0, atol=1e-12)

    def test_rtm_shadowgraphy_two_lens_M1(self):
        """Shadowgraphy two-lens telescope (fp=0) must give M=1, B=0."""
        L = 400
        ops = [('travel', L), ('lens', L / 2), ('travel', 2 * L),
               ('lens', L / 2), ('travel', L)]
        rtm_x, rtm_y = compute_arrangement_rtm(ops)
        np.testing.assert_allclose(rtm_x[0, 0],  1.0, atol=1e-12)
        np.testing.assert_allclose(rtm_x[0, 1],  0.0, atol=1e-12)
        np.testing.assert_array_almost_equal(rtm_x, rtm_y)

    def test_rtm_shadowgraphy_single_lens_M2(self):
        """Shadowgraphy single-lens (fp=0) must give M=-2, B=0."""
        L = 400
        ops = [('travel', 3 * L / 4), ('lens', L / 2),
               ('travel', 3 * L / 2)]
        rtm_x, _ = compute_arrangement_rtm(ops)
        np.testing.assert_allclose(rtm_x[0, 0], -2.0, atol=1e-12)
        np.testing.assert_allclose(rtm_x[0, 1],  0.0, atol=1e-12)

    def test_rtm_schlieren_4f_system(self):
        """Schlieren 4f system (fp=0) must give M=-1, B=0."""
        L = 400
        ops = [('travel', L), ('lens', L), ('travel', 2 * L),
               ('lens', L), ('travel', L)]
        rtm_x, _ = compute_arrangement_rtm(ops)
        np.testing.assert_allclose(rtm_x[0, 0], -1.0, atol=1e-12)
        np.testing.assert_allclose(rtm_x[0, 1],  0.0, atol=1e-12)

    def test_rtm_refractometry_y_axis_angular(self):
        """Refractometry y-axis (fp=0) must be a pure angular axis (A=0)."""
        L = 400
        ops_y = [('travel', 3 * L / 4), ('lens', L / 2),
                 ('travel', 3 * L / 2), ('lens', L / 2), ('travel', L)]
        _, rtm_y = compute_arrangement_rtm(ops_y, ops_y)
        np.testing.assert_allclose(rtm_y[0, 0], 0.0, atol=1e-12,
                                   err_msg="A_y should be 0 for angular axis")
        assert abs(rtm_y[0, 1]) > 1.0, "B_y should be large for angular axis"

    def test_rtm_refractometry_x_axis_imaging(self):
        """Refractometry custom-solve x-axis (fp=0) must image with M=2."""
        f1, f3, img_f1_dist, img_dist = 200, 200, 600, 400
        ops_x = [('travel', 2 * f1), ('lens', f1),
                 ('travel', img_f1_dist),
                 ('lens', (2 * f3) / 3), ('travel', img_dist)]
        rtm_x, _ = compute_arrangement_rtm(ops_x)
        np.testing.assert_allclose(rtm_x[0, 0], 2.0, atol=1e-12,
                                   err_msg="M_x should be 2")
        np.testing.assert_allclose(rtm_x[0, 1], 0.0, atol=1e-12,
                                   err_msg="B_x should be 0 for imaging")

    # -- CCD spatial bounds --

    def test_ccd_spatial_bounds_imaging(self):
        """Spatial bound for imaging axis: x_max = CCD_half / |M|."""
        ccd = (18.0, 13.5)  # mm
        rtm_x = np.array([[-2.0, 0.0], [0.0, -0.5]])  # M=-2
        rtm_y = np.array([[ 1.0, 0.0], [0.0,  1.0]])   # M=+1
        x_max, y_max = compute_ccd_bounds_spatial(ccd, rtm_x, rtm_y)
        assert x_max == pytest.approx(9.0 / 2.0)   # 18/2 / 2
        assert y_max == pytest.approx(13.5 / 2.0)   # 13.5/2 / 1

    def test_ccd_spatial_bounds_angular_axis(self):
        """Spatial bound for pure angular axis (A=0): should return None."""
        ccd = (18.0, 13.5)
        rtm_x = np.array([[2.0, 0.0], [0.0, 0.5]])
        rtm_y = np.array([[0.0, -200.0], [1.0/200, 0.0]])  # A_y=0
        x_max, y_max = compute_ccd_bounds_spatial(ccd, rtm_x, rtm_y)
        assert x_max is not None
        assert y_max is None, "A_y=0 means no spatial bound on y"

    # -- CCD angular bounds --

    def test_ccd_angular_bounds_pure_angular(self):
        """Angular bound: phi_max = CCD_half_y / |B_y|."""
        ccd = (18.0, 13.5)  # mm
        B_y = -200.0  # mm / rad
        rtm_x = np.array([[2.0, 0.0], [0.0, 0.5]])
        rtm_y = np.array([[0.0, B_y], [1.0/200, 0.0]])
        theta_max, phi_max = compute_ccd_bounds_angular(ccd, rtm_x, rtm_y)
        assert theta_max is None, "B_x=0 → no angular bound on theta"
        assert phi_max == pytest.approx(13.5 / 2.0 / 200.0)

    def test_ccd_angular_bounds_imaging_axis(self):
        """For pure imaging axis (B=0), angular bound must be None."""
        ccd = (18.0, 13.5)
        rtm = np.array([[-2.0, 0.0], [0.0, -0.5]])
        theta_max, phi_max = compute_ccd_bounds_angular(ccd, rtm, rtm)
        assert theta_max is None
        assert phi_max is None

    # -- Arrangement dependence --

    def test_angular_bound_changes_with_arrangement(self):
        """Different arrangements must give different angular acceptances."""
        ccd = (18.0, 13.5)
        # Arrangement A: B_y = -200 mm/rad
        rtm_a = np.array([[0.0, -200.0], [1.0/200, 0.0]])
        # Arrangement B: B_y = -100 mm/rad (shorter effective focal length)
        rtm_b = np.array([[0.0, -100.0], [1.0/100, 0.0]])
        rtm_dummy = np.eye(2)

        _, phi_a = compute_ccd_bounds_angular(ccd, rtm_dummy, rtm_a)
        _, phi_b = compute_ccd_bounds_angular(ccd, rtm_dummy, rtm_b)

        assert phi_a != phi_b, "Different arrangements must give different bounds"
        assert phi_b == pytest.approx(2 * phi_a), \
            "Halving B should double the angular acceptance"

    def test_angular_bound_matches_reconstruction(self):
        """CCD phi_max must equal CCD_half_y / |B_y| from the RTM."""
        L = 400
        ccd = (18.0, 13.5)
        # Compute RTM for refractometry y-axis (standard solve, fp=0)
        ops_y = [('travel', 3 * L / 4), ('lens', L / 2),
                 ('travel', 3 * L / 2), ('lens', L / 2), ('travel', L)]
        _, rtm_y = compute_arrangement_rtm(ops_y, ops_y)
        B_y = rtm_y[0, 1]  # angle-to-position scale
        expected_phi_max = (ccd[1] / 2.0) / abs(B_y)
        _, phi_max = compute_ccd_bounds_angular(ccd, np.eye(2), rtm_y)
        assert phi_max == pytest.approx(expected_phi_max)

    def test_shadowgraphy_vs_refractometry_different_bounds(self):
        """Shadowgraphy and refractometry must have different angular bounds."""
        L = 400
        ccd = (18.0, 13.5)

        # Shadowgraphy two-lens: symmetric, B=0 both axes → no angular bound
        ops_shad = [('travel', L), ('lens', L / 2), ('travel', 2 * L),
                    ('lens', L / 2), ('travel', L)]
        rtm_shad_x, rtm_shad_y = compute_arrangement_rtm(ops_shad)
        th_shad, phi_shad = compute_ccd_bounds_angular(ccd, rtm_shad_x, rtm_shad_y)

        # Refractometry: y-axis is angular (A=0, B≠0)
        ops_ref_x = [('travel', 3 * L / 4), ('lens', L / 2),
                     ('travel', 3 * L / 2), ('lens', L / 3), ('travel', L)]
        ops_ref_y = [('travel', 3 * L / 4), ('lens', L / 2),
                     ('travel', 3 * L / 2), ('lens', L / 2), ('travel', L)]
        rtm_ref_x, rtm_ref_y = compute_arrangement_rtm(ops_ref_x, ops_ref_y)
        th_ref, phi_ref = compute_ccd_bounds_angular(ccd, rtm_ref_x, rtm_ref_y)

        assert phi_shad is None, "Shadowgraphy B_y=0 → no angular bound"
        assert phi_ref is not None, "Refractometry B_y≠0 → has angular bound"

    # -- build_ccd_acceptance_mask --

    def test_build_mask_clips_positions(self):
        """build_ccd_acceptance_mask must clip on detector positions."""
        rf = np.zeros((4, 5))
        rf[0] = [0, 3, 6, 10, -10]   # x in mm
        rf[2] = [0, 3, 6, 10, -10]   # y in mm
        ccd = (12.0, 12.0)  # ±6 mm
        mask, info = build_ccd_acceptance_mask(rf, ccd_size_mm=ccd)
        np.testing.assert_array_equal(mask, [True, True, True, False, False])
        assert info['n_accepted'] == 3
        assert info['n_rejected'] == 2

    def test_build_mask_with_rtm_reports_angular_bounds(self):
        """When RTMs are supplied, info must contain angular bounds."""
        rf = np.zeros((4, 10))
        ccd = (18.0, 13.5)
        rtm_x = np.array([[2.0, 0.0], [0.0, 0.5]])
        rtm_y = np.array([[0.0, -200.0], [1.0/200, 0.0]])
        _, info = build_ccd_acceptance_mask(
            rf, ccd_size_mm=ccd, rtm_x=rtm_x, rtm_y=rtm_y)
        assert 'theta_max_rad' in info
        assert 'phi_max_rad' in info
        assert info['theta_max_rad'] is None  # imaging axis
        assert info['phi_max_rad'] == pytest.approx(13.5 / 2.0 / 200.0)

    def test_build_mask_nan_handling(self):
        """NaN positions must be rejected."""
        rf = np.zeros((4, 3))
        rf[0, 1] = np.nan
        ccd = (20.0, 20.0)
        mask, _ = build_ccd_acceptance_mask(rf, ccd_size_mm=ccd)
        assert mask[0] and not mask[1] and mask[2]

    # -- Integration with Diagnostic subclasses --

    def _make_rf_m(self, N=200):
        """(4, N) Jones vector in metres — small angles, positions within ±2 mm."""
        rng = np.random.RandomState(99)
        rf = np.zeros((4, N))
        rf[0] = (rng.rand(N) - 0.5) * 4e-3   # x ±2 mm
        rf[2] = (rng.rand(N) - 0.5) * 4e-3   # y ±2 mm
        rf[1] = rng.randn(N) * 1e-4
        rf[3] = rng.randn(N) * 1e-4
        return rf

    def test_shadowgraphy_stores_rtm(self):
        """Shadowgraphy.two_lens_solve must store RTMs on the instance."""
        rf = self._make_rf_m()
        diag = Shadowgraphy(rf, L=400, R=500)
        diag.two_lens_solve()
        assert diag._rtm_x is not None
        assert diag._rtm_y is not None

    def test_refractometry_stores_rtm(self):
        """Refractometry.incoherent_solve must store RTMs on the instance."""
        rf = self._make_rf_m()
        diag = Refractometry(rf, L=400, R=500)
        diag.incoherent_solve()
        assert diag._rtm_x is not None
        assert diag._rtm_y is not None
        # y-axis should be angular (A≈0)
        np.testing.assert_allclose(diag._rtm_y[0, 0], 0.0, atol=1e-10)

    def test_schlieren_stores_rtm(self):
        """Schlieren.DF_solve must store RTMs on the instance."""
        rf = self._make_rf_m()
        diag = Schlieren(rf, L=400, R=500)
        diag.DF_solve()
        assert diag._rtm_x is not None
        np.testing.assert_allclose(diag._rtm_x[0, 0], -1.0, atol=1e-10)

    def test_ccd_acceptance_info_returns_bounds(self):
        """ccd_acceptance_info must report arrangement-derived bounds."""
        rf = self._make_rf_m()
        ccd = (18e-3, 13.5e-3)
        diag = Refractometry(rf, L=400, R=500, ccd_shape_m=ccd)
        diag.incoherent_solve()
        diag.histogram(pix_x=20, pix_y=20)
        info = diag.ccd_acceptance_info()
        assert info['ccd_enabled']
        assert info.get('phi_max_rad') is not None
        assert info.get('theta_max_rad') is not None  # B_x≠0 for standard solve

    def test_mask_applied_consistently_to_weights(self):
        """CCD mask must clip geometry and weights identically."""
        rf = self._make_rf_m(300)
        weights = np.full(300, 0.75)
        ccd = (18e-3, 13.5e-3)

        diag_wt = Shadowgraphy(rf, weights=weights,
                               L=400, R=500, Lx=50, Ly=50,
                               ccd_shape_m=ccd)
        diag_wt.two_lens_solve()
        diag_wt.histogram(pix_x=20, pix_y=20)

        diag_plain = Shadowgraphy(rf, L=400, R=500, Lx=50, Ly=50,
                                  ccd_shape_m=ccd)
        diag_plain.two_lens_solve()
        diag_plain.histogram(pix_x=20, pix_y=20)

        T = transmission_map(diag_wt.H, diag_plain.H)
        occupied = ~np.isnan(T) & (T > 0)
        if occupied.any():
            np.testing.assert_allclose(T[occupied], 0.75, atol=1e-9)

    def test_focal_plane_changes_rtm(self):
        """Non-zero focal_plane must change the RTM B element."""
        L = 400
        rf = self._make_rf_m()

        diag0 = Shadowgraphy(rf, L=L, R=500, focal_plane=0)
        diag0.two_lens_solve()
        B0 = diag0._rtm_x[0, 1]

        diag1 = Shadowgraphy(rf, L=L, R=500, focal_plane=10)
        diag1.two_lens_solve()
        B1 = diag1._rtm_x[0, 1]

        np.testing.assert_allclose(B0, 0.0, atol=1e-12,
                                   err_msg="fp=0 should give B=0 (perfect focus)")
        assert abs(B1) > 0.1, \
            "fp≠0 should give B≠0 (defocus introduces angular coupling)"

    def test_verbose_output(self, capsys):
        """build_ccd_acceptance_mask verbose mode must print summary."""
        rf = np.zeros((4, 10))
        ccd = (18.0, 13.5)
        rtm_x = np.eye(2)
        rtm_y = np.array([[0.0, -200.0], [1.0/200, 0.0]])
        build_ccd_acceptance_mask(
            rf, ccd_size_mm=ccd, rtm_x=rtm_x, rtm_y=rtm_y, verbose=True)
        captured = capsys.readouterr().out
        assert "CCD acceptance:" in captured
        assert "phi acceptance" in captured.lower() or "Phi" in captured