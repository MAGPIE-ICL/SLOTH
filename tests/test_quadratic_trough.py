"""
Test laser propagation through a quadratic electron density trough.

Analytic test problem:
    ne(y) = ncr/2 * (1 + y^2/yc^2)

A ray entering at y0 (< yc) propagating in x undergoes:
    x(tau) = tau * sqrt(1 - ne(y0)/ncr)
    y(tau) = y0 * cos(tau / (sqrt(2) * yc))

where tau = c * t is the vacuum path-length parameter,
c is the speed of light, and t is the physical time.

The critical density is related to the laser angular frequency by:
    ncr = m_e * epsilon_0 * omega^2 / e^2

Tests are parametrized over multiple laser wavelengths (351 nm, 702 nm,
1053 nm) and tolerance levels to exercise the solver across a range of
physically relevant conditions.
"""

import sys
import os

# Ensure src/ is on the path for relative imports used by the codebase.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
import jax.numpy as jnp
from scipy.constants import c, m_e, e, epsilon_0


# ---------------------------------------------------------------------------
# Physical helpers
# ---------------------------------------------------------------------------

def _critical_density(lwl):
    """Return the critical electron density (m^-3) for laser wavelength *lwl*.

    ncr = m_e * epsilon_0 * omega^2 / e^2
    """
    omega = 2.0 * np.pi * c / lwl
    return omega ** 2 * m_e * epsilon_0 / e ** 2

def _f_analy(x, y0):
    # See https://journals.aps.org/pre/pdf/10.1103/PhysRevE.61.895
    ne0 = (1.0 / 2.0) * (1.0 + y0 ** 2 / YC ** 2)
    vy0 = np.sqrt(1.-ne0)
    tau = 2.0*np.pi*YC/np.sqrt(0.5)
    return y0*np.cos(2.*np.pi*(x+X_LENGTH/2)/(vy0*tau))

# ---------------------------------------------------------------------------
# Domain / ray builders
# ---------------------------------------------------------------------------

def _make_quadratic_trough_domain(x_length, y_length, z_length,
                                   nx, ny, nz, ncr, yc, lwl):
    """Build a Domain whose refractive index corresponds to the quadratic
    electron density profile ne(y) = ncr/2 * (1 + y^2 / yc^2)."""
    from core.domain import Domain

    lengths = jnp.array([x_length, y_length, z_length])
    dims = jnp.array([nx, ny, nz])

    x = jnp.linspace(-x_length / 2, x_length / 2, nx)
    y = jnp.linspace(-y_length / 2, y_length / 2, ny)
    z = jnp.linspace(-z_length / 2, z_length / 2, nz)
    _, YY, _ = jnp.meshgrid(x, y, z, indexing='ij')
    ne = (ncr / 2.0) * (1.0 + YY ** 2 / yc ** 2)

    return Domain.from_ne(ne, x, y, z, lengths, dims, lwl)


def _make_ray(x0, y0, vx0):
    """Return a (6, 1) state vector for a single ray at (x0, y0, 0)
    with velocity (vx0, 0, 0)."""
    s0 = jnp.zeros((6, 1))
    s0 = s0.at[0, 0].set(x0)
    s0 = s0.at[1, 0].set(y0)
    s0 = s0.at[3, 0].set(vx0)
    return s0


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Default domain geometry shared by all tests.
X_LENGTH = 0.50   # 50 cm
Y_LENGTH = 0.10   # 10 cm (±5 cm)
Z_LENGTH = 0.02   #  2 cm (thin; essentially 2-D)
NX, NY, NZ = 16, 128, 4
YC = 0.05         # trough half-width (m) – 5 cm
PROBING_DEPTH = 0.10  # 10 cm trace depth


# ---------------------------------------------------------------------------
# Tolerance presets
# ---------------------------------------------------------------------------

class Tolerances:
    """Bundle of position / velocity tolerances for a single test case."""
    def __init__(self, pos_atol, pos_rtol):
        self.pos_atol = pos_atol
        self.pos_rtol = pos_rtol

    def __repr__(self):
        return (f"Tolerances(pos_atol={self.pos_atol}, pos_rtol={self.pos_rtol})")

TOLERANCE_PRESETS = {
    "standard": Tolerances(
        pos_atol=2e-4,   # 0.2 mm
        pos_rtol=0.01,   # 1 %
    ),
    "tight": Tolerances(
        pos_atol=5e-5,   # 0.05 mm
        pos_rtol=0.005,  # 0.5 %
    ),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[351e-9, 702e-9, 1053e-9],
                ids=["351nm", "702nm", "1053nm"])
def wavelength(request):
    """Laser wavelength in metres."""
    return request.param


@pytest.fixture
def ncr(wavelength):
    """Critical electron density for the current wavelength."""
    return _critical_density(wavelength)


@pytest.fixture
def domain(ncr, wavelength):
    """Domain with quadratic trough profile for the current ncr."""
    return _make_quadratic_trough_domain(
        X_LENGTH, Y_LENGTH, Z_LENGTH, NX, NY, NZ, ncr, YC, wavelength,
    )

# ---------------------------------------------------------------------------
# Single-ray parametrized test
# ---------------------------------------------------------------------------

# Pair each wavelength with appropriate tolerance levels.
_SINGLE_RAY_CASES = [
    pytest.param(351e-9,  "tight",    id="351nm-tol-tight"),
    pytest.param(351e-9,  "standard", id="351nm-tol-standard"),
    pytest.param(702e-9,  "tight", id="702nm-tol-tight"),
    pytest.param(702e-9,  "standard",  id="702nm-tol-standard"),
    pytest.param(1053e-9, "tight", id="1053nm-tol-tight"),
    pytest.param(1053e-9, "standard",  id="1053nm-tol-standard"),
]


@pytest.mark.parametrize("lwl, tol_name", _SINGLE_RAY_CASES)
def test_single_ray(lwl, tol_name):
    """Propagate a single ray and compare against the analytic trajectory.

    Parametrized over wavelength (351 nm, 702 nm, 1053 nm) and tolerance
    level (tight, standard, relaxed).
    """
    from core.propagator import Propagator

    ncr = _critical_density(lwl)
    tol = TOLERANCE_PRESETS[tol_name]

    domain = _make_quadratic_trough_domain(
        X_LENGTH, Y_LENGTH, Z_LENGTH, NX, NY, NZ, ncr, YC, lwl,
    )

    y0 = 0.01  # 1 cm offset
    ne_y0 = (ncr / 2.0) * (1.0 + y0 ** 2 / YC ** 2)
    n_y0 = float(np.sqrt(1.0 - ne_y0 / ncr))
    vx0 = c * n_y0

    s0 = _make_ray(-X_LENGTH / 2.0, y0, vx0)

    prop = Propagator(domain, PROBING_DEPTH)
    sol = prop(s0)

    final = np.asarray(sol.ys[0, -1, :])
    x_num, y_num = float(final[0]), float(final[1])

    y_ana = _f_analy(x_num, y0)

    assert abs(y_num - y_ana) < max(tol.pos_atol, abs(y_ana) * tol.pos_rtol), \
        f"y mismatch (lwl={lwl:.0e}): num={y_num:.6e}, ana={y_ana:.6e}"


# ---------------------------------------------------------------------------
# Multi-ray parametrized test
# ---------------------------------------------------------------------------

_MULTI_RAY_CASES = [
    pytest.param(351e-9,  "tight", id="351nm-tol-tight"),
    pytest.param(351e-9,  "standard",  id="351nm-tol-standard"),
    pytest.param(702e-9,  "tight", id="702nm-tol-tight"),
    pytest.param(702e-9,  "standard",  id="702nm-tol-standard"),
    pytest.param(1053e-9, "standard",  id="1053nm-tol-standard"),
    pytest.param(1053e-9, "tight",  id="1053nm-tol-tight"),
]

Y0_VALUES = [0.005, 0.01, 0.02, 0.03]


@pytest.mark.parametrize("lwl, tol_name", _MULTI_RAY_CASES)
def test_multiple_rays(lwl, tol_name):
    """Propagate several rays at different y0 offsets and verify they
    all match the analytic solution.

    Parametrized over wavelength and tolerance level.
    """
    from core.propagator import Propagator

    ncr = _critical_density(lwl)
    tol = TOLERANCE_PRESETS[tol_name]

    domain = _make_quadratic_trough_domain(
        X_LENGTH, Y_LENGTH, Z_LENGTH, NX, NY, NZ, ncr, YC, lwl,
    )

    Np = len(Y0_VALUES)
    s0 = jnp.zeros((6, Np))
    for j, y0 in enumerate(Y0_VALUES):
        ne_y0 = (ncr / 2.0) * (1.0 + y0 ** 2 / YC ** 2)
        n_y0 = float(np.sqrt(1.0 - ne_y0 / ncr))
        s0 = s0.at[0, j].set(-X_LENGTH / 2.0)
        s0 = s0.at[1, j].set(y0)
        s0 = s0.at[3, j].set(c * n_y0)

    prop = Propagator(domain, PROBING_DEPTH)
    sol = prop(s0)

    for j, y0 in enumerate(Y0_VALUES):
        final = np.asarray(sol.ys[j, -1, :])
        y_num = float(final[1])
        x_num = float(final[0])
        y_ana = _f_analy(x_num, y0)

        err = abs(y_num - y_ana)
        assert err < max(tol.pos_atol, abs(y_ana) * tol.pos_rtol), \
            (f"y mismatch (lwl={lwl:.0e}, y0={y0}): "
             f"num={y_num:.6e}, ana={y_ana:.6e}, err={err:.2e}")


# ---------------------------------------------------------------------------
# Inverse bremsstrahlung test
# ---------------------------------------------------------------------------

_INV_BREMS_CASES = [
    pytest.param(351e-9,  id="351nm"),
    pytest.param(702e-9,  id="702nm"),
    pytest.param(1053e-9, id="1053nm"),
]


class _MinimalDomainIB:
    """Lightweight domain stub with inv_brems fields for trace_and_save_depths."""

    def __init__(self, ne, x, y, z, Te, Z, probing_direction='x'):
        self.ne = jnp.array(ne, dtype=jnp.float32)
        self.x  = jnp.array(x,  dtype=jnp.float32)
        self.y  = jnp.array(y,  dtype=jnp.float32)
        self.z  = jnp.array(z,  dtype=jnp.float32)
        self.probing_direction = probing_direction
        self.lengths = jnp.array(
            [x[-1] - x[0], y[-1] - y[0], z[-1] - z[0]], dtype=jnp.float32
        )
        self.dims = jnp.array([len(x), len(y), len(z)], dtype=jnp.int32)
        self.Np_total = None

        self.inv_brems = True
        self.Te = jnp.asarray(Te, dtype=jnp.float32)
        self.Z  = jnp.asarray(Z,  dtype=jnp.float32)


@pytest.mark.parametrize("lwl", _INV_BREMS_CASES)
def test_inv_brems_quadratic_trough(lwl):
    """Verify that inverse bremsstrahlung attenuates amplitude correctly
    in the quadratic trough.

    Physics checks:
      1. Amplitude at entry depth (t=0) is 1.0 (no absorption yet).
      2. Amplitude at exit is strictly less than 1 (absorption occurred).
      3. Amplitude is positive (no underflow / sign flips).
      4. jvec_weighted = jvec_unweighted * amplitude (consistency).
      5. kappa_inv_brems is non-negative everywhere.

    Te=100keV, Z=1 keeps absorption moderate (tau ~ 0.1–1) so amplitude
    is measurably attenuated but doesn't underflow to zero.
    """
    from core.propagator import trace_and_save_depths, kappa_inv_brems

    ncr  = _critical_density(lwl)
    omega = 2.0 * np.pi * c / lwl

    # Build ne grid (same quadratic trough)
    x = np.linspace(-X_LENGTH / 2, X_LENGTH / 2, NX)
    y = np.linspace(-Y_LENGTH / 2, Y_LENGTH / 2, NY)
    z = np.linspace(-Z_LENGTH / 2, Z_LENGTH / 2, NZ)
    _, YY, _ = np.meshgrid(x, y, z, indexing='ij')
    ne = (ncr / 2.0) * (1.0 + YY ** 2 / YC ** 2)

    # Plasma parameters: high Te + low Z keeps kappa moderate so
    # the optical depth over 10 cm doesn't cause underflow.
    Te_eV = 1e5     # 100 keV electron temperature
    Z_ion = 1.0     # hydrogen

    domain_ib = _MinimalDomainIB(ne, x, y, z, Te=Te_eV, Z=Z_ion,
                                  probing_direction='x')

    # Single ray at y0 = 1 cm
    y0 = 0.01
    ne_y0 = (ncr / 2.0) * (1.0 + y0 ** 2 / YC ** 2)
    n_y0 = float(np.sqrt(1.0 - ne_y0 / ncr))
    vx0 = c * n_y0

    s0 = jnp.zeros((6, 1))
    s0 = s0.at[0, 0].set(-X_LENGTH / 2.0)
    s0 = s0.at[1, 0].set(y0)
    s0 = s0.at[3, 0].set(vx0)

    depth_max = PROBING_DEPTH
    step = depth_max / 2  # save at 0, half, and full depth
    result = trace_and_save_depths(
        s0, domain_ib, step=step, depth_max=depth_max, output_path=None,
        lwl=lwl, jones_components='all', jitted=True, verbose=False,
    )

    # --- 1. Amplitude key present ---
    assert 'amplitude' in result, "inv_brems result must contain 'amplitude' key"
    assert 'jvec_unweighted' in result, "inv_brems result must contain 'jvec_unweighted'"

    # --- 2. Amplitude at entry is 1.0 ---
    amp_entry = result['amplitude'][0]
    np.testing.assert_allclose(amp_entry, 1.0, atol=1e-6,
                               err_msg="amplitude at entry should be 1.0")

    # --- 3. Amplitude at exit is 0 < a < 1 ---
    amp_final = result['amplitude'][-1]
    assert np.all(amp_final > 0), \
        f"amplitude must be positive (got {amp_final})"
    assert np.all(amp_final < 1.0), \
        f"amplitude must be < 1 (got {float(amp_final[0]):.6f})"

    # --- 4. Monotonic decay: amplitude at mid-depth >= amplitude at exit ---
    amp_mid = result['amplitude'][1]
    assert np.all(amp_mid >= amp_final - 1e-7), \
        "amplitude must decrease monotonically with depth"

    # --- 5. Weighted = unweighted * amplitude ---
    for j in range(len(result['jvec'])):
        jvec_w  = result['jvec'][j]
        jvec_uw = result['jvec_unweighted'][j]
        amp_j   = result['amplitude'][j]
        expected = jvec_uw * amp_j[np.newaxis, :]
        np.testing.assert_allclose(jvec_w, expected, atol=1e-10,
                                   err_msg=f"jvec mismatch at depth index {j}")

    # --- 6. kappa is non-negative ---
    kappa = kappa_inv_brems(
        jnp.asarray(ne, dtype=jnp.float32),
        jnp.float32(Te_eV),
        jnp.float32(Z_ion),
        omega,
    )
    assert np.all(np.asarray(kappa) >= 0), "kappa_inv_brems must be non-negative"
