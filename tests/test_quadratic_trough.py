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
                                   nx, ny, nz, ncr, yc):
    """Build a ScalarDomain whose electron density follows
    ne(y) = ncr/2 * (1 + y^2 / yc^2)."""
    from core.domain import ScalarDomain

    lengths = jnp.array([x_length, y_length, z_length])
    dims = jnp.array([nx, ny, nz])

    y = jnp.linspace(-y_length / 2, y_length / 2, ny)
    _, YY, _ = jnp.meshgrid(
        jnp.linspace(-x_length / 2, x_length / 2, nx),
        y,
        jnp.linspace(-z_length / 2, z_length / 2, nz),
        indexing='ij',
    )
    ne = (ncr / 2.0) * (1.0 + YY ** 2 / yc ** 2)

    domain = ScalarDomain(
        lengths, dims,
        ne_type="import",
        probing_direction='x',
        auto_batching=False,
        ne=ne,
    )
    return domain


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
def domain(ncr):
    """ScalarDomain with quadratic trough profile for the current ncr."""
    return _make_quadratic_trough_domain(
        X_LENGTH, Y_LENGTH, Z_LENGTH, NX, NY, NZ, ncr, YC,
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
        X_LENGTH, Y_LENGTH, Z_LENGTH, NX, NY, NZ, ncr, YC,
    )

    y0 = 0.01  # 1 cm offset
    ne_y0 = (ncr / 2.0) * (1.0 + y0 ** 2 / YC ** 2)
    n_y0 = float(np.sqrt(1.0 - ne_y0 / ncr))
    vx0 = c * n_y0

    s0 = _make_ray(-X_LENGTH / 2.0, y0, vx0)

    prop = Propagator(domain, PROBING_DEPTH, lwl)
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
        X_LENGTH, Y_LENGTH, Z_LENGTH, NX, NY, NZ, ncr, YC,
    )

    Np = len(Y0_VALUES)
    s0 = jnp.zeros((6, Np))
    for j, y0 in enumerate(Y0_VALUES):
        ne_y0 = (ncr / 2.0) * (1.0 + y0 ** 2 / YC ** 2)
        n_y0 = float(np.sqrt(1.0 - ne_y0 / ncr))
        s0 = s0.at[0, j].set(-X_LENGTH / 2.0)
        s0 = s0.at[1, j].set(y0)
        s0 = s0.at[3, j].set(c * n_y0)

    prop = Propagator(domain, PROBING_DEPTH, lwl)
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
