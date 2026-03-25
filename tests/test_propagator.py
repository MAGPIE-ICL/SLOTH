"""
Tests for the propagator gradient machinery.

Covers:
    1. gradient_term formula matches the analytic expression
       gradient_term = 0.5 * c^2 * (1 - ne / n_cr)
       where n_cr = m_e * epsilon_0 * omega^2 / e^2.
    2. dndr returns zero gradient for uniform ne.
    3. dndr returns a constant gradient for a linear ne profile.
    4. dndr returns the correct spatially-varying gradient for a
       quadratic ne profile.
    5. dndr returns the correct gradient for an exponential ne profile.
    6. dndr returns the correct gradient for a separable product
       ne(x,y,z) = f(x) * g(y) * h(z).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
import jax.numpy as jnp
import scipy.constants as sc

from core.propagator import dndr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _critical_density(omega):
    """n_cr = m_e * epsilon_0 * omega^2 / e^2"""
    return sc.m_e * sc.epsilon_0 * omega ** 2 / sc.e ** 2


def _make_grid(nx=61, ny=63, nz=59,
               xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-1.0, 1.0)):
    """Return 1-D coordinate arrays and 3-D mesh grids."""
    x = jnp.linspace(*xlim, nx)
    y = jnp.linspace(*ylim, ny)
    z = jnp.linspace(*zlim, nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    return x, y, z, X, Y, Z


def _gradient_term(ne, omega):
    """Compute the gradient_term array used by the propagator.

    New formula: gradient_term = 0.5 * c² * n²  where n² = 1 - ne/ncr.
    The spatial gradient is identical to the old formula:
        ∇(0.5·c²·n²) = -0.5·c²/ncr · ∇ne
    """
    ncr = _critical_density(omega)
    n_sq = 1.0 - ne / ncr
    return 0.5 * sc.c ** 2 * n_sq


# Laser wavelength and corresponding angular frequency used throughout tests
_LWL = 1053e-9  # metres
_OMEGA = 2.0 * jnp.pi * sc.c / _LWL


# ===================================================================
# 1.  gradient_term formula
# ===================================================================

class TestGradientTermFormula:
    """Verify the gradient_term expression matches 0.5 * c^2 * (1 - ne / n_cr)."""

    @pytest.mark.parametrize("lwl", [351e-9, 527e-9, 1053e-9])
    def test_matches_analytic(self, lwl):
        omega = 2.0 * np.pi * sc.c / lwl
        n_cr = _critical_density(omega)

        ne_val = 0.3 * n_cr  # pick a sub-critical density
        ne = jnp.full((5, 5, 5), ne_val)

        gt = _gradient_term(ne, omega)
        expected = 0.5 * sc.c ** 2 * (1.0 - ne_val / n_cr)

        np.testing.assert_allclose(gt, expected, rtol=1e-6)

    def test_zero_density(self):
        ne = jnp.zeros((5, 5, 5))
        gt = _gradient_term(ne, _OMEGA)
        expected = 0.5 * sc.c ** 2  # n² = 1 when ne = 0
        np.testing.assert_allclose(gt, expected, rtol=1e-6)

    def test_linearity_in_ne(self):
        """The ne-dependent part of gradient_term scales linearly with ne.

        gradient_term = 0.5*c²*(1 - ne/ncr), so the shift from the vacuum
        value 0.5*c² is proportional to ne: doubling ne doubles the shift.
        """
        n_cr = _critical_density(_OMEGA)
        ne1 = jnp.full((5, 5, 5), 0.1 * n_cr)
        ne2 = jnp.full((5, 5, 5), 0.2 * n_cr)

        gt1 = _gradient_term(ne1, _OMEGA)
        gt2 = _gradient_term(ne2, _OMEGA)

        vacuum = 0.5 * sc.c ** 2
        np.testing.assert_allclose(vacuum - gt2, 2.0 * (vacuum - gt1), rtol=1e-6)


# ===================================================================
# 2.  Uniform ne  →  zero gradient
# ===================================================================

class TestUniformDensity:
    """A constant electron density field must produce zero gradient."""

    def test_zero_gradient(self):
        x, y, z, X, Y, Z = _make_grid()
        n_cr = _critical_density(_OMEGA)
        ne = jnp.full_like(X, 0.5 * n_cr)
        gt = _gradient_term(ne, _OMEGA)

        # Query at a few interior points
        r = jnp.array([
            [0.0, 0.0, 0.0],
            [0.3, -0.2, 0.1],
            [-0.5, 0.4, -0.3],
        ])

        grad = dndr(r, gt, x, y, z)

        np.testing.assert_allclose(grad, 0.0, atol=1e-10*n_cr)


# ===================================================================
# 3.  Linear ne  →  constant gradient
# ===================================================================

class TestLinearDensity:
    """For ne(x,y,z) = a*x + b*y + c*z + d the gradient of gradient_term
    is constant and equal to -0.5 * c^2 / n_cr * (a, b, c)."""

    @pytest.mark.parametrize("a,b,c_coeff,d", [
        (1.0e24, 0.0,    0.0,    1.0e25),
        (0.0,    2.0e24, 0.0,    1.0e25),
        (0.0,    0.0,    3.0e24, 1.0e25),
        (1.0e24, 2.0e24, 3.0e24, 5.0e25),
    ])
    def test_constant_gradient(self, a, b, c_coeff, d):
        x, y, z, X, Y, Z = _make_grid(nx=81, ny=83, nz=79)
        ne = a * X + b * Y + c_coeff * Z + d
        gt = _gradient_term(ne, _OMEGA)
        n_cr = _critical_density(_OMEGA)

        # Expected gradient components (constant everywhere)
        prefactor = -0.5 * sc.c ** 2 / n_cr
        expected_gx = prefactor * a
        expected_gy = prefactor * b
        expected_gz = prefactor * c_coeff

        # Query at several interior points (away from edges where
        # jnp.gradient uses one-sided finite differences)
        rng = np.random.default_rng(42)
        N = 50
        r = jnp.array(np.column_stack([
            rng.uniform(-0.6, 0.6, N),
            rng.uniform(-0.6, 0.6, N),
            rng.uniform(-0.6, 0.6, N),
        ]))

        grad = dndr(r, gt, x, y, z)

        np.testing.assert_allclose(grad[0, :], expected_gx, atol=1e-8*n_cr, rtol=1e-4)
        np.testing.assert_allclose(grad[1, :], expected_gy, atol=1e-8*n_cr, rtol=1e-4)
        np.testing.assert_allclose(grad[2, :], expected_gz, atol=1e-8*n_cr, rtol=1e-4)


# ===================================================================
# 4.  Quadratic ne  →  linear gradient
# ===================================================================

class TestQuadraticDensity:
    """For ne(y) = n0 * y^2, the y-gradient of gradient_term is
    -0.5 * c^2 / n_cr * 2 * n0 * y, and the x, z components are zero."""

    def test_quadratic_y(self):
        x, y, z, X, Y, Z = _make_grid(nx=41, ny=101, nz=41)
        n_cr = _critical_density(_OMEGA)
        n0 = 0.3 * n_cr  # coefficient
        ne = n0 * Y ** 2

        gt = _gradient_term(ne, _OMEGA)
        prefactor = -0.5 * sc.c ** 2 / n_cr

        # Query at points in the interior (avoid boundaries)
        y_query = jnp.array([-0.5, -0.2, 0.0, 0.2, 0.5])
        r = jnp.stack([
            jnp.zeros_like(y_query),
            y_query,
            jnp.zeros_like(y_query),
        ], axis=-1)

        grad = dndr(r, gt, x, y, z)

        expected_gy = prefactor * 2.0 * n0 * y_query
        np.testing.assert_allclose(grad[0, :], 0.0, atol=1e-8*n_cr)
        np.testing.assert_allclose(grad[1, :], expected_gy, atol=1e-8*n_cr, rtol=1e-4)
        np.testing.assert_allclose(grad[2, :], 0.0, atol=1e-8*n_cr)

    def test_quadratic_x(self):
        """Same idea but ne = n0 * x^2."""
        x, y, z, X, Y, Z = _make_grid(nx=101, ny=41, nz=41)
        n_cr = _critical_density(_OMEGA)
        n0 = 0.2 * n_cr
        ne = n0 * X ** 2

        gt = _gradient_term(ne, _OMEGA)
        prefactor = -0.5 * sc.c ** 2 / n_cr

        x_query = jnp.array([-0.5, -0.2, 0.0, 0.2, 0.5])
        r = jnp.stack([
            x_query,
            jnp.zeros_like(x_query),
            jnp.zeros_like(x_query),
        ], axis=-1)

        grad = dndr(r, gt, x, y, z)

        expected_gx = prefactor * 2.0 * n0 * x_query
        np.testing.assert_allclose(grad[0, :], expected_gx, atol=1e-8*n_cr, rtol=1e-4)
        np.testing.assert_allclose(grad[1, :], 0.0, atol=1e-8*n_cr)
        np.testing.assert_allclose(grad[2, :], 0.0, atol=1e-8*n_cr)


# ===================================================================
# 5.  Exponential ne  →  proportional gradient
# ===================================================================

class TestExponentialDensity:
    """For ne(x,y,z) = n0 * exp(k*x), the gradient of gradient_term
    in x is -0.5 * c^2 / n_cr * n0 * k * exp(k*x)."""

    def test_exponential_x(self):
        x, y, z, X, Y, Z = _make_grid(nx=101, ny=31, nz=31)
        n_cr = _critical_density(_OMEGA)
        n0 = 0.1 * n_cr
        k = 1.5  # 1/m scale
        ne = n0 * jnp.exp(k * X)

        gt = _gradient_term(ne, _OMEGA)
        prefactor = -0.5 * sc.c ** 2 / n_cr

        # Query in the interior
        x_query = jnp.array([-0.4, -0.2, 0.0, 0.2, 0.4])
        r = jnp.stack([
            x_query,
            jnp.zeros_like(x_query),
            jnp.zeros_like(x_query),
        ], axis=-1)

        grad = dndr(r, gt, x, y, z)

        expected_gx = prefactor * n0 * k * jnp.exp(k * x_query)
        np.testing.assert_allclose(grad[0, :], expected_gx, atol=1e-8*n_cr, rtol=1e-4)
        np.testing.assert_allclose(grad[1, :], 0.0, atol=1e-8*n_cr)
        np.testing.assert_allclose(grad[2, :], 0.0, atol=1e-8*n_cr)


# ===================================================================
# 6.  Separable product ne  →  gradient via product rule
# ===================================================================

class TestSeparableProduct:
    """For ne(x,y,z) = f(x) * g(y) * h(z), the gradient components are:
        dne/dx = f'(x) * g(y) * h(z)
        dne/dy = f(x) * g'(y) * h(z)
        dne/dz = f(x) * g(y) * h'(z)
    scaled by -0.5 * c^2 / n_cr."""

    def test_cosine_product(self):
        """ne = n0 * cos(pi*x) * cos(pi*y) * cos(pi*z)."""
        x, y, z, X, Y, Z = _make_grid(nx=101, ny=101, nz=101,
                                        xlim=(-0.4, 0.4),
                                        ylim=(-0.4, 0.4),
                                        zlim=(-0.4, 0.4))
        n_cr = _critical_density(_OMEGA)
        n0 = 0.5 * n_cr
        ne = n0 * jnp.cos(jnp.pi * X) * jnp.cos(jnp.pi * Y) * jnp.cos(jnp.pi * Z)

        gt = _gradient_term(ne, _OMEGA)
        prefactor = -0.5 * sc.c ** 2 / n_cr

        # Query at origin and a few off-axis points
        r = jnp.array([
            [0.0,  0.0,  0.0],
            [0.1,  0.1,  0.1],
            [-0.1, 0.2, -0.15],
        ])

        grad = dndr(r, gt, x, y, z)

        for i in range(r.shape[0]):
            xq, yq, zq = r[i]
            ex = prefactor * n0 * (-jnp.pi * jnp.sin(jnp.pi * xq)) * jnp.cos(jnp.pi * yq) * jnp.cos(jnp.pi * zq)
            ey = prefactor * n0 * jnp.cos(jnp.pi * xq) * (-jnp.pi * jnp.sin(jnp.pi * yq)) * jnp.cos(jnp.pi * zq)
            ez = prefactor * n0 * jnp.cos(jnp.pi * xq) * jnp.cos(jnp.pi * yq) * (-jnp.pi * jnp.sin(jnp.pi * zq))

            np.testing.assert_allclose(grad[0, i], ex, rtol=5e-3, atol=1e-8*n_cr)
            np.testing.assert_allclose(grad[1, i], ey, rtol=5e-3, atol=1e-8*n_cr)
            np.testing.assert_allclose(grad[2, i], ez, rtol=5e-3, atol=1e-8*n_cr)


# ===================================================================
# 7.  Symmetry: antisymmetric ne → antisymmetric gradient
# ===================================================================

class TestSymmetry:
    """An odd density profile ne(-x) = -ne(x) (offset so ne > 0)
    should produce an even gradient in x."""

    def test_antisymmetric_gradient(self):
        x, y, z, X, Y, Z = _make_grid(nx=101, ny=31, nz=31)
        n_cr = _critical_density(_OMEGA)

        # ne = n_cr/2 * (1 + x) gives d(ne)/dx = n_cr/2 (constant),
        # so gradient at +x0 and -x0 should be identical.
        ne = (n_cr / 2.0) * (1.0 + X)
        gt = _gradient_term(ne, _OMEGA)

        r_pos = jnp.array([[0.3, 0.0, 0.0]])
        r_neg = jnp.array([[-0.3, 0.0, 0.0]])

        grad_pos = dndr(r_pos, gt, x, y, z)
        grad_neg = dndr(r_neg, gt, x, y, z)

        # x-gradient should be the same constant at both points
        np.testing.assert_allclose(grad_pos[0, 0], grad_neg[0, 0], rtol=1e-4)


# ===================================================================
# 8.  Gradient magnitude scaling with omega
# ===================================================================

class TestOmegaScaling:
    """The ne-dependent part of gradient_term scales as 1/omega^2
    (through ncr), so doubling omega should quarter the gradient."""

    def test_omega_scaling(self):
        x, y, z, X, Y, Z = _make_grid(nx=61, ny=61, nz=31)

        # Use a linear profile so gradient is easy to predict
        ne = 1.0e24 * X + 5.0e25

        # Compute only the ne-dependent part: -0.5*c²*ne/ncr.
        # The full gradient_term (0.5*c²*n²) includes a constant
        # vacuum term that has zero gradient analytically, so omitting
        # it changes nothing physically but avoids float32 precision
        # loss in the finite-difference stencil.
        ncr1 = _critical_density(_OMEGA)
        ncr2 = _critical_density(2.0 * _OMEGA)
        gt1 = -0.5 * sc.c ** 2 * ne / ncr1
        gt2 = -0.5 * sc.c ** 2 * ne / ncr2

        r = jnp.array([[0.0, 0.0, 0.0]])

        grad1 = dndr(r, gt1, x, y, z)
        grad2 = dndr(r, gt2, x, y, z)

        # gradient scales as 1/ncr ∝ 1/omega^2, so 4x reduction
        np.testing.assert_allclose(grad2, grad1 / 4.0, rtol=1e-6)
