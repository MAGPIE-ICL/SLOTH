"""
Tests for the tri-linear RegularGridInterpolator.

Covers:
    1. Exact recovery at grid nodes (1D, 2D, 3D).
    2. Exact interpolation of linear fields (tri-linear interpolation
       should reproduce any function of the form a*x + b*y + c*z + d).
    3. Mid-point interpolation on a simple quadratic field.
    4. Out-of-bounds handling (fill_value).
    5. Batch / vectorised queries.
    6. Scalar-valued vs vector-valued fields.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from core.interpolator import RegularGridInterpolator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_3d_grid(nx=11, ny=13, nz=9,
                  xlim=(-1.0, 1.0), ylim=(-2.0, 2.0), zlim=(0.0, 3.0)):
    """Return (points, X, Y, Z) for a regular 3-D grid."""
    x = jnp.linspace(*xlim, nx)
    y = jnp.linspace(*ylim, ny)
    z = jnp.linspace(*zlim, nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    return (x, y, z), X, Y, Z


# ===================================================================
# 1.  Exact recovery at grid nodes
# ===================================================================

class TestExactAtNodes:
    """Interpolating at the grid points must return the stored values."""

    def test_1d(self):
        x = jnp.linspace(0.0, 5.0, 20)
        values = jnp.sin(x)
        xi = x.reshape(-1, 1)
        result = RegularGridInterpolator((x,), values, xi)
        np.testing.assert_allclose(result, values, rtol = 1e-6, atol=1e-6)

    def test_2d(self):
        x = jnp.linspace(0.0, 1.0, 10)
        y = jnp.linspace(0.0, 1.0, 12)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        values = jnp.sin(X) * jnp.cos(Y)

        # Query every grid node
        xi = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
        result = RegularGridInterpolator((x, y), values, xi)
        np.testing.assert_allclose(result, values.ravel(), rtol = 1e-6, atol=1e-6)

    def test_3d(self):
        (x, y, z), X, Y, Z = _make_3d_grid(nx=6, ny=7, nz=8)
        values = X ** 2 + Y + Z
        xi = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        result = RegularGridInterpolator((x, y, z), values, xi)
        np.testing.assert_allclose(result, values.ravel(), rtol = 1e-6, atol=1e-6)


# ===================================================================
# 2.  Linear fields must be reproduced exactly
# ===================================================================

class TestLinearFieldExact:
    """Tri-linear interpolation is exact for any multi-linear function
    f(x,y,z) = a*x + b*y + c*z + d."""

    @pytest.mark.parametrize("a,b,c,d", [
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (2.5, -1.3, 0.7, 4.0),
    ])
    def test_linear_field_3d(self, a, b, c, d):
        (x, y, z), X, Y, Z = _make_3d_grid(nx=5, ny=6, nz=7)
        values = a * X + b * Y + c * Z + d

        # Random query points *inside* the domain
        rng = np.random.default_rng(42)
        N = 200
        xi = jnp.array(np.column_stack([
            rng.uniform(float(x[0]), float(x[-1]), N),
            rng.uniform(float(y[0]), float(y[-1]), N),
            rng.uniform(float(z[0]), float(z[-1]), N),
        ]))
        result = RegularGridInterpolator((x, y, z), values, xi)
        expected = a * xi[:, 0] + b * xi[:, 1] + c * xi[:, 2] + d
        np.testing.assert_allclose(result, expected, rtol = 1e-6, atol=1e-6)

    def test_linear_field_1d(self):
        x = jnp.linspace(-3.0, 3.0, 50)
        values = 2.0 * x + 7.0
        rng = np.random.default_rng(99)
        xi = jnp.array(rng.uniform(-3.0, 3.0, (80, 1)))
        result = RegularGridInterpolator((x,), values, xi)
        expected = 2.0 * xi[:, 0] + 7.0
        np.testing.assert_allclose(result, expected, rtol = 1e-6, atol=1e-6)


# ===================================================================
# 3.  Mid-point interpolation on a known field
# ===================================================================

class TestMidpointInterpolation:
    """Check the value at the centre of a cell against the analytic
    tri-linear average for a simple quadratic field."""

    def test_midpoint_quadratic(self):
        """For f(x,y,z) = x^2, the tri-linear interpolant at the midpoint
        of a cell equals the average of the 8 corner values."""
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 1.0])
        z = jnp.array([0.0, 1.0])
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        values = X ** 2  # non-linear in x

        # Query the midpoint of the first cell [0,1]x[0,1]x[0,1]
        xi = jnp.array([[0.5, 0.5, 0.5]])
        result = RegularGridInterpolator((x, y, z), values, xi)

        # Average of corner values: (0^2 + 1^2) / 2 = 0.5
        # (y and z are constant factors of 1 in the average)
        expected = 0.5
        np.testing.assert_allclose(result, expected, rtol = 1e-6, atol=1e-12)


# ===================================================================
# 4.  Out-of-bounds handling
# ===================================================================

class TestOutOfBounds:
    """Points outside the grid should be replaced by fill_value."""

    def test_fill_value_default(self):
        x = jnp.linspace(0.0, 1.0, 10)
        values = jnp.ones(10)

        xi_oob = jnp.array([[-0.5], [1.5]])  # both out of bounds
        result = RegularGridInterpolator((x,), values, xi_oob, fill_value=0.0)
        np.testing.assert_allclose(result, 0.0, rtol = 1e-6, atol=1e-12)

    def test_fill_value_custom(self):
        x = jnp.linspace(0.0, 1.0, 10)
        values = jnp.ones(10)

        xi_oob = jnp.array([[-1.0], [2.0]])
        result = RegularGridInterpolator((x,), values, xi_oob, fill_value=-999.0)
        np.testing.assert_allclose(result, -999.0, rtol = 1e-6, atol=1e-6)

    def test_mixed_in_and_out(self):
        x = jnp.linspace(0.0, 1.0, 11)
        values = x * 2.0  # linear: f(x)=2x

        xi = jnp.array([[-0.1], [0.5], [1.1]])
        result = RegularGridInterpolator((x,), values, xi, fill_value=-1.0)
        # index 0 and 2 are OOB, index 1 is in-bounds
        np.testing.assert_allclose(result[0], -1.0, rtol = 1e-6, atol=1e-6)
        np.testing.assert_allclose(result[1], 1.0, rtol = 1e-6, atol=1e-6)
        np.testing.assert_allclose(result[2], -1.0, rtol = 1e-6, atol=1e-6)

    def test_fill_none_no_masking(self):
        """When fill_value=None, out-of-bounds points are extrapolated
        (clamped to the nearest edge)."""
        x = jnp.linspace(0.0, 1.0, 11)
        values = x * 3.0

        xi = jnp.array([[-0.5], [1.5]])
        result = RegularGridInterpolator((x,), values, xi, fill_value=None)
        # Should extrapolate / clamp – not NaN or zero
        assert jnp.all(jnp.isfinite(result))


# ===================================================================
# 5.  Batch / vectorised queries
# ===================================================================

class TestBatchQueries:
    """Multiple query points in a single call."""

    def test_many_points_3d(self):
        (x, y, z), X, Y, Z = _make_3d_grid(nx=10, ny=10, nz=10)
        values = jnp.sin(X) * jnp.cos(Y) + Z

        rng = np.random.default_rng(7)
        N = 500
        xi = jnp.array(np.column_stack([
            rng.uniform(float(x[0]), float(x[-1]), N),
            rng.uniform(float(y[0]), float(y[-1]), N),
            rng.uniform(float(z[0]), float(z[-1]), N),
        ]))
        result = RegularGridInterpolator((x, y, z), values, xi)

        # Just verify shape and finiteness (exact values hard to check
        # for a non-linear field, but should be bounded by data range).
        assert result.shape == (N,)
        assert jnp.all(jnp.isfinite(result))
        assert float(jnp.min(result)) >= float(jnp.min(values)) - 0.1
        assert float(jnp.max(result)) <= float(jnp.max(values)) + 0.1

    def test_single_point(self):
        x = jnp.linspace(0.0, 1.0, 5)
        values = x ** 2
        xi = jnp.array([[0.5]])
        result = RegularGridInterpolator((x,), values, xi)
        assert result.shape == (1,) or result.shape == ()


# ===================================================================
# 6.  Tuple-of-arrays input for xi
# ===================================================================

class TestTupleInput:
    """xi can be passed as a tuple of coordinate arrays."""

    def test_tuple_xi_2d(self):
        x = jnp.linspace(0.0, 1.0, 10)
        y = jnp.linspace(0.0, 1.0, 10)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        values = X + Y

        qx = jnp.array([0.25, 0.75])
        qy = jnp.array([0.25, 0.75])
        result = RegularGridInterpolator((x, y), values, (qx, qy))
        expected = jnp.array([0.5, 1.5])
        np.testing.assert_allclose(result, expected, rtol = 1e-6, atol=1e-6)


# ===================================================================
# 7.  Symmetry check
# ===================================================================

class TestSymmetry:
    """A symmetric field queried at symmetric points should give equal values."""

    def test_symmetric_field(self):
        x = jnp.linspace(-1.0, 1.0, 21)
        y = jnp.linspace(-1.0, 1.0, 21)
        z = jnp.linspace(-1.0, 1.0, 21)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        values = X ** 2 + Y ** 2 + Z ** 2  # radially symmetric

        xi = jnp.array([
            [ 0.5,  0.5,  0.5],
            [-0.5,  0.5,  0.5],
            [ 0.5, -0.5,  0.5],
            [-0.5, -0.5, -0.5],
        ])
        result = RegularGridInterpolator((x, y, z), values, xi)
        # All four points are at the same radius
        np.testing.assert_allclose(result, result[0], rtol = 1e-6, atol=1e-6)
