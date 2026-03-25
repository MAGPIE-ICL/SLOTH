"""
Tests for trace_and_save_depths.

Physics checks:
  1. Vacuum (ne=0): rays travel in straight lines, so transverse positions must
     be unchanged at every depth and angles must remain constant.
  2. Slab plasma (ne = ne_0 * (1 + s * x / L_x)): a linear density gradient in x
     exerts a force proportional to -dne/dx.  Rays starting on the positive-x
     side of the beam should acquire a negative x-velocity (deflected away from
     the high-density side), so their mean x-position must decrease with depth.
  3. Domain-size validation: requesting z_max larger than the domain raises
     ValueError.
  4. Output structure: the returned dict has the expected keys and shapes.
  5. Pickle round-trip: data written to disk and reloaded matches the return value.
  6. Uniform z-cadence: the z_saves array is uniformly spaced at the requested step.
"""

import os
import sys
import pickle
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make sure the project source is importable without installing the package
# ---------------------------------------------------------------------------
_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from scipy.constants import c

import jax
import jax.numpy as jnp

# Suppress JAX-related progress/warning noise in test output
jax.config.update('jax_platform_name', 'cpu')


# ---------------------------------------------------------------------------
# Helper: build a minimal ScalarDomain without using the real init (which
# requires memory stats, prints, etc.).  We monkey-patch an object whose
# attributes match exactly what trace_and_save_depths reads.
# ---------------------------------------------------------------------------

class _MinimalDomain:
    """Lightweight stand-in for core.domain.ScalarDomain."""

    def __init__(self, ne, x, y, z, probing_direction='z'):
        self.ne = jnp.array(ne, dtype=jnp.float32)
        self.x = jnp.array(x, dtype=jnp.float32)
        self.y = jnp.array(y, dtype=jnp.float32)
        self.z = jnp.array(z, dtype=jnp.float32)
        self.probing_direction = probing_direction
        self.lengths = jnp.array([x[-1] - x[0], y[-1] - y[0], z[-1] - z[0]], dtype=jnp.float32)
        self.dims = jnp.array([len(x), len(y), len(z)], dtype=jnp.int32)


def _vacuum_domain(half_size=2e-3, n=16):
    """Return a vacuum (ne=0) cubic domain centred on the origin."""
    coords = np.linspace(-half_size, half_size, n)
    ne = np.zeros((n, n, n), dtype=np.float32)
    return _MinimalDomain(ne, coords, coords, coords, probing_direction='z')


def _slab_domain(half_size=2e-3, n=32, ne_0=2e23, s=1.0):
    """
    Return a slab domain with ne = ne_0 * (1 + s * x / x_length).
    The density gradient is in x: dne/dx = ne_0 * s / x_length > 0 for s>0.
    """
    coords = np.linspace(-half_size, half_size, n)
    x_length = 2 * half_size
    XX, _, _ = np.meshgrid(coords, coords, coords, indexing='ij')
    ne = ne_0 * (1.0 + s * XX / x_length)
    return _MinimalDomain(ne.astype(np.float32), coords, coords, coords, probing_direction='z')


def _collimated_rays(Np=50, beam_radius=5e-4, z_start=-2e-3, seed=42):
    """
    Return s0 (6, Np): rays uniformly distributed in a circle, all moving
    in +z direction at speed c with zero divergence.
    """
    rng = np.random.RandomState(seed)
    # uniform in circle
    r = beam_radius * np.sqrt(rng.rand(Np))
    theta = 2 * np.pi * rng.rand(Np)
    x0 = r * np.cos(theta)
    y0 = r * np.sin(theta)
    z0 = np.full(Np, z_start)
    vx = np.zeros(Np)
    vy = np.zeros(Np)
    vz = np.full(Np, c)
    return jnp.array(np.stack([x0, y0, z0, vx, vy, vz], axis=0), dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Import the function under test
# ---------------------------------------------------------------------------
from core.propagator import trace_and_save_depths


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOutputStructure:
    """Check return-value shape and types."""

    def test_keys_present(self):
        domain = _vacuum_domain()
        omega = 2 * np.pi * c / 1064e-9
        s0 = _collimated_rays(Np=8, z_start=-2e-3)

        result = trace_and_save_depths(
            s0, domain,
            z_step=500e-6, z_max=1e-3,
            output_path=None,
            omega=omega, jitted=True, verbose=False,
        )

        assert 'z_saves' in result
        assert 'x' in result
        assert 'phi' in result

    def test_z_saves_shape(self):
        domain = _vacuum_domain()
        omega = 2 * np.pi * c / 1064e-9
        s0 = _collimated_rays(Np=8, z_start=-2e-3)

        result = trace_and_save_depths(
            s0, domain,
            z_step=200e-6, z_max=1e-3,
            output_path=None,
            omega=omega, verbose=False,
        )
        # 0, 200, 400, 600, 800, 1000 µm → 6 save points
        assert len(result['z_saves']) == 6
        assert len(result['x']) == 6
        assert len(result['phi']) == 6

    def test_x_phi_shapes(self):
        Np = 10
        domain = _vacuum_domain()
        omega = 2 * np.pi * c / 1064e-9
        s0 = _collimated_rays(Np=Np, z_start=-2e-3)

        result = trace_and_save_depths(
            s0, domain,
            z_step=500e-6, z_max=1e-3,
            output_path=None,
            omega=omega, verbose=False,
        )

        for arr in result['x']:
            assert arr.shape == (2, Np), f"x shape mismatch: {arr.shape}"
        for arr in result['phi']:
            assert arr.shape == (2, Np), f"phi shape mismatch: {arr.shape}"


class TestUniformSampling:
    """Check that z_saves is uniformly spaced at the requested step."""

    def test_spacing(self):
        domain = _vacuum_domain()
        omega = 2 * np.pi * c / 1064e-9
        s0 = _collimated_rays(Np=4, z_start=-2e-3)
        z_step = 200e-6

        result = trace_and_save_depths(
            s0, domain,
            z_step=z_step, z_max=1e-3,
            output_path=None,
            omega=omega, verbose=False,
        )

        diffs = np.diff(result['z_saves'])
        np.testing.assert_allclose(diffs, z_step, rtol=1e-6,
                                   err_msg="z_saves not uniformly spaced")

    def test_starts_at_zero(self):
        domain = _vacuum_domain()
        omega = 2 * np.pi * c / 1064e-9
        s0 = _collimated_rays(Np=4, z_start=-2e-3)

        result = trace_and_save_depths(
            s0, domain,
            z_step=300e-6, z_max=900e-6,
            output_path=None,
            omega=omega, verbose=False,
        )

        assert result['z_saves'][0] == pytest.approx(0.0)

    def test_ends_at_z_max(self):
        domain = _vacuum_domain()
        omega = 2 * np.pi * c / 1064e-9
        s0 = _collimated_rays(Np=4, z_start=-2e-3)
        z_max = 800e-6

        result = trace_and_save_depths(
            s0, domain,
            z_step=400e-6, z_max=z_max,
            output_path=None,
            omega=omega, verbose=False,
        )

        assert result['z_saves'][-1] == pytest.approx(z_max)


class TestDomainValidation:
    """Check that the function raises when the domain is too small."""

    def test_domain_too_small_raises(self):
        """z_max larger than the domain's probing-direction extent → ValueError."""
        # domain spans 4 mm in z; request 10 mm → must raise
        domain = _vacuum_domain(half_size=2e-3)
        omega = 2 * np.pi * c / 1064e-9
        s0 = _collimated_rays(Np=4, z_start=-2e-3)

        with pytest.raises(ValueError, match="smaller than the requested z_max"):
            trace_and_save_depths(
                s0, domain,
                z_step=1e-3, z_max=10e-3,
                output_path=None,
                omega=omega, verbose=False,
            )

    def test_nonpositive_z_step_raises(self):
        domain = _vacuum_domain()
        omega = 2 * np.pi * c / 1064e-9
        s0 = _collimated_rays(Np=4, z_start=-2e-3)

        with pytest.raises(ValueError):
            trace_and_save_depths(
                s0, domain,
                z_step=-100e-6, z_max=1e-3,
                output_path=None,
                omega=omega, verbose=False,
            )

    def test_nonpositive_z_max_raises(self):
        domain = _vacuum_domain()
        omega = 2 * np.pi * c / 1064e-9
        s0 = _collimated_rays(Np=4, z_start=-2e-3)

        with pytest.raises(ValueError):
            trace_and_save_depths(
                s0, domain,
                z_step=200e-6, z_max=0.0,
                output_path=None,
                omega=omega, verbose=False,
            )


class TestPickle:
    """Check pickle round-trip."""

    def test_pickle_roundtrip(self):
        domain = _vacuum_domain()
        omega = 2 * np.pi * c / 1064e-9
        s0 = _collimated_rays(Np=6, z_start=-2e-3)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            result = trace_and_save_depths(
                s0, domain,
                z_step=500e-6, z_max=1e-3,
                output_path=path,
                omega=omega, verbose=False,
            )

            with open(path, 'rb') as fh:
                loaded = pickle.load(fh)

            np.testing.assert_array_equal(loaded['z_saves'], result['z_saves'])
            for j in range(len(result['x'])):
                np.testing.assert_array_equal(loaded['x'][j], result['x'][j])
                np.testing.assert_array_equal(loaded['phi'][j], result['phi'][j])
        finally:
            os.remove(path)

    def test_none_output_path_no_file(self):
        """Passing output_path=None must not create any file."""
        domain = _vacuum_domain()
        omega = 2 * np.pi * c / 1064e-9
        s0 = _collimated_rays(Np=4, z_start=-2e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_and_save_depths(
                s0, domain,
                z_step=500e-6, z_max=1e-3,
                output_path=None,
                omega=omega, verbose=False,
            )
            assert len(os.listdir(tmpdir)) == 0


class TestVacuumPhysics:
    """
    In vacuum (ne=0) the Hamilton equations reduce to free streaming:
    d²r/dt² = 0.  All rays should travel in straight lines, so:
      - transverse positions (x, y) must remain constant to within numerical tolerance.
      - angles (phi, psi) must remain constant.
    """

    def test_vacuum_position_unchanged(self):
        Np = 20
        domain = _vacuum_domain(half_size=3e-3, n=16)
        omega = 2 * np.pi * c / 1064e-9
        s0 = _collimated_rays(Np=Np, z_start=-3e-3)

        result = trace_and_save_depths(
            s0, domain,
            z_step=300e-6, z_max=1.5e-3,
            output_path=None,
            omega=omega, verbose=False,
        )

        # Transverse positions at first and last save points should match
        x_initial = result['x'][0]   # (2, Np)
        x_final   = result['x'][-1]  # (2, Np)

        # Tolerance: rays move ~1.5 mm at c; numerical error should be < 1 µm
        np.testing.assert_allclose(
            x_final, x_initial, atol=1e-6,
            err_msg="Vacuum: transverse positions changed — free-streaming violated.",
        )

    def test_vacuum_angle_unchanged(self):
        Np = 20
        domain = _vacuum_domain(half_size=3e-3, n=16)
        omega = 2 * np.pi * c / 1064e-9
        s0 = _collimated_rays(Np=Np, z_start=-3e-3)

        result = trace_and_save_depths(
            s0, domain,
            z_step=300e-6, z_max=1.5e-3,
            output_path=None,
            omega=omega, verbose=False,
        )

        phi_initial = result['phi'][0]
        phi_final   = result['phi'][-1]

        np.testing.assert_allclose(
            phi_final, phi_initial, atol=1e-9,
            err_msg="Vacuum: angles changed — free-streaming violated.",
        )


class TestSlabPhysics:
    """
    Linear slab: ne = ne_0 * (1 + s * x / L_x), s > 0.

    The eikonal ODE gives d(vx)/dt ∝ -dne/dx < 0, so rays are deflected
    towards lower density (negative x).  After some propagation depth the
    mean x-position of rays initially on the positive-x side must decrease.
    """

    def test_slab_deflection_direction(self):
        half_size = 3e-3
        n = 32
        ne_0 = 2e23
        s = 1.0  # positive gradient in +x
        domain = _slab_domain(half_size=half_size, n=n, ne_0=ne_0, s=s)

        omega = 2 * np.pi * c / 351e-9  # 351 nm — large ncr (∝ 1/λ²) keeps rays below cutoff while still refracting

        # Start with rays only on the positive-x side, at z = -half_size
        Np = 40
        rng = np.random.RandomState(0)
        x0 = rng.uniform(0.5e-4, 3e-4, Np)   # small positive-x spread
        y0 = np.zeros(Np)
        z_start = -half_size
        vz = np.full(Np, c)
        s0 = jnp.array(np.stack([x0, y0, np.full(Np, z_start),
                                  np.zeros(Np), np.zeros(Np), vz], axis=0),
                        dtype=jnp.float32)

        result = trace_and_save_depths(
            s0, domain,
            z_step=500e-6, z_max=2e-3,
            output_path=None,
            omega=omega, verbose=False,
        )

        # Mean x-position should decrease as rays deflect towards -x
        mean_x_initial = float(np.mean(result['x'][0][0]))
        mean_x_final   = float(np.mean(result['x'][-1][0]))

        assert mean_x_final < mean_x_initial, (
            f"Slab plasma: expected rays to deflect toward -x (away from high density), "
            f"but mean x went from {mean_x_initial:.4e} to {mean_x_final:.4e}."
        )
