"""
Tests for inverse bremsstrahlung amplitude attenuation in the JAX ray tracer.

Physics checks:
  1. Vacuum (ne=0, inv_brems=True):  amplitude must remain 1.0 everywhere
     (no plasma ⟹ no collisions ⟹ no absorption).
  2. Uniform plasma, single ray on-axis: the amplitude at depth L must match
     the analytic prediction  a(L) = exp(-κ · L/c),  where κ is the NRL
     inverse bremsstrahlung rate computed from the domain parameters.
  3. Monotonic decay: amplitude decreases (or stays equal) at every successive
     depth save in a uniform plasma.
  4. Density scaling: a denser plasma produces stronger attenuation
     (lower final amplitude) than a dilute plasma with the same Te and Z.
"""

import os
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make sure the project source is importable without installing the package
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
# Helper: lightweight domain stand-in (matches what trace_and_save_depths reads)
# ---------------------------------------------------------------------------

class _MinimalDomain:
    """Lightweight stand-in for core.domain.ScalarDomain."""

    def __init__(self, ne, x, y, z, probing_direction='z',
                 inv_brems=False, Te=None, Z=None):
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

        self.inv_brems = inv_brems
        if Te is not None:
            self.Te = jnp.asarray(Te, dtype=jnp.float32)
        else:
            self.Te = None
        self.Z = jnp.float32(Z) if Z is not None else None


# ---------------------------------------------------------------------------
# Helpers: domain / beam factories
# ---------------------------------------------------------------------------

_HALF = 2e-3  # domain half-size (m)
_N    = 16    # grid points per axis


def _vacuum_domain_ib(half_size=_HALF, n=_N):
    """Vacuum domain with inv_brems enabled (ne=0, so no actual absorption)."""
    coords = np.linspace(-half_size, half_size, n)
    ne     = np.zeros((n, n, n), dtype=np.float32)
    return _MinimalDomain(
        ne, coords, coords, coords, probing_direction='z',
        inv_brems=True, Te=100.0, Z=1.0,
    )


def _uniform_plasma_domain(ne_val, Te_val=100.0, Z_val=1.0,
                            half_size=_HALF, n=_N):
    """Uniform-density domain with inv_brems enabled."""
    coords = np.linspace(-half_size, half_size, n)
    ne     = np.full((n, n, n), ne_val, dtype=np.float32)
    return _MinimalDomain(
        ne, coords, coords, coords, probing_direction='z',
        inv_brems=True, Te=float(Te_val), Z=float(Z_val),
    )


def _collimated_rays(Np=20, beam_radius=3e-4, z_start=-_HALF):
    """
    Return s0 (6, Np): collimated rays travelling in +z at speed c.
    """
    rng = np.random.RandomState(0)
    r   = beam_radius * np.sqrt(rng.rand(Np))
    th  = 2 * np.pi * rng.rand(Np)
    x0  = r * np.cos(th)
    y0  = r * np.sin(th)
    z0  = np.full(Np, z_start)
    vz  = np.full(Np, c)
    return jnp.array(
        np.stack([x0, y0, z0, np.zeros(Np), np.zeros(Np), vz], axis=0),
        dtype=jnp.float32,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVacuumAmplitude:
    """With ne=0 the amplitude must remain 1 regardless of Te/Z."""

    def test_amplitude_is_one_in_vacuum(self):
        domain = _vacuum_domain_ib()
        s0     = _collimated_rays(Np=10)

        result = trace_and_save_depths(
            s0, domain,
            step=500e-6, depth_max=1e-3,
            output_path=None,
            lwl=1064e-9, jitted=True, verbose=False,
        )

        assert 'amplitude' in result, "Expected 'amplitude' key when inv_brems=True"

        for j, amp_j in enumerate(result['amplitude']):
            np.testing.assert_allclose(
                amp_j, 1.0, atol=1e-5,
                err_msg=f"Amplitude deviated from 1.0 in vacuum at depth index {j}",
            )


class TestAnalyticalAmplitude:
    """
    For a uniform plasma the amplitude must match the analytical formula:
        a(L) = exp(-κ · L / c)
    where κ = kappa_inv_brems(ne, Te, Z, omega).
    """

    # Parameters chosen to give ≈10 % amplitude loss over 2 mm — clearly
    # measurable while remaining in the physically valid regime.
    NE_VAL  = 1e25   # m^-3
    TE_VAL  = 100.0  # eV
    Z_VAL   = 1.0
    LWL     = 1064e-9
    DEPTH   = 2e-3   # m   (must fit inside domain)
    HALF    = 3e-3   # domain half-size (m) — larger than DEPTH/2

    def _expected_amplitude(self, depth):
        """Analytical amplitude at a given depth (metres)."""
        omega = 2 * np.pi * c / self.LWL
        kappa = float(kappa_inv_brems(
            jnp.float32(self.NE_VAL),
            jnp.float32(self.TE_VAL),
            self.Z_VAL,
            omega,
        ))
        return np.exp(-kappa * depth / c)

    def test_final_amplitude_matches_analytic(self):
        domain = _uniform_plasma_domain(
            self.NE_VAL, Te_val=self.TE_VAL, Z_val=self.Z_VAL,
            half_size=self.HALF, n=24,
        )
        s0 = _collimated_rays(Np=16, beam_radius=2e-4, z_start=-self.HALF)

        result = trace_and_save_depths(
            s0, domain,
            step=500e-6, depth_max=self.DEPTH,
            output_path=None,
            lwl=self.LWL, jitted=True, verbose=False,
        )

        assert 'amplitude' in result

        # Check the final depth snapshot
        amp_final = np.asarray(result['amplitude'][-1])
        expected  = self._expected_amplitude(self.DEPTH)

        # 5 % relative tolerance accounts for ODE solver discretisation
        np.testing.assert_allclose(
            amp_final.mean(), expected, rtol=0.05,
            err_msg=(
                f"Final amplitude {amp_final.mean():.6f} differs from analytic "
                f"prediction {expected:.6f} by more than 5 %"
            ),
        )

    def test_amplitude_at_intermediate_depths(self):
        """Amplitude at each saved depth must match the analytic formula."""
        domain = _uniform_plasma_domain(
            self.NE_VAL, Te_val=self.TE_VAL, Z_val=self.Z_VAL,
            half_size=self.HALF, n=24,
        )
        s0 = _collimated_rays(Np=16, beam_radius=2e-4, z_start=-self.HALF)

        result = trace_and_save_depths(
            s0, domain,
            step=400e-6, depth_max=self.DEPTH,
            output_path=None,
            lwl=self.LWL, jitted=True, verbose=False,
        )

        depths = result['depth_saves']
        for depth, amp_snap in zip(depths, result['amplitude']):
            expected = self._expected_amplitude(depth)
            amp_mean = float(np.asarray(amp_snap).mean())
            np.testing.assert_allclose(
                amp_mean, expected, rtol=0.05,
                err_msg=(
                    f"Amplitude {amp_mean:.6f} at depth {depth*1e3:.2f} mm "
                    f"deviates from analytic {expected:.6f} by more than 5 %"
                ),
            )


class TestMonotonicDecay:
    """Amplitude must be non-increasing at successive depth snapshots."""

    def test_amplitude_non_increasing(self):
        domain = _uniform_plasma_domain(ne_val=1e25, n=20)
        s0     = _collimated_rays(Np=12)

        result = trace_and_save_depths(
            s0, domain,
            step=250e-6, depth_max=1e-3,
            output_path=None,
            lwl=1064e-9, jitted=True, verbose=False,
        )

        amps = [np.asarray(a).mean() for a in result['amplitude']]
        for i in range(1, len(amps)):
            assert amps[i] <= amps[i - 1] + 1e-6, (
                f"Amplitude increased at depth index {i}: "
                f"{amps[i - 1]:.8f} → {amps[i]:.8f}"
            )


class TestDensityScaling:
    """Higher density → lower final amplitude (more absorption)."""

    @pytest.mark.parametrize("ne_low, ne_high", [
        (1e24, 1e25),
        (1e25, 5e25),
    ])
    def test_higher_density_more_attenuation(self, ne_low, ne_high):
        s0 = _collimated_rays(Np=10)

        def _run(ne_val):
            domain = _uniform_plasma_domain(ne_val, n=20)
            result = trace_and_save_depths(
                s0, domain,
                step=500e-6, depth_max=1e-3,
                output_path=None,
                lwl=1064e-9, jitted=True, verbose=False,
            )
            return float(np.asarray(result['amplitude'][-1]).mean())

        amp_low  = _run(ne_low)
        amp_high = _run(ne_high)

        assert amp_high < amp_low, (
            f"Expected higher density ne={ne_high:.0e} to give lower amplitude "
            f"({amp_high:.8f}) than ne={ne_low:.0e} ({amp_low:.8f})"
        )


class TestNoAmplitudeKeyWithoutIB:
    """When inv_brems is not set, 'amplitude' must NOT appear in the result."""

    def test_no_amplitude_key_without_inv_brems(self):
        coords = np.linspace(-_HALF, _HALF, _N)
        ne     = np.zeros((_N, _N, _N), dtype=np.float32)
        domain = _MinimalDomain(ne, coords, coords, coords)  # inv_brems=False

        s0 = _collimated_rays(Np=6)
        result = trace_and_save_depths(
            s0, domain,
            step=500e-6, depth_max=1e-3,
            output_path=None,
            lwl=1064e-9, jitted=True, verbose=False,
        )

        assert 'amplitude' not in result, (
            "Did not expect 'amplitude' key when inv_brems is False"
        )
