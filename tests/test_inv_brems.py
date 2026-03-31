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
  5. Wavelength scaling: kappa ∝ (ne/ω)², so higher frequency means less absorption.
  6. Np independence: amplitude must not depend on the number of rays traced.
  7. Coulomb logarithm classical regime: for cold plasma or high-Z ions where the
     classical minimum impact parameter b_classical = Ze/(4πε₀·Te) exceeds the
     quantum parameter b_quantum, the correct b_min must be used.
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

from scipy.constants import c, epsilon_0
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
        self.Z = jnp.asarray(Z, dtype=jnp.float32) if Z is not None else None


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
        assert 'jvec_unweighted' not in result, (
            "Did not expect 'jvec_unweighted' key when inv_brems is False"
        )


class TestWeightedJvec:
    """
    When inv_brems=True the returned jvec must be the amplitude-weighted Jones
    vector, and jvec_unweighted must equal jvec / amplitude.
    """

    NE_VAL = 1e25
    TE_VAL = 100.0
    Z_VAL  = 1.0
    LWL    = 1064e-9
    HALF   = 3e-3

    def _run(self):
        domain = _uniform_plasma_domain(
            self.NE_VAL, Te_val=self.TE_VAL, Z_val=self.Z_VAL,
            half_size=self.HALF, n=20,
        )
        s0 = _collimated_rays(Np=16, beam_radius=2e-4, z_start=-self.HALF)
        return trace_and_save_depths(
            s0, domain,
            step=500e-6, depth_max=2e-3,
            output_path=None,
            lwl=self.LWL, jitted=True, verbose=False,
        )

    def test_jvec_unweighted_key_present(self):
        result = self._run()
        assert 'jvec_unweighted' in result, (
            "Expected 'jvec_unweighted' key when inv_brems=True"
        )

    def test_jvec_weighted_equals_unweighted_times_amplitude(self):
        """jvec[j] must equal jvec_unweighted[j] * amplitude[j][newaxis]."""
        result = self._run()
        for j, (jv_w, jv_u, amp_j) in enumerate(
            zip(result['jvec'], result['jvec_unweighted'], result['amplitude'])
        ):
            expected = jv_u * np.asarray(amp_j)[np.newaxis, :]
            np.testing.assert_allclose(
                jv_w, expected, rtol=1e-6,
                err_msg=f"jvec weighted/unweighted mismatch at depth index {j}",
            )

    def test_jvec_weighted_smaller_than_unweighted(self):
        """
        In an absorbing plasma the amplitude < 1, so |jvec| < |jvec_unweighted|
        at the final depth (ignoring the zero-position depth 0).
        """
        result = self._run()
        # Compare RMS magnitudes at the final save depth (index > 0)
        jv_w = result['jvec'][-1]
        jv_u = result['jvec_unweighted'][-1]
        rms_w = float(np.sqrt(np.mean(jv_w ** 2)))
        rms_u = float(np.sqrt(np.mean(jv_u ** 2)))
        assert rms_w < rms_u, (
            f"Expected weighted jvec RMS ({rms_w:.6f}) < unweighted ({rms_u:.6f})"
        )

    def test_jvec_unweighted_shape_matches_jvec(self):
        result = self._run()
        for j, (jv_w, jv_u) in enumerate(
            zip(result['jvec'], result['jvec_unweighted'])
        ):
            assert jv_w.shape == jv_u.shape, (
                f"Shape mismatch at depth {j}: jvec {jv_w.shape} vs "
                f"jvec_unweighted {jv_u.shape}"
            )


class TestArrayZ:
    """Z charge state can be a spatially varying array, not just a scalar."""

    NE_VAL  = 1e25
    TE_VAL  = 100.0
    Z_VAL   = 1.0
    LWL     = 1064e-9
    DEPTH   = 2e-3
    HALF    = 3e-3

    def _domain_with_scalar_z(self, n=20):
        coords = np.linspace(-self.HALF, self.HALF, n)
        ne     = np.full((n, n, n), self.NE_VAL, dtype=np.float32)
        return _MinimalDomain(
            ne, coords, coords, coords, probing_direction='z',
            inv_brems=True, Te=self.TE_VAL, Z=self.Z_VAL,
        )

    def _domain_with_uniform_array_z(self, n=20):
        """Z supplied as a 3-D array filled with the same scalar value."""
        coords = np.linspace(-self.HALF, self.HALF, n)
        ne     = np.full((n, n, n), self.NE_VAL, dtype=np.float32)
        Z_arr  = np.full((n, n, n), self.Z_VAL, dtype=np.float32)
        return _MinimalDomain(
            ne, coords, coords, coords, probing_direction='z',
            inv_brems=True, Te=self.TE_VAL, Z=Z_arr,
        )

    def test_array_z_stored_as_array(self):
        """When Z is provided as a 3-D array it must be stored as a JAX array with correct values."""
        domain = self._domain_with_uniform_array_z()
        assert hasattr(domain.Z, 'shape'), "Z should be a JAX array with a .shape attribute"
        assert domain.Z.shape == (20, 20, 20), f"Unexpected Z shape: {domain.Z.shape}"
        np.testing.assert_allclose(
            np.asarray(domain.Z), self.Z_VAL, rtol=1e-6,
            err_msg="Array Z values should all equal Z_VAL",
        )

    def test_scalar_z_stored_as_array(self):
        """When Z is provided as a scalar it must also be stored as a JAX array with the correct value."""
        domain = self._domain_with_scalar_z()
        assert hasattr(domain.Z, 'shape'), "Z should be a JAX array with a .shape attribute"
        assert float(domain.Z) == self.Z_VAL, (
            f"Scalar Z value {float(domain.Z)} should equal Z_VAL {self.Z_VAL}"
        )

    def test_uniform_array_z_matches_scalar_z(self):
        """A uniform array Z must produce the same final amplitude as the equivalent scalar Z."""
        s0 = _collimated_rays(Np=12, beam_radius=2e-4, z_start=-self.HALF)

        def _run(domain):
            result = trace_and_save_depths(
                s0, domain,
                step=500e-6, depth_max=self.DEPTH,
                output_path=None,
                lwl=self.LWL, jitted=True, verbose=False,
            )
            return float(np.asarray(result['amplitude'][-1]).mean())

        amp_scalar = _run(self._domain_with_scalar_z())
        amp_array  = _run(self._domain_with_uniform_array_z())

        np.testing.assert_allclose(
            amp_array, amp_scalar, rtol=1e-5,
            err_msg=(
                f"Uniform array Z ({amp_array:.8f}) should match scalar Z ({amp_scalar:.8f})"
            ),
        )

    def test_varying_z_changes_absorption(self):
        """A higher Z in the beam path must produce more absorption than Z=1."""
        n      = 20
        coords = np.linspace(-self.HALF, self.HALF, n)
        ne     = np.full((n, n, n), self.NE_VAL, dtype=np.float32)

        # Z=1 everywhere (low absorption)
        domain_low_z = _MinimalDomain(
            ne, coords, coords, coords, probing_direction='z',
            inv_brems=True, Te=self.TE_VAL, Z=1.0,
        )

        # Z=4 everywhere (high absorption)
        domain_high_z = _MinimalDomain(
            ne, coords, coords, coords, probing_direction='z',
            inv_brems=True, Te=self.TE_VAL, Z=4.0,
        )

        s0 = _collimated_rays(Np=12, beam_radius=2e-4, z_start=-self.HALF)

        def _run(domain):
            result = trace_and_save_depths(
                s0, domain,
                step=500e-6, depth_max=self.DEPTH,
                output_path=None,
                lwl=self.LWL, jitted=True, verbose=False,
            )
            return float(np.asarray(result['amplitude'][-1]).mean())

        amp_low  = _run(domain_low_z)
        amp_high = _run(domain_high_z)

        assert amp_high < amp_low, (
            f"Expected higher Z to give lower amplitude: Z=4 gave {amp_high:.6f}, "
            f"Z=1 gave {amp_low:.6f}"
        )


# ---------------------------------------------------------------------------
# New tests: wavelength scaling, Np independence, Coulomb-log classical regime
# ---------------------------------------------------------------------------

class TestWavelengthScaling:
    """
    Inverse bremsstrahlung absorption scales as κ ∝ (nₑ/ω)², so higher laser
    frequency (shorter wavelength) produces less absorption.

    For each wavelength the measured amplitude at a fixed depth must match the
    analytic formula  a(L) = exp(-κ · L/c)  where κ = kappa_inv_brems(..., ω).
    """

    NE_VAL = 1e25    # m^-3  — underdense for all three wavelengths
    TE_VAL = 100.0   # eV
    Z_VAL  = 1.0
    DEPTH  = 2e-3    # m
    HALF   = 3e-3    # domain half-size (m)

    def _expected_amplitude(self, lwl):
        omega = 2 * np.pi * c / lwl
        kappa = float(kappa_inv_brems(
            jnp.float32(self.NE_VAL),
            jnp.float32(self.TE_VAL),
            self.Z_VAL,
            omega,
        ))
        return np.exp(-kappa * self.DEPTH / c)

    @pytest.mark.parametrize("lwl", [351e-9, 527e-9, 1064e-9])
    def test_amplitude_matches_analytic_at_wavelength(self, lwl):
        """Amplitude at DEPTH matches exp(-κ·L/c) for each laser wavelength."""
        domain = _uniform_plasma_domain(
            self.NE_VAL, Te_val=self.TE_VAL, Z_val=self.Z_VAL,
            half_size=self.HALF, n=24,
        )
        s0 = _collimated_rays(Np=16, beam_radius=2e-4, z_start=-self.HALF)

        result = trace_and_save_depths(
            s0, domain,
            step=500e-6, depth_max=self.DEPTH,
            output_path=None,
            lwl=lwl, jitted=True, verbose=False,
        )

        amp_final = float(np.asarray(result['amplitude'][-1]).mean())
        expected  = self._expected_amplitude(lwl)

        np.testing.assert_allclose(
            amp_final, expected, rtol=0.05,
            err_msg=(
                f"Wavelength {lwl*1e9:.0f} nm: final amplitude {amp_final:.6f} "
                f"differs from analytic {expected:.6f} by more than 5 %"
            ),
        )

    def test_shorter_wavelength_less_absorption(self):
        """
        351 nm (3ω) must produce more surviving amplitude than 1064 nm (1ω)
        because κ ∝ 1/ω² — higher frequency means weaker absorption.
        """
        domain = _uniform_plasma_domain(
            self.NE_VAL, Te_val=self.TE_VAL, Z_val=self.Z_VAL,
            half_size=self.HALF, n=24,
        )
        s0 = _collimated_rays(Np=16, beam_radius=2e-4, z_start=-self.HALF)

        def _run(lwl):
            result = trace_and_save_depths(
                s0, domain,
                step=500e-6, depth_max=self.DEPTH,
                output_path=None,
                lwl=lwl, jitted=True, verbose=False,
            )
            return float(np.asarray(result['amplitude'][-1]).mean())

        amp_351  = _run(351e-9)
        amp_1064 = _run(1064e-9)

        assert amp_351 > amp_1064, (
            f"Expected 351 nm (amp={amp_351:.6f}) to survive more than 1064 nm "
            f"(amp={amp_1064:.6f}); κ ∝ 1/ω² so higher ω → less absorption."
        )

    def test_kappa_scales_as_inverse_omega_squared(self):
        """
        For the same plasma parameters, kappa(ω₁)/kappa(ω₂) must equal (ω₂/ω₁)²
        when both wavelengths are in the underdense limit (ωpe < ω).
        The Coulomb logarithm changes slightly with ω (through ω_max), so we
        allow a 10 % tolerance.
        """
        omega_351  = 2 * np.pi * c / 351e-9
        omega_1064 = 2 * np.pi * c / 1064e-9

        kappa_351  = float(kappa_inv_brems(
            jnp.float32(self.NE_VAL), jnp.float32(self.TE_VAL), self.Z_VAL, omega_351))
        kappa_1064 = float(kappa_inv_brems(
            jnp.float32(self.NE_VAL), jnp.float32(self.TE_VAL), self.Z_VAL, omega_1064))

        ratio_computed   = kappa_351 / kappa_1064
        ratio_expected   = (omega_1064 / omega_351) ** 2   # = (351/1064)²  ≈ 0.109

        np.testing.assert_allclose(
            ratio_computed, ratio_expected, rtol=0.10,
            err_msg=(
                f"κ(351nm)/κ(1064nm) = {ratio_computed:.4f}, "
                f"expected ≈ (ω_1064/ω_351)² = {ratio_expected:.4f} (10 % tolerance)"
            ),
        )


class TestNpScaling:
    """
    Each ray is traced independently, so the mean amplitude must not depend on
    the total number of rays Np (within sampling noise and ODE tolerances).
    """

    NE_VAL = 1e25    # m^-3
    TE_VAL = 100.0   # eV
    Z_VAL  = 1.0
    LWL    = 1064e-9
    DEPTH  = 2e-3    # m
    HALF   = 3e-3

    def _expected_amplitude(self):
        omega = 2 * np.pi * c / self.LWL
        kappa = float(kappa_inv_brems(
            jnp.float32(self.NE_VAL), jnp.float32(self.TE_VAL), self.Z_VAL, omega))
        return np.exp(-kappa * self.DEPTH / c)

    @pytest.mark.parametrize("Np", [10, 50, 100, 500])
    def test_amplitude_matches_analytic_for_Np(self, Np):
        """Mean amplitude must match the analytic formula regardless of Np."""
        expected = self._expected_amplitude()

        domain = _uniform_plasma_domain(
            self.NE_VAL, Te_val=self.TE_VAL, Z_val=self.Z_VAL,
            half_size=self.HALF, n=24,
        )
        s0 = _collimated_rays(Np=Np, beam_radius=2e-4, z_start=-self.HALF)

        result = trace_and_save_depths(
            s0, domain,
            step=500e-6, depth_max=self.DEPTH,
            output_path=None,
            lwl=self.LWL, jitted=True, verbose=False,
        )

        amp_final = float(np.asarray(result['amplitude'][-1]).mean())

        np.testing.assert_allclose(
            amp_final, expected, rtol=0.05,
            err_msg=(
                f"Np={Np}: amplitude {amp_final:.6f} differs from analytic "
                f"{expected:.6f} by more than 5 %"
            ),
        )

    def test_mean_amplitude_consistent_across_Np(self):
        """
        The mean amplitude from Np=50 and Np=200 must agree to within 5 %
        (independent of number of rays).
        """
        domain = _uniform_plasma_domain(
            self.NE_VAL, Te_val=self.TE_VAL, Z_val=self.Z_VAL,
            half_size=self.HALF, n=24,
        )

        def _run(Np):
            s0 = _collimated_rays(Np=Np, beam_radius=2e-4, z_start=-self.HALF)
            result = trace_and_save_depths(
                s0, domain,
                step=500e-6, depth_max=self.DEPTH,
                output_path=None,
                lwl=self.LWL, jitted=True, verbose=False,
            )
            return float(np.asarray(result['amplitude'][-1]).mean())

        amp_50  = _run(50)
        amp_200 = _run(200)

        np.testing.assert_allclose(
            amp_200, amp_50, rtol=0.05,
            err_msg=(
                f"Mean amplitude differs between Np=50 ({amp_50:.6f}) and "
                f"Np=200 ({amp_200:.6f}) by more than 5 %"
            ),
        )


class TestCoulombLogClassicalRegime:
    """
    When the classical minimum impact parameter
        b_classical = Z·e / (4πε₀·Tₑ[eV])   [m]
    exceeds the quantum one
        b_quantum   = 2.76e-10 / √Tₑ[eV]     [m],
    the Coulomb logarithm must use b_classical as b_min.

    This happens for cold plasmas (Tₑ < ~27 Z² eV).  The original buggy code
    used  b_classical = Z·e / Tₑ  (missing 1/(4πε₀)), so b_classical was ~9×10⁹
    times too small and the quantum term always dominated — overestimating the
    Coulomb logarithm and therefore kappa.

    These tests verify that kappa_inv_brems returns the value consistent with the
    correct Coulomb log in both the classical and quantum regimes.
    """

    from scipy.constants import e as _e

    @staticmethod
    def _reference_kappa(ne_val, Te_eV, Z_val, omega):
        """Compute the expected kappa using the correct physics formulae."""
        from scipy.constants import e as e_charge, epsilon_0 as eps0
        ne_cc = ne_val * 1e-6
        v_the = 4.19e5 * np.sqrt(Te_eV)
        o_pe  = 5.64e4 * np.sqrt(ne_cc)
        o_max = max(o_pe, omega)
        # correct classical minimum impact parameter
        b_classical = Z_val * e_charge / (4.0 * np.pi * eps0 * Te_eV)
        b_quantum   = 2.760428269727312e-10 / np.sqrt(Te_eV)
        b_min       = max(b_classical, b_quantum)
        CL = max(2.0, np.log(v_the / (o_max * b_min)))
        return 3.1e-5 * Z_val * c * (ne_cc / omega) ** 2 * CL * Te_eV ** (-1.5)

    # (Te_eV, Z_val) pairs — classical regime (b_classical > b_quantum) marked *
    @pytest.mark.parametrize("Te_eV, Z_val", [
        (100.0, 1.0),   # quantum regime  (Te = 100 eV > 27·1² = 27 eV)
        (5.0,   1.0),   # classical regime*  (Te = 5 eV < 27 eV)
        (10.0,  1.0),   # classical regime*  (Te = 10 eV < 27 eV)
        (100.0, 4.0),   # classical regime*  (Te = 100 eV < 27·4² = 432 eV)
        (200.0, 6.0),   # classical regime*  (Te = 200 eV < 27·6² = 972 eV)
    ])
    def test_kappa_matches_correct_coulomb_log(self, Te_eV, Z_val):
        """kappa_inv_brems must agree with the reference (correct b_min) formula."""
        ne_val = 1e25   # m^-3 — underdense
        omega  = 2 * np.pi * c / 1064e-9

        kappa_ref = self._reference_kappa(ne_val, Te_eV, Z_val, omega)
        kappa_got = float(kappa_inv_brems(
            jnp.float32(ne_val), jnp.float32(Te_eV), float(Z_val), omega))

        np.testing.assert_allclose(
            kappa_got, kappa_ref, rtol=1e-4,
            err_msg=(
                f"Te={Te_eV} eV, Z={Z_val}: kappa={kappa_got:.6g}, "
                f"expected={kappa_ref:.6g}. "
                f"Check that b_classical = Ze/(4πε₀·Te) is used when it exceeds b_quantum."
            ),
        )

    def test_classical_regime_gives_lower_kappa_than_buggy(self):
        """
        In the classical regime (b_classical > b_quantum), using the correct
        b_classical (larger) increases b_min and therefore REDUCES the Coulomb log
        and kappa relative to the buggy implementation that always used b_quantum.

        This verifies the direction of the fix: the corrected kappa must be
        ≤ the buggy kappa in the classical-dominated regime.
        """
        from scipy.constants import e as e_charge, epsilon_0 as eps0
        # Cold plasma, Z=4: clearly in classical regime
        Te_eV  = 10.0
        Z_val  = 4.0
        ne_val = 1e25
        omega  = 2 * np.pi * c / 1064e-9

        # Compute CL with correct b_classical
        ne_cc = ne_val * 1e-6
        v_the = 4.19e5 * np.sqrt(Te_eV)
        o_pe  = 5.64e4 * np.sqrt(ne_cc)
        o_max = max(o_pe, omega)
        b_classical_correct = Z_val * e_charge / (4.0 * np.pi * eps0 * Te_eV)
        b_classical_buggy   = Z_val * e_charge / Te_eV       # missing 1/(4πε₀)
        b_quantum           = 2.760428269727312e-10 / np.sqrt(Te_eV)

        CL_correct = max(2.0, np.log(v_the / (o_max * max(b_classical_correct, b_quantum))))
        CL_buggy   = max(2.0, np.log(v_the / (o_max * max(b_classical_buggy,   b_quantum))))

        assert b_classical_correct > b_quantum, (
            "Test setup error: b_classical_correct should exceed b_quantum here."
        )
        assert CL_correct < CL_buggy, (
            f"CL_correct ({CL_correct:.4f}) should be < CL_buggy ({CL_buggy:.4f}) "
            f"because larger b_min → smaller argument of log."
        )

        # The actual kappa_inv_brems must match the corrected value
        kappa_correct = 3.1e-5 * Z_val * c * (ne_cc/omega)**2 * CL_correct * Te_eV**(-1.5)
        kappa_got     = float(kappa_inv_brems(
            jnp.float32(ne_val), jnp.float32(Te_eV), float(Z_val), omega))

        np.testing.assert_allclose(
            kappa_got, kappa_correct, rtol=1e-4,
            err_msg=(
                f"Te={Te_eV} eV, Z={Z_val}: kappa={kappa_got:.6g} should equal "
                f"corrected value {kappa_correct:.6g} (not buggy {3.1e-5*Z_val*c*(ne_cc/omega)**2*CL_buggy*Te_eV**(-1.5):.6g})."
            ),
        )
