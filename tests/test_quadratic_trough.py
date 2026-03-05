"""
Test laser propagation through a quadratic electron density trough.

Analytic test problem:
    ne(y) = ncr/2 * (1 + y^2/yc^2)

A ray entering at y0 (< yc) propagating in x undergoes:
    x(tau) = tau * sqrt(1 - ne(y0)/ncr)
    y(tau) = y0 * cos(tau / (sqrt(2) * yc))

where tau = c * t is the vacuum path-length parameter,
c is the speed of light, and t is the physical time.

For a laser wavelength of 351 nm, ncr ~ 9.05e27 m^-3 (9.05e21 cm^-3).
Domain: x in [-25, 25] cm, y in [-5, 5] cm, yc = 5 cm.
"""

import sys
import os

# Ensure src/ is on the path for relative imports used by the codebase.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import jax.numpy as jnp
from scipy.constants import c, m_e, e, epsilon_0


def _integration_end_time(trace_depth):
    """Return the physical end-time of the ODE integration.

    The propagator integrates from t = 0 to t = sqrt(8) * trace_depth / c.
    The sqrt(8) safety factor ensures that rays have enough time to
    traverse the domain even when refraction slows them down.
    See propagator.solve() for the matching definition.
    """
    return float(np.sqrt(8.0) * trace_depth / c)


def _make_quadratic_trough_domain(x_length, y_length, z_length,
                                   nx, ny, nz, ncr, yc):
    """Build a ScalarDomain whose electron density follows
    ne(y) = ncr/2 * (1 + y^2 / yc^2)."""
    from core.domain import ScalarDomain

    lengths = jnp.array([x_length, y_length, z_length])
    dims = jnp.array([nx, ny, nz])

    # Build the density on a regular grid (uniform in x and z).
    y = jnp.linspace(-y_length / 2, y_length / 2, ny)
    # Broadcast to 3-D (nx, ny, nz) via meshgrid.
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


def test_quadratic_trough_single_ray():
    """
    Propagate a single ray through the quadratic trough and compare
    against the analytic trajectory.
    """
    import core.propagator as propagator

    # --- physical parameters ---
    lwl = 351e-9                          # wavelength (m)
    omega = 2.0 * np.pi * c / lwl
    ncr = omega ** 2 * m_e * epsilon_0 / e ** 2  # critical density (m^-3)
    yc = 0.05                             # trough scale (m) – 5 cm

    # --- domain geometry ---
    x_length = 0.50   # 50 cm – large enough so the ray stays inside
    y_length = 0.10   # 10 cm  (±5 cm)
    z_length = 0.02   #  2 cm  (thin; essentially 2-D problem)
    nx, ny, nz = 16, 256, 4

    domain = _make_quadratic_trough_domain(
        x_length, y_length, z_length, nx, ny, nz, ncr, yc,
    )

    # --- ray initial conditions ---
    y0 = 0.01                             # 1 cm offset
    ne_y0 = (ncr / 2.0) * (1.0 + y0 ** 2 / yc ** 2)
    n_y0 = float(np.sqrt(1.0 - ne_y0 / ncr))
    vx0 = c * n_y0                        # group velocity at entry

    s0 = _make_ray(-x_length / 2.0, y0, vx0)

    # --- propagation ---
    probing_depth = 0.10  # 10 cm trace depth
    solutions, _, duration = propagator.solve(
        s0, domain, probing_depth,
        parallelise=True,
        jitted=True,
        save_points_per_region=2,
        lwl=lwl,
        return_raw_results=True,
        verbose=False,
    )

    # --- extract final state ---
    # solutions is length-1 array of diffrax Solution objects.
    # ys shape: (Np, save_points, 6)
    final = np.asarray(solutions[0].ys[0, -1, :])
    x_num, y_num = float(final[0]), float(final[1])
    vx_num, vy_num = float(final[3]), float(final[4])

    # --- analytic solution at t_end ---
    trace_depth = probing_depth           # region_count == 1
    t_end = _integration_end_time(trace_depth)
    omega_y = c / (np.sqrt(2.0) * yc)

    x_ana = -x_length / 2.0 + vx0 * t_end
    y_ana = y0 * np.cos(omega_y * t_end)
    vx_ana = vx0
    vy_ana = -y0 * omega_y * np.sin(omega_y * t_end)

    # --- assertions ---
    pos_atol = 2e-4       # 0.2 mm absolute tolerance
    vel_rtol = 0.02       # 2 % relative tolerance

    assert abs(x_num - x_ana) < max(pos_atol, abs(x_ana) * 0.01), \
        f"x position mismatch: numerical {x_num:.6e}, analytic {x_ana:.6e}"

    assert abs(y_num - y_ana) < max(pos_atol, abs(y_ana) * 0.01), \
        f"y position mismatch: numerical {y_num:.6e}, analytic {y_ana:.6e}"

    assert abs(vx_num - vx_ana) < abs(vx_ana) * vel_rtol, \
        f"vx mismatch: numerical {vx_num:.6e}, analytic {vx_ana:.6e}"

    assert abs(vy_num - vy_ana) < max(abs(vy_ana) * vel_rtol, 1e4), \
        f"vy mismatch: numerical {vy_num:.6e}, analytic {vy_ana:.6e}"

    print("\n=== Quadratic-trough single-ray test PASSED ===")
    print(f"  x : num={x_num:.6e}  ana={x_ana:.6e}  err={abs(x_num-x_ana):.2e}")
    print(f"  y : num={y_num:.6e}  ana={y_ana:.6e}  err={abs(y_num-y_ana):.2e}")
    print(f"  vx: num={vx_num:.6e}  ana={vx_ana:.6e}")
    print(f"  vy: num={vy_num:.6e}  ana={vy_ana:.6e}")
    print(f"  Duration: {duration:.3f} s")


def test_quadratic_trough_multiple_rays():
    """
    Propagate several rays at different y0 offsets and verify they
    all match the analytic solution.
    """
    import core.propagator as propagator

    lwl = 351e-9
    omega = 2.0 * np.pi * c / lwl
    ncr = omega ** 2 * m_e * epsilon_0 / e ** 2
    yc = 0.05

    x_length = 0.50
    y_length = 0.10
    z_length = 0.02
    nx, ny, nz = 16, 256, 4

    domain = _make_quadratic_trough_domain(
        x_length, y_length, z_length, nx, ny, nz, ncr, yc,
    )

    # Several offsets (all < yc)
    y0_values = [0.005, 0.01, 0.02, 0.03]
    Np = len(y0_values)

    s0 = jnp.zeros((6, Np))
    n_vals = []
    for j, y0 in enumerate(y0_values):
        ne_y0 = (ncr / 2.0) * (1.0 + y0 ** 2 / yc ** 2)
        n_y0 = float(np.sqrt(1.0 - ne_y0 / ncr))
        n_vals.append(n_y0)
        vx0 = c * n_y0
        s0 = s0.at[0, j].set(-x_length / 2.0)
        s0 = s0.at[1, j].set(y0)
        s0 = s0.at[3, j].set(vx0)

    probing_depth = 0.10
    solutions, _, duration = propagator.solve(
        s0, domain, probing_depth,
        parallelise=True,
        jitted=True,
        save_points_per_region=2,
        lwl=lwl,
        return_raw_results=True,
        verbose=False,
    )

    trace_depth = probing_depth
    t_end = _integration_end_time(trace_depth)
    omega_y = c / (np.sqrt(2.0) * yc)

    pos_atol = 2e-4
    print("\n=== Quadratic-trough multi-ray test ===")
    all_pass = True
    for j, y0 in enumerate(y0_values):
        final = np.asarray(solutions[0].ys[j, -1, :])
        y_num = float(final[1])

        y_ana = y0 * np.cos(omega_y * t_end)

        err = abs(y_num - y_ana)
        ok = err < max(pos_atol, abs(y_ana) * 0.01)
        status = "PASS" if ok else "FAIL"
        print(f"  y0={y0:.4f}: y_num={y_num:.6e}  y_ana={y_ana:.6e}  err={err:.2e}  [{status}]")
        if not ok:
            all_pass = False

    assert all_pass, "One or more rays failed the analytic comparison."
    print("=== Multi-ray test PASSED ===")


if __name__ == '__main__':
    test_quadratic_trough_single_ray()
    test_quadratic_trough_multiple_rays()
