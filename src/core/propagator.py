import jax
import jax.numpy as jnp
import numpy as np

import os

from scipy.integrate import odeint, solve_ivp
from time import time
from sys import getsizeof as getsizeof_default

# object type of diffrax output
from diffrax import Solution

from scipy.constants import c
from scipy.constants import e
from jax.scipy.interpolate import RegularGridInterpolator
from shared.utils import getsizeof
from shared.utils import mem_conversion
from shared.printing import colour
from shared.utils import add_integer_postfix
# change name when it actualy is a trilinear interpolator - if it's still a regular grid, change it.
from core.interpolator import RegularGridInterpolator as trilinearInterpolator

from shared.propagation import ray_to_Jonesvector
from shared.propagation import back_propogate

##
## Helper functions for calculations
##

def omega_pe(ne):
    """Calculate electron plasma freq. Output units are rad/sec. From nrl pp 28"""
    return 5.64e4 * jnp.sqrt(ne)

# Plasma refractive index
def n_refrac(ne, omega):
    return jnp.sqrt(1.0 - (omega_pe(ne * 1e-6) / omega) ** 2)

def precompute_gradients(ne, x, y, z, omega):
    """
    Pre-compute the spatial gradients of the refractive-index driving term once,
    before the ODE solve.  Passing the resulting arrays into the ODE function
    avoids repeating the full-grid ``jnp.gradient`` calls at every adaptive
    time step.

    The driving term is::

        gradient_term = -0.5 * c² * ne / (3.14207787e-4 * ω²)

    where ``3.14207787e-4 = mₑ ε₀ / e²`` (SI).

    The scalar coefficient ``-0.5 * c² / (3.14207787e-4 * ω²)`` is evaluated
    in float64 and then multiplied into the (potentially float32) *ne* array.
    This ordering prevents float32 overflow: computing ``c² * ne`` first would
    overflow for ``ne ≳ 4×10²²`` m⁻³ (since c² ≈ 9×10¹⁶ and float32 max ≈
    3.4×10³⁸), whereas the coefficient itself is ≈ −4.6×10⁻¹¹ so the product
    remains well within float32 range for any sub-critical plasma.

    Args:
        ne    (jax.Array): Electron density grid **in m⁻³**, shape ``(Nx, Ny, Nz)``.
            If your density is in cm⁻³ convert with ``ne_m3 = ne_cc * 1e6`` before
            passing.  Cells with ``ne ≥ ncr`` (critical density) are physically
            evanescent; the function warns if any such cells are present, as this
            causes large gradient magnitudes that can prevent the ODE from converging.
        x, y, z (jax.Array): 1-D coordinate arrays in metres.
        omega (float): Laser angular frequency in rad/s.

    Returns:
        tuple: ``(dndx, dndy, dndz)`` — each a ``(Nx, Ny, Nz)`` JAX array giving
        the gradient of ``gradient_term`` along the respective axis.
    """
    # Critical density for this laser frequency [m^-3]: ncr = me*eps0*omega^2/e^2
    ncr = 3.14207787e-4 * float(omega) ** 2
    ne_max = float(jnp.max(ne))
    if ne_max >= ncr:
        import warnings
        warnings.warn(
            f"precompute_gradients: {ne_max:.3g} m⁻³ cells at or above critical "
            f"density ncr = {ncr:.3g} m⁻³ (ne/ncr = {ne_max/ncr:.2g}).  "
            "This causes large gradient magnitudes that prevent ODE convergence.  "
            "Check that ne is in m⁻³ (not cm⁻³ — convert with ne_cc * 1e6) and "
            "that your domain does not include over-critical cells.",
            stacklevel=2,
        )

    # Compute the scalar coefficient in float64 to preserve precision, then
    # cast to the dtype of ne to avoid widening the array unnecessarily.
    coeff = np.float64(-0.5) * float(c) ** 2 / (3.14207787e-4 * float(omega) ** 2)
    gradient_term = ne * ne.dtype.type(coeff)
    dndx = jnp.gradient(gradient_term, x, axis=0)
    dndy = jnp.gradient(gradient_term, y, axis=1)
    dndz = jnp.gradient(gradient_term, z, axis=2)
    return dndx, dndy, dndz


def dndr(r, dndx, dndy, dndz, x, y, z):
    """
    Returns the gradient of the refractive-index driving term at the ray
    positions *r* by trilinear interpolation of the pre-computed gradient grids.

    Args:
        r (jax.Array): Shape ``(N, 3)`` — N ray positions ``[x, y, z]``.
        dndx, dndy, dndz (jax.Array): Pre-computed gradient grids of shape
            ``(Nx, Ny, Nz)``, produced by :func:`precompute_gradients`.
        x, y, z (jax.Array): 1-D coordinate arrays in metres.

    Returns:
        jax.Array: Shape ``(3, N)`` — gradient components at each ray position.
    """
    grad = jnp.zeros_like(r.T)
    grad = grad.at[0, :].set(trilinearInterpolator((x, y, z), dndx, r, fill_value=0.0))
    grad = grad.at[1, :].set(trilinearInterpolator((x, y, z), dndy, r, fill_value=0.0))
    grad = grad.at[2, :].set(trilinearInterpolator((x, y, z), dndz, r, fill_value=0.0))
    return grad

def kappa_inv_brems(ne, Te, Z, omega):
    """
    Compute the inverse bremsstrahlung amplitude absorption rate [1/s] at each
    grid point using the NRL formulary (NRL Plasma Formulary, p.58).

    This follows the same approach as the legacy full_solver implementation.
    The absorption coefficient is always non-negative; it is used in the ray-
    tracing ODE as ``da/dt = -kappa * a`` so that amplitude decreases
    monotonically as rays travel through the plasma.

    Args:
        ne  (jax.Array or float): Electron density in m\ :sup:`-3`.
        Te  (jax.Array or float): Electron temperature **in eV** (not Kelvin,
            not Joules).  Must be strictly positive.  May be a scalar (uniform
            temperature) or a 3-D array with the same shape as *ne*.
        Z   (jax.Array or float): Mean ion charge state (dimensionless).  May be a scalar
            (uniform charge state) or a 3-D array with the same shape as *ne*.
        omega (float): Laser angular frequency in rad/s.

    Returns:
        jax.Array: Amplitude absorption rate with the same shape as *ne*, units of 1/s.

    Note:
        The Coulomb logarithm uses the classical minimum impact parameter
        ``b_min = Z·e / (4πε₀·Tₑ[eV])`` [m] throughout.  This is the
        simplest NRL formulation and matches the legacy full_solver behaviour.
        The density must be in m\ :sup:`-3`, temperature in eV, and Z is the
        mean ion charge state.
    """
    from scipy.constants import e as e_charge, epsilon_0

    ne_cc = ne * 1e-6  # convert m^-3 to cm^-3

    # Electron thermal speed (m/s), Te in eV
    v_the = 4.19e5 * jnp.sqrt(Te)

    # Plasma frequency (rad/s), ne_cc in cm^-3
    o_pe = 5.64e4 * jnp.sqrt(ne_cc)

    # Upper limit for Coulomb logarithm argument: max(omega_pe, omega)
    o_max = jnp.maximum(o_pe, omega)

    # Classical minimum impact parameter: b_min = Z·e / (4πε₀·T_e[eV])  [m]
    b_min = Z * e_charge / (4.0 * np.pi * epsilon_0 * Te)

    # Coulomb logarithm (clamped to ≥ 2)
    CL = jnp.maximum(2.0, jnp.log(v_the / (o_max * b_min)))

    return 3.1e-5 * Z * c * (ne_cc / omega) ** 2 * CL * Te ** (-1.5)


# ODEs of photon paths, standalone function to support the solve()
def dsdt(t, s, parallelise, dndx, dndy, dndz, x, y, z, lengths, dims, kappa=None):
    """
    Returns an array with the gradients and velocity per ray for ode_int.

    Accepts pre-computed gradient grids ``dndx``, ``dndy``, ``dndz`` (produced
    by :func:`precompute_gradients`) rather than the raw electron-density array
    and laser frequency.  Computing the spatial gradients of the driving term
    once before the ODE loop — rather than at every adaptive time step — is the
    primary performance optimisation for large-scale runs.

    When *kappa* is ``None`` the state vector has 6 elements per ray
    ``(x, y, z, vx, vy, vz)``.  When *kappa* is a 3-D JAX array (the
    pre-computed inverse bremsstrahlung absorption rate) the state vector has 7
    elements per ray, with the 7th element being the accumulated **optical depth**
    ``τ = ∫κ dt``.  The ODE is ``dτ/dt = κ(r)`` — a non-stiff, purely
    positive accumulation.  Amplitude is recovered at output as ``a = exp(-τ)``.

    Tracking τ rather than amplitude directly avoids the stiffness that the
    equivalent amplitude ODE ``da/dt = -κ·a`` would introduce: that equation has
    eigenvalue ``-κ``, forcing an explicit solver (Tsit5/RK45) to take step
    sizes ``Δt ≲ C/κ_norm``.  For realistic FLASH plasmas this can be
    ``~10¹¹–10¹⁴ s⁻¹``, leading to millions of tiny steps or a solver hang.

    Args:
        t (float): Dummy time variable (problem is time-invariant).
        s (jax.Array): Flattened 6N (or 7N) state vector used by the ODE solver.
        parallelise (bool): When ``True`` each vmap call carries a single ray;
            the state is reshaped to ``(6, 1)`` (or ``(7, 1)``).  When ``False``
            the serial path reshapes to ``(6, N)`` (or ``(7, N)``).
        dndx, dndy, dndz (jax.Array): Pre-computed gradient grids of the
            refractive-index driving term, shape ``(Nx, Ny, Nz)``.
        x, y, z (jax.Array): 1-D coordinate arrays in metres.
        lengths, dims (jax.Array): Domain lengths and grid dimensions (passed
            through for legacy compatibility; not used inside this function).
        kappa (jax.Array or None): Pre-computed inverse bremsstrahlung absorption
            rate grid [1/s].  Pass ``None`` (default) to disable inverse
            bremsstrahlung.  When provided, state element 6 is the accumulated
            optical depth ``τ`` (initialised to 0); amplitude is ``exp(-τ)``.

    Returns:
        jax.Array: Flattened 6N (or 7N) derivative array.
    """

    nstate = 7 if kappa is not None else 6

    if not parallelise:
        # jnp.reshape() auto converts to a jax array rather than having to do after a numpy reshape
        s = jnp.reshape(s, (nstate, s.size // nstate))
    else:
        # forces s to be a matrix even if has the indexes of a 1d array such that dsdt() can be generalised
        s = jnp.reshape(s, (nstate, 1))  # one ray per vmap iteration if parallelised

    sprime = jnp.zeros_like(s)

    # Position and velocity
    # needs to be before the reshape to avoid indexing errors
    r = s[:3, :].T  # transposed so it is of the correct shape for interpolators
    v = s[3:6, :]

    del s

    # must unpack x, y, z tuple here for the sake of dndr, could be earlier but this is easier to pass and more generalised
    # r must be transposed within dndr(...) else we get an AbstractTerm error due to the effect on the return value
    sprime = sprime.at[3:6, :].set(dndr(r, dndx, dndy, dndz, x, y, z))
    sprime = sprime.at[:3, :].set(v)

    # Inverse bremsstrahlung: accumulate optical depth τ.  dτ/dt = κ(r).
    # This is non-stiff (κ ≥ 0, no exponential feedback), unlike the
    # equivalent amplitude ODE da/dt = -κ·a which has eigenvalue -κ and
    # forces tiny step sizes for large κ.  Amplitude is recovered as exp(-τ).
    if kappa is not None:
        kappa_at_r = trilinearInterpolator((x, y, z), kappa, r, fill_value=0.0)
        sprime = sprime.at[6, :].set(kappa_at_r)

    # Keep derivative shape consistent with solver state shape (flattened 1D state vector).
    return jnp.ravel(sprime)
   
def process_results(solutions, depth_traced, trace_depth, probing_direction, duration, save_points_per_region, ray_batch_count, verbose, inv_brems=False):
    """
    #for i in enumerate(sol.result):
    #    print(i)
    for idx, result in enumerate(sol.result):
        # Check if each result is successful
        if result.success:
            print(f"Solution at index {idx} succeeded.")
        else:
            print(f"Solution at index {idx} failed.")

    #print(next(sol.result))
    #print(next(sol.result))
    #print(type(sol.result[0]))  # Check the type of results
    """

    #else:
    #    print("Ray tracer failed. This could be a case of diffrax exceeding max steps again due to apparent 'strictness' compared to solve_ivp, check error log.")

    #if sol.result == RESULTS.successful:
    #rf = sol.ys[:, -1, :].reshape(6, Np)# / scalar

    if ray_batch_count > 1:
        # Concatenate time and state arrays
        ts = jnp.concatenate([sol.ts for sol in solutions], axis = 0)
        ys = jnp.concatenate([sol.ys for sol in solutions], axis = 0)

        # Combine stats
        stats_keys = solutions[0].stats.keys()
        stats = {
            key: jnp.concatenate([sol.stats[key] for sol in solutions], axis = 0)
            for key in stats_keys
        }

        # Combine other fields
        t0 = solutions[0].t0
        t1 = solutions[-1].t1
        result = solutions[-1].result  # Use the last result

        del solutions

        # if info is missing that you need, this is why - implement it !
        solutions = Solution(
            t0 = t0,
            t1 = t1,
            ts = ts,
            ys = ys,
            interpolation = None,  # Optional: you can implement logic to keep interpolations
            stats = stats,
            result = result,
            solver_state = None,
            controller_state = None,
            made_jump = None,
            event_mask = None
        )

        solutions = np.asarray([solutions], dtype = Solution)

    if verbose:
        print("\nParallelised output has resulting 3D matrix of form: [batch_count, (save_points_per_region - 1) * ScalarDomain.region_count, 6]:", solutions[0].ys.shape)
        print(" - 2 to account for the start and end results (typical, can be greater if set)")
        print(" - 6 containing the 3 position and velocity components")
        print(" - If batch_count is lower than expected, this is likely due to jax's forced integer batch sharding requirement over cpu cores.")

        print("\nWe slice the", end = " ")
        if len(solutions[0].ys.shape) == 3:
            print("results", end = " ")
        else:
            print("end result", end = " ")
        print("and transpose into the form:", solutions[0].ys.shape, "to work with later code.")

    if save_points_per_region == 2 or save_points_per_region == 1:
        rf_state = solutions[0].ys[:, -1, :].T  # shape (6 or 7, N)

        # Use keep_current_plane=True so that rays are recorded at their actual
        # positions — consistent with trace_and_save_depths.
        rf_geo, _ = ray_to_Jonesvector(rf_state[:6, :], keep_current_plane=True,
                                        probing_direction=probing_direction)

        if inv_brems and rf_state.shape[0] == 7:
            amp = np.asarray(np.exp(-rf_state[6, :]))  # a = exp(-τ), shape (N,)
            return np.asarray(rf_geo), amp, duration
        return rf_geo, None, duration
    elif save_points_per_region > 2:
        slice_rf_list = []
        slice_Jf_list = []

        for i in range(len(solutions)):
            #save_point_depth = depth_traced
            for j in range(save_points_per_region):
                '''
                if j == save_points_per_region - 1:
                    save_point_depth = depth_traced + trace_depth
                else:
                    save_point_depth += trace_depth // save_points_per_region
                '''

                if j < save_points_per_region - 1 or (j == save_points_per_region - 1 and i == len(solutions) - 1):
                    # sol.ts having shape of (Np, save_points_per_region) per region is very inefficent given there are N - 1 duplications
                    # - issue with diffrax though I can't fix this
                    rf_slice, Jf_slice = ray_to_Jonesvector(solutions[i].ys[:, j, :].T, ne_extent = depth_traced + trace_depth * solutions[i].ts[0, j], probing_direction = probing_direction, keep_current_plane = True)

                    slice_rf_list.append(rf_slice)
                    if Jf_slice is not None:
                        slice_Jf_list.append(Jf_slice)

        rf = jnp.stack(slice_rf_list, axis = 0)
        del slice_rf_list

        if len(slice_Jf_list) > 0:
            Jf = jnp.stack(slice_Jf_list, axis = 0)
            del slice_Jf_list
        else:
            Jf = None

        return rf, Jf, duration
    else:
        assert "\nWhat."

def solve(beam, ScalarDomain, probing_depth, *, parallelise = True, jitted = True, save_points_per_region = 2, memory_debug = False, lwl = 1064e-9, keep_domain = False, return_raw_results = False, verbose = True):
    """
    Trace rays through a scalar plasma domain and return their final state as a
    Jones vector suitable for downstream diagnostics.

    The beam can be supplied either as a pre-created ray matrix or as a compact
    parameter tuple that is expanded into rays internally (necessary when domain
    or ray batching is enabled).

    Args:
        beam (jax.Array or tuple): Either

            * a ``(6, N)`` array of pre-created rays with rows
              ``(x, y, z, vx, vy, vz)``; or
            * a tuple ``(beam_size, divergence, ne_extent, probing_direction,
              beam_type, seeded)`` whose elements are passed directly to
              ``core.beam.Beam``.  This form is required when
              ``ScalarDomain.ray_batch_count > 1``.

        ScalarDomain (core.domain.ScalarDomain): Domain object produced by
            ``core.domain.ScalarDomain``.  Its ``region_count`` and
            ``ray_batch_count`` attributes control domain and ray batching
            respectively.
        probing_depth (float): Maximum propagation depth in metres along the
            probing direction.
        parallelise (bool): Use ``jax.vmap`` to parallelise over rays (default
            ``True``).  Set to ``False`` to use the legacy serial solver
            (single domain region only).
        jitted (bool): JIT-compile the ODE solver with ``equinox.filter_jit``
            (default ``True``).
        save_points_per_region (int): Number of time-points saved per domain
            region by the ODE solver (default ``2``, i.e. start and end).
            Values greater than 2 return intermediate save points as a stacked
            array.
        memory_debug (bool): Print memory diagnostics and write a JAX device
            memory profile to disk (default ``False``).
        lwl (float): Laser wavelength in metres (default ``1064e-9``).  Used to
            compute the angular frequency ``omega = 2π·c/lwl``.
        keep_domain (bool): Reserved for future use (default ``False``).
        return_raw_results (bool): Return the raw ``diffrax.Solution`` objects
            instead of the processed Jones vector (default ``False``).
        verbose (bool): Print progress and shape information (default ``True``).

    Returns:
        tuple: ``(rf, amp, duration)``

            * ``rf`` – Geometric Jones vector of shape ``(4, N)``.  Rows are
              transverse position and angle pairs; the exact mapping depends on
              ``probing_direction`` (see ``shared.propagation.ray_to_Jonesvector``).
              Absorption does **not** modify these values; ``rf`` contains the
              pure geometric ray state regardless of ``inv_brems``.
            * ``amp`` – When ``ScalarDomain.inv_brems=True``: a ``(N,)`` array
              of per-ray amplitudes ``exp(-τ)``.  ``None`` otherwise.
            * ``duration`` – wall-clock time of the ODE solve in seconds
              (``numpy.float64``).

            When ``return_raw_results=True`` the tuple is
            ``(solutions, None, duration)`` where ``solutions`` is a
            ``numpy`` array of ``diffrax.Solution`` objects.

    Example::

        import jax.numpy as jnp
        import core.domain as d
        import core.propagator as p

        # --- domain ---
        lwl              = 1064e-9          # laser wavelength (m)
        probing_direction = 'z'
        Np               = int(1e5)

        domain = d.ScalarDomain(
            lengths, dims,
            leeway_factor     = 3,
            ne_type           = "import",
            probing_direction = probing_direction,
            Np                = Np,
            ne                = ne.v * 1e6,   # electron density in m⁻³
        )

        # --- beam ---
        beam_size      = [extent_x, extent_y]  # half-widths (m)
        probing_extent = extent_z
        ne_extent      = probing_extent         # initialisation depth
        divergence     = 0.1e-3                 # half-angle (rad)
        beam_type      = "rectangular"

        # --- solve ---
        rf_jax, _, duration = p.solve(
            (beam_size, divergence, ne_extent, probing_direction, beam_type, False),
            domain,
            probing_extent,
            lwl     = lwl,
            verbose = False,
        )

        # rf_jax has shape (4, Np): rows are x, φ_x, y, φ_y
        # Pass to diagnostics, e.g.:
        #   import processing.diagnostics as diag
        #   shadowgrapher = diag.Shadowgraphy(rf_jax, focal_plane=-35)
        #   shadowgrapher.single_lens_solve()
    """

    omega = 2 * jnp.pi * (c / lwl)

    region_count = ScalarDomain.region_count
    ray_batch_count = ScalarDomain.ray_batch_count

    print("\nNumber of domain batches:", region_count)
    print("Number of ray batches:", ray_batch_count)

    from core.beam import Beam
    assert not isinstance(beam, Beam), "\nThis function does not take in the direct output of the Beam object, pass either Beam.s0 rays, or the parameters passed to be Beam here as a tuple if batching rays."

    unbatched_beam = False
    if ray_batch_count == 1:
        import array
        if isinstance(beam, array.array) or isinstance(beam, np.ndarray) or isinstance(beam, jax.Array):
            assert len(beam.shape) == 2, "\nExpected a matrix of pre-created rays."

            s0_import = beam
            del beam

            Np = s0_import.shape[1]

            Np_total = Np
            rays_per_batch = Np # not necessary, just so there is something to print if someone tries

            rays = np.array([Np], dtype = np.int64)
        elif isinstance(beam, tuple):
            unbatched_beam = True

            print("\nUsing tuple values to create the unbatched beam, domain must be used in the same fashion.")

            Np_total = ScalarDomain.Np_total
            rays_per_batch = Np_total

            rays = np.array([Np_total], dtype = np.int64)
    else:
        assert isinstance(beam, tuple), "\nExpect a tuple of Beam properties if you wish to batch rays."

        Np_total = ScalarDomain.Np_total

        #Np = Np_total // ray_batch_count
        rays_per_batch = Np_total // ray_batch_count
        rays = np.array([rays_per_batch] * (ray_batch_count - 1) + [Np_total - rays_per_batch * (ray_batch_count - 1)], dtype = np.int64)

    # s0_import[:, 0] and s0_import input to getsizeof_default(...) produce the same result
    # I think this estimation is correct, if jax reports failing to allocate a lower amount, check the amount reported isn't just the max memory available
    # if it is, estimation is likely correct and this is just an issue with reporting
    # if it is lower, you likely have a memory leak
    # this is relevant generally not just for ray memory - just cropped up as an issue here first

    # if batched: or if auto_batching: etc.
    # proing_depth /= some integer with some corrections I expect
    # make logic too loop it and pick up from previous solution

    duration = np.float64(0.0)
    solutions = np.empty(ray_batch_count, dtype = Solution)

    for ray_index, Np in enumerate(rays):
        depth_traced = 0.0

        if ray_batch_count > 1 or unbatched_beam:
            s0_import = Beam(Np, beam_size = beam[0], divergence = beam[1], ne_extent = beam[2], probing_direction = beam[3], beam_type = beam[4], seeded = beam[5]).s0

        single_ray_size = getsizeof_default(s0_import[:, 0])
        print("\nEst. size in memory of rays (1 = {}): {}".format(mem_conversion(single_ray_size), mem_conversion(single_ray_size * Np)))
        total_ray_size_estimate_raw = getsizeof_default(s0_import[:, 0]) * Np_total
        if ray_batch_count > 1:
            print("Est. potential size in memory of total rays:", mem_conversion(total_ray_size_estimate_raw))
            print(" --> Np (total) = {} (in {} batches) - {} for this batch".format(Np_total, ray_batch_count, Np))
        else:
            print(" --> Np = {}".format(Np))

        for i in range(1, ScalarDomain.region_count + 1):
            if ScalarDomain.region_count == 1:
                print("\nNo need to generate any sections of the domain, batching not utilised.")

                trace_depth = probing_depth
            else:
                if i == 1:
                    print("\nUsing pre-generated 1st section of domain.")
                else:
                    print("\nGenerating", add_integer_postfix(i), "section of the domain...")

                    lengths = ScalarDomain.lengths
                    dims = ScalarDomain.dims

                    ne_type = ScalarDomain.ne_type

                    probing_direction = ScalarDomain.probing_direction

                    region_count = ScalarDomain.region_count

                    leeway_factor = ScalarDomain.leeway_factor

                    coord_backup = ScalarDomain.coord_backup
                    future_dims = ScalarDomain.future_dims

                    try:
                        del ScalarDomain
                    except:
                        ScalarDomain = None

                    import core.domain as d
                    ScalarDomain = d.ScalarDomain(
                        lengths, dims,
                        ne_type = ne_type,
                        probing_direction = probing_direction,
                        auto_batching = True,
                        iteration = i,
                        region_count = region_count,
                        leeway_factor = leeway_factor,
                        coord_backup = coord_backup,
                        future_dims = future_dims
                    )

                    del lengths
                    del dims

                    del ne_type

                    del densities

                    del probing_direction

                    del region_count

                    del leeway_factor

                    del coord_backup
                    del future_dims

                # Need to make sure all rays have left volume
                # Conservative estimate of diagonal across volume
                # Then can backproject to surface of volume

                depth_remaining = probing_depth - depth_traced

                trace_depth = ScalarDomain.lengths[['x', 'y', 'z'].index(ScalarDomain.probing_direction)]
                if trace_depth > depth_remaining:
                    trace_depth = depth_remaining

                del depth_remaining

            target_depth = trace_depth + depth_traced

            # it isn't tracing up till this depth, it is tracing this amount further
            # at end positions are r(vector) + trace_depth (ish) NOT trace_depth(vector)
            print(" --> tracing a depth of", trace_depth, "mm's to the target depth of", target_depth, "mm's")

            t = jnp.linspace(0.0, jnp.sqrt(8.0) * trace_depth / c, 2)
            norm_factor = jnp.max(t)

            # 8.0^0.5 is an arbritrary factor to ensure rays have enough time to escape the box
            # think we should change this???

            # passed args must be hashable to be made static for jax.jit, tuple is hashable, array & dict are not
            inv_brems = getattr(ScalarDomain, 'inv_brems', False)
            kappa_grid = None
            if inv_brems:
                kappa_grid = kappa_inv_brems(
                    ScalarDomain.ne,
                    ScalarDomain.Te,
                    ScalarDomain.Z,
                    omega,
                )

            # Pre-compute gradient grids once here so the ODE body only does
            # trilinear interpolation at each adaptive step, not full-grid
            # jnp.gradient calls.
            dndx, dndy, dndz = precompute_gradients(
                ScalarDomain.ne,
                ScalarDomain.x, ScalarDomain.y, ScalarDomain.z,
                omega,
            )

            args = (
                parallelise,
                dndx, dndy, dndz,
                ScalarDomain.x, ScalarDomain.y, ScalarDomain.z,
                ScalarDomain.lengths, ScalarDomain.dims,
                kappa_grid,
            )

            ###
            ### Check the original algorithm still works for the sake of testing
            ###

            if not parallelise:
                from numpy import array

                assert i == 1, "\nDomain batching is not set up to work with the legacy solver yet."

                s0 = array(jnp.ravel(s0_import))
                #s0 = s0.flatten() #odeint insists

                '''
                # need a backpropogation algorithm that works for this too
                s0 = array(jnp.ravel(sol))
                del sol
                '''

                start = time()
                # wrapper allows dummy variables t & y to be used by solve_ivp(), self is required by dsdt
                sol = solve_ivp(lambda t, y: dsdt(t, y, *args), [0, t[-1]], s0, t_eval = t)
            else:
                # transposed as jax.vmap() expects form of [batch_idx, items] not [items, batch_idx]
                available_devices = jax.devices()
                running_device = jax.default_backend()
                # running_device = jax.lib.xla_bridge.get_backend().platform # - deprecated, using still as needed for HPC
                #running_device = jax.extend.backend.get_backend().platform
                print("\nRunning device:", running_device, end='')

                if i == 1:
                    s0_transformed = s0_import.T
                    # When inv_brems is enabled, append an optical-depth column (0.0).
                    # State[6] accumulates τ = ∫κ dt; amplitude is recovered as exp(-τ).
                    if inv_brems:
                        tau_init = jnp.zeros((Np, 1), dtype=s0_import.dtype)
                        s0_transformed = jnp.concatenate([s0_transformed, tau_init], axis=1)
                    del s0_import
                else:
                    # change target_depth back to trace_depth and check the difference
                    s0_transformed = back_propogate(sol.ys[:, -1, :].T, target_depth, ScalarDomain.probing_direction).T
                    del sol

                if running_device == 'cpu':
                    cpu_devices = jax.devices('cpu')
                    core_count = len(cpu_devices)
                    print(", with:", core_count, "cores.")

                    # Avoid manual NamedSharding mesh setup here; recent JAX versions can
                    # raise mesh-axis errors in this path. vmap still parallelises compute.
                    s0 = jax.device_put(s0_transformed)

                    if Np < core_count:
                        print(colour.BOLD + "Not enough rays to parallelise over cores" + colour.END + ": increase to at least " + str(core_count) + " to utilise parallelisation")
                        print(" --> Running CPU processes sequentially")
                elif running_device == 'gpu':
                    gpu_devices = jax.devices('gpu')
                    print("\nThere are", len(gpu_devices), "available GPU devices:", gpu_devices)
                    assert len(gpu_devices) > 0, "Running on GPU yet none detected?"

                    s0 = jax.device_put(s0_transformed, gpu_devices[0])
                elif running_device == 'tpu':
                    pass

                    s0 = s0_transformed
                else:
                    assert "No suitable device detected!"

                del s0_transformed
                # optional for aggressive cleanup?
                #jax.clear_caches()

                # wrapper for same reason, diffrax.ODETerm instantiaties this and passes args
                # I have no idea why, but this has to be defined in solve rather than as a global function - else there is an abstract variable error
                def dsdt_ODE(t, y, args):
                    return dsdt(t, y, *args) * norm_factor

                from diffrax import ODETerm, Tsit5, SaveAt, PIDController, diffeqsolve
                #import optax - diffrax uses as a dependency, don't need to import directly

                # using lengths and/or dims to set parameters of diffeqsolve(...) results in BooleanConversionError due to tracing variable resolution
                # rtol & atol are good here - setting too precise increases runtime dramatically for little change in results, it overcompensates
                def diffrax_solve(dydt, t0, t1, Nt, lengths, dims, *, rtol = 1e-3, atol = 1e-5):
                    """
                    Here we wrap the diffrax diffeqsolve function such that we can easily parallelise it
                    """

                    # We convert our python function to a diffrax ODETerm
                    # should use the function passed into the wrapper - not the local definition
                    term = ODETerm(dydt)

                    # We chose a solver (time-stepping) method from within diffrax library
                    solver = Tsit5() # (RK45 - closest I could find to solve_ivp's default method)

                    # At what time points you want to save the solution
                    saveat = SaveAt(ts = jnp.linspace(t0, t1, Nt))
        
                    # Diffrax uses adaptive time stepping to gain accuracy within certain tolerances
                    # setting dtmax increases runtime significantly - maybe this is too high and thus calculations are not precise due to scale of change?
                    #dtmax = 0.5 * ((lengths[0] / dims[0])**2 + (lengths[1] / dims[1])**2 + (lengths[2] / dims[2])**2) ** (1 / 2) / (c * norm_factor)
                    stepsize_controller = PIDController(rtol = rtol, atol = atol)#, dtmax = dtmax)

                    return lambda s0, args : diffeqsolve(
                        term,
                        solver,
                        y0 = jnp.array(s0),
                        args = args,# + (atten, ),
                        t0 = t0,
                        t1 = t1,
                        # None (leaving up to controller) shows better performance than setting ourselves
                        dt0 = None,#(t1 - t0) * norm_factor / Nt, # can set = 0 if dtmax is set apparently?
                        saveat = saveat,
                        stepsize_controller = stepsize_controller,
                        # set max steps to no. of cells x100
                        # cannot be passed as dims --> causes boolean conversion error, has to be passed directly
                        # need to pass this correctly so that it remains consistent with class when batching
                        max_steps = int(2e8) #dims[0] * dims[1] * dims[2] * 100 #10000 - default for solve_ivp?????
                    ) # the 2e8 choice is very arbritrary

                # hardcode to normalise to 1 due to diffrax bug
                ODE_solve = diffrax_solve(dsdt_ODE, 0, 1, save_points_per_region, ScalarDomain.lengths, ScalarDomain.dims)

                if jitted:
                    start_comp = time()

                    from equinox import filter_jit
                    # equinox.filter_jit() (imported as filter_jit()) provides debugging info unlike jax.jit() - it does not like static args though so sticking with jit for now
                    #ODE_solve = jax.jit(ODE_solve)#, static_argnums = 1)#, device = available_devices[0])
                    ODE_solve = filter_jit(ODE_solve)#, device = available_devices[0])
                    # not sure about the performance of non-static specified arguments with filter_jit() - only use for debugging not in 'production'

                    print("\njax compilation of solver took:", time() - start_comp, "seconds", end='')

                # pass s0[:, i] for each ray via a jax.vmap for parallelisation
                start = time()

                sol = jax.block_until_ready(
                    # in_axes version ensures that vmap doesn't map args parameters, just s0
                    #jax.vmap(lambda rays, args: ODE_solve, in_axes = (0, None))(s0, args)

                    # default vmap_method argument is sequential, this is deprecated though and will cause a warning (if debugging) past jax 0.6.0
                    # look into different options for this parameter at a later date

                    jax.vmap(ODE_solve, in_axes = (0, None))(s0, args)
                )

            duration += np.float64(time() - start)

            if memory_debug:
                if parallelise:
                    # Visualises sharding, looks cool, but pretty useless - and a pain with higher core counts
                    jax.debug.visualize_array_sharding(sol.ys[:, -1, :])

                from utils import domain_estimate

                print(colour.BOLD + "\nMemory summary - total estimate:", mem_conversion(domain_estimate(ScalarDomain.dims) + (getsizeof_default(s0) + getsizeof_default(sol)) * Np) + colour.END)
                print("\nEst. size of domain:", mem_conversion(getsizeof_default(s0) * Np))
                print("Est. size of initial rays:", mem_conversion(getsizeof_default(s0) * Np))
                print("Est. size of solution class / single ray (?):", getsizeof(sol))
                print("Est. size of solution (bef. JV):", mem_conversion(getsizeof_default(sol) * Np))

                folder_name = "memory"
                postfix = "_benchmarks/"

                path = "evaluation/benchmarks/" + folder_name + "/"

                if os.path.isdir(os.getcwd() + "/" + path):
                    pass
                else:
                    path = os.getcwd() + "/../" + folder_name + postfix

                    if os.path.isdir(path):
                        pass
                    else:
                        try:
                            os.mkdir(path)
                        except OSError as e:
                            import errno

                            print("\nFailed to create folder above current working directory, attempting in cwd:")

                            path = os.getcwd() + "/" + folder_name + postfix

                            if os.path.isdir(path):
                                path = folder_name + postfix
                            else:
                                try:
                                    os.mkdir(path)
                                except OSError as e:
                                    print("\nFailed in cwd too! No folder created.")
                                    if e.errno != errno.EEXIST:
                                        raise

                                #if e.errno != errno.EEXIST:
                                #    raise

                from datetime import datetime
                path += "memory-domain" + str(ScalarDomain.dims[0]) + "_rays"+ str(s0.shape[1]) + "-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".prof"
                jax.profiler.save_device_memory_profile(path)

                print("\n", end = '')
                if os.path.isfile(os.path.expanduser("~") + "/go/bin/pprof"):
                    #import sys
                    from os import system

                    #system(f"~/go/bin/pprof -top {sys.executable} memory_{N}.prof")
                    system(f"~/go/bin/pprof -top /bin/ls " + path)
                    #system(f"~/go/bin/pprof --web " + path)
                else:
                    print("No pprof install detected. Please download to visualise memory usage - requires Golang to run.")

            ###
            ### Test if streaming is still the source of memory issues by using, del s0 test again
            ###

            #del s0

            #del sol - # this (and commenting out below section) prevents memory issues, so clearly solutions[...] needs to be
            # forced written to storage if over a certain memory limit

            # if est. solutions < ram but > vram, write to ram
            # if > both, write to storage
            # if < vram, keep on gpu - but then it wouldn't be batched anyway so sort of irrelevant

            if i == ScalarDomain.region_count:
                from shared.utils import memory_report

                if total_ray_size_estimate_raw >= memory_report("cpu")['free_raw']:
                    target_folder = os.getcwd() + "/saves"
                    if not os.path.isdir(target_folder):
                        try:
                            os.mkdir(target_folder)
                        except OSError as e:
                            print("\nFailed to create folder at " + target_folder)
                            if e.errno != errno.EEXIST:
                                raise

                    tar_gz_path = target_folder + "/ray_output_total_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".hdf5.tar.gz"

                    '''
                    from utils.handle_filetypes import save_jax_matrix_to_hdf5 as compressed_solution_export
                    filepath, filename = compressed_solution_export(
                        ray_to_Jonesvector(sol.ys[:,-1].reshape(6, Np), ne_extent = probing_depth, probing_direction = ScalarDomain.probing_direction)[0],
                        file_path = target_folder
                        #filename = None, file_path = ".", dataset_name = 'data', compression = 'gzip', compression_level = 4
                    )

                    from utils.handle_filetypes import move_file_to_tar_gz
                    move_file_to_tar_gz(tar_gz_path, filepath)
                    '''

                    from utils.handle_filetypes import compress_matrix_to_hdf5_BytesIO
                    from utils.handle_filetypes import stream_data_to_tar_gz

                    filename = "run_" + str(ray_index)
                    stream_data_to_tar_gz(tar_gz_path, filename,
                        compress_matrix_to_hdf5_BytesIO(
                            ray_to_Jonesvector(sol.ys[:,-1].reshape(6, Np), ne_extent = probing_depth, probing_direction = ScalarDomain.probing_direction)[0]
                        )
                    )
                else:
                    solutions[ray_index] = sol
                    del sol

            depth_traced += trace_depth

    print("\nCompleted ray trace in", colour.BOLD + str(np.round(duration, 3).astype(np.float64)) + colour.END, "seconds.")

    if total_ray_size_estimate_raw < memory_report("cpu")['free_raw']:
        if return_raw_results:
            return solutions, None, duration
        else:
            if not parallelise:
                rf_geo, _ = ray_to_Jonesvector(solutions.ys[:,-1].reshape(6, Np), keep_current_plane=True, probing_direction = ScalarDomain.probing_direction)
                return rf_geo, None, duration
            else:
                # need to confirm there is no mismatch between total depth_traced and the target probing_depth
                return process_results(solutions, depth_traced, trace_depth, ScalarDomain.probing_direction, duration, save_points_per_region, ray_batch_count, verbose, inv_brems=inv_brems)
    else:
        print("\nData output as a hdf4.tar.gz file due to limitations of vram/ram space.")
        print("Graphs can be iteratively plotted by cycling through the 'run_n' entries after extraction from .tar.gz format.")


def trace_and_save_depths(beam, ScalarDomain, step, depth_max, output_path, *,
                          lwl=1064e-9, jones_components=None, jitted=True,
                          rtol=1e-3, atol=1e-5, verbose=True):
    """
    Trace rays through the domain and save the Jones vector at uniformly-spaced
    depths along the propagation direction, then pickle the results.

    Works for any probing direction ('x', 'y', or 'z'); the direction is read
    from ``ScalarDomain.probing_direction``.

    The Jones vector at each depth is a 4-row array::

        row 0 – transverse position along the first  transverse axis
        row 1 – angle        along the first  transverse axis
        row 2 – transverse position along the second transverse axis
        row 3 – angle        along the second transverse axis

    The mapping between rows and physical axes depends on the probing direction:

    ==================  =======  =======  =======  =======
    probing_direction   row 0    row 1    row 2    row 3
    ==================  =======  =======  =======  =======
    'z'                 x        φ_x      y        φ_y
    'y'                 x        φ_x      z        φ_z
    'x'                 y        φ_y      z        φ_z
    ==================  =======  =======  =======  =======

    Args:
        beam (jax.Array or tuple): Either

            * a ``(6, N)`` array of pre-created rays with rows
              ``(x, y, z, vx, vy, vz)``; or
            * a tuple ``(beam_size, divergence, ne_extent, probing_direction,
              beam_type, seeded)`` whose elements are passed directly to
              ``core.beam.Beam``.  When a tuple is supplied the number of rays
              is taken from ``ScalarDomain.Np_total``, which must be set.

        ScalarDomain (core.domain.ScalarDomain): Domain object. Its extent in the probing
            direction must be >= depth_max.  When *beam* is a tuple,
            ``ScalarDomain.Np_total`` must be set to the desired number of rays.
        step (float): Cadence of depth saves, in metres (e.g. 200e-6 for 200 µm).
        depth_max (float): Maximum propagation depth to record, in metres (e.g. 1e-3 for 1 mm).
            The domain length in the probing direction must be >= depth_max.
        output_path (str or None): File path for the output pickle. Pass None to skip
            writing the file (results are still returned).
        lwl (float): Laser wavelength in metres (default 1064e-9). Used to compute
            the angular frequency ``omega = 2π·c/lwl``.
        jones_components: Selects which rows of the Jones vector to save.
            Accepted values:

            * ``None`` or ``'all'``  – save all four rows (default).
            * ``'position'``         – save rows 0 and 2 (transverse positions only).
            * ``'angle'``            – save rows 1 and 3 (angles only).
            * list / tuple of ints  – save the specified row indices, e.g. ``[0, 1]``.

        jitted (bool): Whether to JIT-compile the ODE solver (default True).
        rtol (float): Relative ODE tolerance (default 1e-3).
        atol (float): Absolute ODE tolerance (default 1e-5).
        verbose (bool): Print progress information (default True).

    Returns:
        dict: Keys:
            ``depth_saves``         – 1-D numpy array of depth positions along the
                                      probing axis (metres).
            ``jvec``                – list of ``(n_components, N)`` numpy arrays; the
                                      selected Jones-vector rows at each depth.  Always
                                      the **geometric** (ray position / angle) Jones vector
                                      — absorption does not modify these values.
            ``jones_components``    – list of int indices recording which rows were saved.
            ``amplitude``           – *(only when inv_brems=True)* list of ``(N,)`` numpy
                                      arrays containing the per-ray amplitude ``exp(-τ)``
                                      at each depth snapshot, where τ is the accumulated
                                      inverse-bremsstrahlung optical depth.  Multiply
                                      ``jvec[j] * amplitude[j][np.newaxis, :]`` to obtain
                                      the amplitude-weighted Jones vector if needed.

    Raises:
        ValueError: If the domain length in the probing direction is smaller than depth_max.
        ValueError: If step is not positive or depth_max <= 0.
        ValueError: If jones_components contains an index outside [0, 3].

    Example (tuple beam)::

        import core.domain as d
        import core.propagator as p

        # --- parameters ---
        lwl               = 1064e-9          # laser wavelength (m)
        probing_direction = 'z'
        Np                = int(1e5)

        # --- domain (Np stored here, not in the beam tuple) ---
        domain = d.ScalarDomain(
            lengths, dims,
            leeway_factor     = 3,
            ne_type           = "import",
            probing_direction = probing_direction,
            Np                = Np,
            ne                = ne.v * 1e6,  # electron density in m⁻³
        )

        # --- trace and save Jones vector every 200 µm up to 1 mm ---
        result = p.trace_and_save_depths(
            (500e-6, 0.1e-3, probing_extent, probing_direction, "circular", False),
            domain,
            step       = 200e-6,            # save cadence (m)
            depth_max  = 1e-3,              # maximum depth (m)
            output_path= "depth_saves.pkl", # set to None to skip file write
            lwl        = lwl,
            jones_components = 'position',  # save transverse positions only
            verbose    = True,
        )

        # result['depth_saves']  – 1-D array of save depths (m)
        # result['jvec']         – list of (2, Np) arrays at each depth
        # result['jones_components'] – [0, 2]  (rows saved)

        depth_mm = result['depth_saves'] * 1e3
        for depth, jv in zip(depth_mm, result['jvec']):
            x_pos = jv[0]          # transverse x at this depth (m)
            y_pos = jv[1]          # transverse y at this depth (m)
            print(f"depth={depth:.2f} mm  <x>={x_pos.mean()*1e3:.3f} mm")
    """

    import pickle

    omega = 2 * np.pi * (c / lwl)

    # ── Resolve beam: accept either a pre-built (6, N) ray array or a compact
    # parameter tuple (beam_size, divergence, ne_extent, probing_direction,
    # beam_type, seeded).  When a tuple is supplied the number of rays is read
    # from ScalarDomain.Np_total so that Np does not need to appear in the
    # tuple, consistent with how solve() works.
    if isinstance(beam, tuple):
        from core.beam import Beam
        assert getattr(ScalarDomain, 'Np_total', None) is not None, (
            "\nScalarDomain.Np_total must be set when passing beam as a tuple. "
            "Pass Np=<number_of_rays> to ScalarDomain(...)."
        )
        s0 = Beam(
            ScalarDomain.Np_total,
            beam_size         = beam[0],
            divergence        = beam[1],
            ne_extent         = beam[2],
            probing_direction = beam[3],
            beam_type         = beam[4],
            seeded            = beam[5],
        ).s0
    else:
        s0 = beam

    if step <= 0 or depth_max <= 0:
        raise ValueError("step and depth_max must be positive.")

    # ── Parse jones_components ────────────────────────────────────────────────
    if jones_components is None or jones_components == 'all':
        comp_indices = [0, 1, 2, 3]
    elif jones_components == 'position':
        comp_indices = [0, 2]
    elif jones_components == 'angle':
        comp_indices = [1, 3]
    else:
        comp_indices = list(jones_components)
        invalid = [i for i in comp_indices if i not in (0, 1, 2, 3)]
        if invalid:
            raise ValueError(
                f"jones_components indices {invalid} are out of range. "
                f"Valid indices are 0, 1, 2, 3."
            )

    probing_direction = ScalarDomain.probing_direction
    dir_idx = ['x', 'y', 'z'].index(probing_direction)
    trace_depth = float(ScalarDomain.lengths[dir_idx])

    if trace_depth < depth_max:
        raise ValueError(
            f"Domain length in probing direction '{probing_direction}' ({trace_depth:.6g} m) "
            f"is smaller than the requested depth_max ({depth_max:.6g} m). "
            f"Increase the domain size to at least depth_max."
        )

    # Uniform depth save positions: [0, step, 2*step, ..., depth_max]
    n_saves = int(np.round(depth_max / step)) + 1
    depth_saves = np.linspace(0.0, depth_max, n_saves)

    # ── Inverse bremsstrahlung ────────────────────────────────────────────────
    inv_brems = getattr(ScalarDomain, 'inv_brems', False)
    kappa = None
    if inv_brems:
        kappa = kappa_inv_brems(
            ScalarDomain.ne,
            ScalarDomain.Te,
            ScalarDomain.Z,
            omega,
        )

    if verbose:
        print(f"\ntrace_and_save_depths: saving at {n_saves} depth(s) from 0 to {depth_max*1e3:.4g} mm "
              f"(step = {step*1e6:.4g} µm, probing direction = '{probing_direction}', "
              f"jones_components = {comp_indices}, inv_brems = {inv_brems}).")

    # Map depth positions to normalised diffrax time [0, 1].
    # The ODE solver normalises real time by norm_factor = sqrt(8)*trace_depth/c so
    # that t_norm=1 corresponds to the end of the trace.  A ray travelling at ~c
    # covers depth d in real time d/c, giving normalised time d/(sqrt(8)*trace_depth).
    norm_factor = np.sqrt(8.0) * trace_depth / c
    t_saves_norm = depth_saves / (np.sqrt(8.0) * trace_depth)

    Np = s0.shape[1]

    # Pre-compute gradient grids once here so the ODE body only does
    # trilinear interpolation at each adaptive step, not full-grid
    # jnp.gradient calls.
    dndx, dndy, dndz = precompute_gradients(
        ScalarDomain.ne,
        ScalarDomain.x, ScalarDomain.y, ScalarDomain.z,
        omega,
    )

    args = (
        True,  # parallelise=True (each vmap'd call handles one ray)
        dndx, dndy, dndz,
        ScalarDomain.x, ScalarDomain.y, ScalarDomain.z,
        ScalarDomain.lengths, ScalarDomain.dims,
        kappa,
    )

    def dsdt_ODE(t, y, args):
        return dsdt(t, y, *args) * norm_factor

    from diffrax import ODETerm, Tsit5, SaveAt, PIDController, diffeqsolve

    term = ODETerm(dsdt_ODE)
    solver = Tsit5()
    saveat = SaveAt(ts=jnp.array(t_saves_norm))
    stepsize_controller = PIDController(rtol=rtol, atol=atol)

    def _ode_solve(s0_ray, args):
        return diffeqsolve(
            term,
            solver,
            y0=jnp.array(s0_ray),
            args=args,
            t0=float(t_saves_norm[0]),
            t1=float(t_saves_norm[-1]),
            dt0=None,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=int(2e8),
        )

    if jitted:
        from equinox import filter_jit
        _ode_solve = filter_jit(_ode_solve)

    # When inv_brems is enabled, append an optical-depth column (0.0) to each
    # ray so the state vector is (7,) per ray.  State[6] accumulates
    # τ = ∫κ dt; amplitude is recovered as exp(-τ) at output.
    if inv_brems:
        tau_init = jnp.zeros((Np, 1), dtype=s0.dtype)
        s0_T = jnp.concatenate([s0.T, tau_init], axis=1)  # shape (N, 7)
    else:
        s0_T = s0.T  # shape (N, 6)

    sol = jax.block_until_ready(
        jax.vmap(_ode_solve, in_axes=(0, None))(s0_T, args)
    )

    # sol.ys has shape (N, n_saves, 6) [or (N, n_saves, 7) with inv_brems].
    # For each depth snapshot: compute the full 4-row Jones vector via
    # ray_to_Jonesvector (keep_current_plane=True records the actual position
    # and angle at that depth, not propagated to the exit plane), then keep
    # only the user-requested rows.
    # jvec always contains the geometric Jones vector (positions and angles).
    # Absorption changes only ray amplitude, not trajectory — keep them separate.
    jvec_list = []
    amp_list  = [] if inv_brems else None

    for j in range(n_saves):
        rays_j = sol.ys[:, j, :6].T  # shape (6, N)  — geometric state only
        jvec_full, _ = ray_to_Jonesvector(rays_j, keep_current_plane=True,
                                           probing_direction=probing_direction)
        # rows: [pos_axis1, angle_axis1, pos_axis2, angle_axis2]
        jvec_list.append(np.asarray(jvec_full)[comp_indices, :])  # (n_comp, N)

        if inv_brems:
            amp_list.append(np.asarray(np.exp(-sol.ys[:, j, 6])))  # exp(-τ)

    result = {
        'depth_saves': depth_saves,
        'jvec': jvec_list,
        'jones_components': comp_indices,
    }
    if inv_brems:
        result['amplitude'] = amp_list

    if output_path is not None:
        with open(output_path, 'wb') as fh:
            pickle.dump(result, fh)
        if verbose:
            print(f"trace_and_save_depths: results saved to '{output_path}'.")

    return result
