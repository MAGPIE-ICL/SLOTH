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

def dndr(r, gradient_term, omega, x, y, z):
    """
    Returns the gradient at the locations r

    Args:
        r (3xN float): N [x, y, z] locations

    Returns:
        3 x N float: N [dx, dy, dz] electron density gradients
    """

    grad = jnp.zeros_like(r.T)

    dndx = jnp.gradient(gradient_term, x, axis = 0)
    grad = grad.at[0, :].set(trilinearInterpolator((x, y, z), dndx, r, fill_value = 0.0))
    del dndx

    dndy = jnp.gradient(gradient_term, y, axis = 1)
    grad = grad.at[1, :].set(trilinearInterpolator((x, y, z), dndy, r, fill_value = 0.0))
    del dndy

    dndz = jnp.gradient(gradient_term, z, axis = 2)
    grad = grad.at[2, :].set(trilinearInterpolator((x, y, z), dndz, r, fill_value = 0.0))
    del dndz

    return grad

# ODEs of photon paths, standalone function to support the solve()
def dsdt(t, s, parallelise, ne, x, y, z, omega, lengths, dims):
    """
    Returns an array with the gradients and velocity per ray for ode_int

    Args:
        t (float array): I think this is a dummy variable for ode_int - our problem is time invarient
        s (6N float array): flattened 6xN array of rays used by ode_int
        ScalarDomain (ScalarDomain): an ScalarDomain object which can calculate gradients

    Returns:
        6N float array: flattened array for ode_int
    """

    if not parallelise:
        print("False")
        # jnp.reshape() auto converts to a jax array rather than having to do after a numpy reshape
        s = jnp.reshape(s, (6, s.size // 6))
    else:
        print("True")
        # forces s to be a matrix even if has the indexes of a 1d array such that dsdt() can be generalised
        s = jnp.reshape(s, (6, 1))  # one ray per vmap iteration if parallelised

    sprime = jnp.zeros_like(s)

    # Position and velocity
    # needs to be before the reshape to avoid indexing errors
    r = s[:3, :].T  # transposed so it is of the correct shape for interpolators
    v = s[3:6, :]

    # was deleting before it needed using before by accident - obviously caused issues (AbstractTerm error)
    # - fine to delete after used, only one slice of s0 rather than deleting s0
    # although probably really unnecessary?
    del s

    gradient_term = -0.5 * c ** 2 * ne / (3.14207787e-4 * omega ** 2)

    # must unpack x, y, z tuple here for the sake of dndr, could be earlier but this is easier to pass and more generalised
    # r must be transposed within dndr(...) else we get an AbstractTerm error due to the effect on the return value
    sprime = sprime.at[3:6, :].set(dndr(r, gradient_term, omega, x, y, z))
    sprime = sprime.at[:3, :].set(v)

    ###
    ### Sort out passed functions and objects
    ###

    # Keep derivative shape consistent with solver state shape (flattened 1D state vector).
    return jnp.ravel(sprime)
   
def process_results(solutions, depth_traced, trace_depth, probing_direction, duration, save_points_per_region, ray_batch_count, verbose):
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
        rf = solutions[0].ys[:, -1, :].T

        # depth_traced + trace_depth or just trace_depth
        return *ray_to_Jonesvector(rf, ne_extent = depth_traced + trace_depth, probing_direction = probing_direction), duration
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
            args = (
                parallelise, 
                ScalarDomain.ne,
                ScalarDomain.x, ScalarDomain.y, ScalarDomain.z,
                omega, 
                ScalarDomain.lengths, ScalarDomain.dims
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
                return *ray_to_Jonesvector(solutions.ys[:,-1].reshape(6, Np), ne_extent = probing_depth, probing_direction = ScalarDomain.probing_direction), duration
            else:
                # need to confirm there is no mismatch between total depth_traced and the target probing_depth
                return process_results(solutions, depth_traced, trace_depth, ScalarDomain.probing_direction, duration, save_points_per_region, ray_batch_count, verbose)
    else:
        print("\nData output as a hdf4.tar.gz file due to limitations of vram/ram space.")
        print("Graphs can be iteratively plotted by cycling through the 'run_n' entries after extraction from .tar.gz format.")


def trace_and_save_depths(s0, ScalarDomain, z_step, z_max, output_path, *,
                          omega, jones_components=None, jitted=True,
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
        s0 (jax.Array): Initial ray state, shape (6, N). Rows are (x, y, z, vx, vy, vz).
        ScalarDomain (core.domain.ScalarDomain): Domain object. Its extent in the probing
            direction must be >= z_max.
        z_step (float): Cadence of depth saves, in metres (e.g. 200e-6 for 200 µm).
        z_max (float): Maximum propagation depth to record, in metres (e.g. 1e-3 for 1 mm).
            The domain length in the probing direction must be >= z_max.
        output_path (str or None): File path for the output pickle. Pass None to skip
            writing the file (results are still returned).
        omega (float): Angular frequency of the probing beam, rad/s.
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
            ``depth_saves``       – 1-D numpy array of depth positions along the
                                    probing axis (metres).
            ``jvec``              – list of ``(n_components, N)`` numpy arrays; the
                                    selected Jones-vector rows at each depth.
            ``jones_components``  – list of int indices recording which rows were saved.

    Raises:
        ValueError: If the domain length in the probing direction is smaller than z_max.
        ValueError: If z_step is not positive or z_max <= 0.
        ValueError: If jones_components contains an index outside [0, 3].
    """

    import pickle

    if z_step <= 0 or z_max <= 0:
        raise ValueError("z_step and z_max must be positive.")

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

    if trace_depth < z_max:
        raise ValueError(
            f"Domain length in probing direction '{probing_direction}' ({trace_depth:.6g} m) "
            f"is smaller than the requested z_max ({z_max:.6g} m). "
            f"Increase the domain size to at least z_max."
        )

    # Uniform depth save positions: [0, z_step, 2*z_step, ..., z_max]
    n_saves = int(np.round(z_max / z_step)) + 1
    depth_saves = np.linspace(0.0, z_max, n_saves)

    if verbose:
        print(f"\ntrace_and_save_depths: saving at {n_saves} depth(s) from 0 to {z_max*1e3:.4g} mm "
              f"(step = {z_step*1e6:.4g} µm, probing direction = '{probing_direction}', "
              f"jones_components = {comp_indices}).")

    # Map depth positions to normalised diffrax time [0, 1].
    # The ODE solver normalises real time by norm_factor = sqrt(8)*trace_depth/c so
    # that t_norm=1 corresponds to the end of the trace.  A ray travelling at ~c
    # covers depth d in real time d/c, giving normalised time d/(sqrt(8)*trace_depth).
    norm_factor = np.sqrt(8.0) * trace_depth / c
    t_saves_norm = depth_saves / (np.sqrt(8.0) * trace_depth)

    Np = s0.shape[1]

    args = (
        True,  # parallelise=True (each vmap'd call handles one ray)
        ScalarDomain.ne,
        ScalarDomain.x, ScalarDomain.y, ScalarDomain.z,
        omega,
        ScalarDomain.lengths, ScalarDomain.dims,
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

    # vmap over rays; each call handles one ray (shape (6,))
    s0_T = s0.T  # shape (N, 6)
    sol = jax.block_until_ready(
        jax.vmap(_ode_solve, in_axes=(0, None))(s0_T, args)
    )

    # sol.ys has shape (N, n_saves, 6).
    # For each depth snapshot: compute the full 4-row Jones vector via
    # ray_to_Jonesvector (keep_current_plane=True records the actual position
    # and angle at that depth, not propagated to the exit plane), then keep
    # only the user-requested rows.
    jvec_list = []

    for j in range(n_saves):
        rays_j = sol.ys[:, j, :].T  # shape (6, N)
        jvec_full, _ = ray_to_Jonesvector(rays_j, keep_current_plane=True,
                                           probing_direction=probing_direction)
        # jvec_full rows: [pos_axis1, angle_axis1, pos_axis2, angle_axis2]
        jvec_list.append(np.asarray(jvec_full)[comp_indices, :])  # (n_comp, N)

    result = {
        'depth_saves': depth_saves,
        'jvec': jvec_list,
        'jones_components': comp_indices,
    }

    if output_path is not None:
        with open(output_path, 'wb') as fh:
            pickle.dump(result, fh)
        if verbose:
            print(f"trace_and_save_depths: results saved to '{output_path}'.")

    return result
