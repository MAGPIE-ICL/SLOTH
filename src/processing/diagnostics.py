import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
from sympy import Matrix

from shared.utils import count_nans
from shared.utils import round_to_n
from shared.printing import colour

"""
(rtm_solver)
Ray Transfer Matrix Solver - Modified from Jack Hare's Version
"""

def m_to_mm(r):
    rr = np.copy(r)
    rr[0::2,:]*=1e3

    return rr

def mm_to_m(r):
    rr = np.copy(r)
    rr[0::2,:]*=1e-3

    return rr

def lens(r, f1, f2):
    """
    4x4 matrix for a thin lens, focal lengths f1 and f2 in orthogonal axes
    See: https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
    """

    l1 = np.asarray([[1, 0],
            [-1 / f1, 1]])
    l2 = np.asarray([[1, 0],
            [-1 / f2, 1]])

    L = np.zeros((4, 4))
    L[:2,:2]=l1
    L[2:,2:]=l2

    return np.matmul(L, r)

def sym_lens(r, f):
    """
    Helper function to create an axisymmetryic lens
    """

    return lens(r, f, f)

def travel(r, d):
    """4x4 matrix  matrix for travelling a travel d
    See: https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
    """

    d = np.asarray([[1, d],
                     [0, 1]])

    L = np.zeros((4, 4))

    L[:2,:2]=d
    L[2:,2:]=d

    return np.matmul(L, r)

def circular_aperture(r, R, E = None):
    """
    Rejects rays outside radius R
    """

    filt = r[0, :] ** 2 + r[2, :] ** 2 > R ** 2
    r[:,filt]=None
    
    return r

def circular_stop(r, R):
    """
    Rejects rays inside a radius R
    """

    filt = r[0, :] ** 2 + r[2,:] ** 2 < R ** 2
    r[:,filt]=None

    return r

def annular_stop(r, R1, R2):
    """
    Rejects rays which fall between R1 and R2
    """

    filt1 = (r[0,:]**2+r[2,:]**2 > R1**2)
    filt2 = (r[0,:]**2+r[2,:]**2 < R2**2)
    filt = (filt1 & filt2)

    return filt

def rect_aperture(r, Lx, Ly):
    """
    Rejects rays outside a rectangular aperture, total size 2*Lx x 2*Ly
    """

    filt1 = (r[0, :] ** 2 > Lx ** 2)
    filt2 = (r[2, :] ** 2 > Ly ** 2)

    filt=filt1*filt2
    r[:,filt]=None

    return r

def knife_edge(r, offset, axis, direction):
    """
    Filters rays using a knife edge.
    Default is a knife edge in y, can also do a knife edge in x.
    """

    if axis == 'y':
        a=2
    if axis == 'x':
        a=0
    if direction > 0:
        filt = r[a,:] > offset
    if direction < 0:
        filt = r[a,:] < offset
    if direction == 0:
        print('Direction must be <0 or >0')
    r[:,filt]=None
    return r

def clear_rays(self):
    """
    Clears the r0, rf variables to save memory
    """
    # does this actually save memory in the best way?
    # would it be better to del self.r_ instead?

    self.r0 = None
    self.rf = None

def ray(x, θ, y, ϕ):
    """
    Returns a 4x1 matrix representing a ray. Spatial units must be consistent, angular units in radians.
    """

    return Matrix([x, θ, y, ϕ])

def d2r(d):
    # helper function, degrees to radians
    return d * np.pi / 180

def lens_cutoff(rf, *, L = 400, R = 25):
    """
    Masks the Jonesvector resulting array to avoid plotting any values outside of some set limit
    - important as even if you set limits for the histogram to "zoom in", binning is based on raw data
    --> leading to low resolutions if this is not used!

    Args:
        rf (np.Array): Jonesvector output from solver
        L (int): Length till next lens
        R (int): Radius of lens

    Return:
        rf (jax.Array): Masked Jonesvector
    """
    mask = np.pow(np.pow(L * np.tan(rf[1]) + rf[0], 2) + np.pow(L * np.tan(rf[3]) + rf[2], 2), 0.5) <= R

    rf = np.asarray(rf)[:, mask]
    
    return rf


# ---------------------------------------------------------------------------
# Ray-transfer-matrix (RTM) helpers
# ---------------------------------------------------------------------------
# The angular acceptance of a CCD is not a generic global constant — it must
# be derived from the specific optical arrangement of each diagnostic.
#
# The 2×2 RTM [A, B; C, D] maps object-plane coordinates to detector-plane
# coordinates:
#
#   x_det = A · x_obj + B · θ_obj
#
# For an *imaging* axis  (B ≈ 0): position maps to position with
#   magnification M = A → spatial acceptance  x_max = CCD_half / |A|
# For an *angular* axis  (A ≈ 0): angle maps to position with scale B →
#   angular acceptance  θ_max = CCD_half / |B|
# For a *mixed* axis     (A, B ≠ 0): the CCD limits a linear combination;
#   the angular bound at x_obj = 0 is CCD_half / |B| (upper bound).
#
# Different diagnostics (shadowgraphy vs refractometry) have different RTMs
# and therefore different angular acceptances even for the same CCD.
# ---------------------------------------------------------------------------

def _rtm_2x2_travel(d):
    """2×2 free-space propagation matrix for distance *d* (mm)."""
    return np.array([[1.0, float(d)], [0.0, 1.0]])


def _rtm_2x2_lens(f):
    """2×2 thin-lens matrix for focal length *f* (mm)."""
    return np.array([[1.0, 0.0], [-1.0 / float(f), 1.0]])


def compute_arrangement_rtm(ops_x, ops_y=None):
    """
    Compute 2×2 RTMs from an ordered sequence of optical operations.

    Each operation is a ``(type, param)`` tuple where *type* is ``'travel'``
    or ``'lens'`` and *param* is the distance or focal length in mm.

    Args:
        ops_x: operation list for the x-axis (rows 0-1 of the Jones vector).
        ops_y: operation list for the y-axis (rows 2-3).  Defaults to *ops_x*
               (symmetric optics).

    Returns:
        (rtm_x, rtm_y) — each a 2×2 numpy array.
    """
    if ops_y is None:
        ops_y = ops_x

    def _build(ops):
        M = np.eye(2)
        for kind, param in ops:
            if kind == 'travel':
                M = _rtm_2x2_travel(param) @ M
            elif kind == 'lens':
                M = _rtm_2x2_lens(param) @ M
            else:
                raise ValueError(f"Unknown RTM operation: {kind!r}")
        return M

    return _build(ops_x), _build(ops_y)


def compute_ccd_bounds_spatial(ccd_size_mm, rtm_x, rtm_y, *, tol=1e-12):
    """
    Object-plane *spatial* acceptance implied by the CCD extent.

    Returns ``(x_max_mm, y_max_mm)`` — the largest object-plane position
    that maps onto the CCD.  ``None`` if the axis does not carry spatial
    information (|A| < *tol*).
    """
    half_x = ccd_size_mm[0] / 2.0
    half_y = ccd_size_mm[1] / 2.0
    A_x = rtm_x[0, 0]
    A_y = rtm_y[0, 0]
    x_max = half_x / abs(A_x) if abs(A_x) > tol else None
    y_max = half_y / abs(A_y) if abs(A_y) > tol else None
    return x_max, y_max


def compute_ccd_bounds_angular(ccd_size_mm, rtm_x, rtm_y, *, tol=1e-12):
    """
    Object-plane *angular* acceptance implied by the CCD extent.

    The angular acceptance for a given axis is ``CCD_half / |B|`` where *B*
    is the (0, 1) element of the 2×2 RTM — the mapping from input angle to
    detector position.

    Returns ``(theta_max_rad, phi_max_rad)``.  ``None`` if the axis does not
    carry angular information (|B| < *tol*).
    """
    half_x = ccd_size_mm[0] / 2.0
    half_y = ccd_size_mm[1] / 2.0
    B_x = rtm_x[0, 1]
    B_y = rtm_y[0, 1]
    theta_max = half_x / abs(B_x) if abs(B_x) > tol else None
    phi_max   = half_y / abs(B_y) if abs(B_y) > tol else None
    return theta_max, phi_max


def build_ccd_acceptance_mask(rf_mm, *, ccd_size_mm, rtm_x=None, rtm_y=None,
                              verbose=False):
    """
    Build a boolean CCD acceptance mask for detector-plane rays.

    The mask clips on detector *positions* — the fundamental CCD constraint.
    When the RTMs are provided the derived spatial and angular acceptance
    bounds (in the object plane) are computed and returned in the *info*
    dict for inspection.

    Args:
        rf_mm:       (4, N) Jones vector at the detector, in mm / rad.
        ccd_size_mm: ``(Lx_mm, Ly_mm)`` full physical CCD dimensions.
        rtm_x:       2×2 RTM for the x-axis (optional).
        rtm_y:       2×2 RTM for the y-axis (optional).
        verbose:     print a compact acceptance summary.

    Returns:
        mask  — boolean array, shape ``(N,)``.
        info  — dict with keys ``ccd_size_mm``, ``x_half_mm``, ``y_half_mm``,
                ``n_accepted``, ``n_rejected``, ``n_total``, and (when RTMs
                are supplied) ``x_max_mm``, ``y_max_mm``, ``theta_max_rad``,
                ``phi_max_rad``.
    """
    half_x = ccd_size_mm[0] / 2.0
    half_y = ccd_size_mm[1] / 2.0

    mask = np.isfinite(rf_mm[0]) & np.isfinite(rf_mm[2])
    mask &= (np.abs(rf_mm[0]) <= half_x)
    mask &= (np.abs(rf_mm[2]) <= half_y)

    info = {
        'ccd_size_mm': ccd_size_mm,
        'x_half_mm':   half_x,
        'y_half_mm':   half_y,
    }

    if rtm_x is not None and rtm_y is not None:
        x_max, y_max       = compute_ccd_bounds_spatial(ccd_size_mm, rtm_x, rtm_y)
        theta_max, phi_max = compute_ccd_bounds_angular(ccd_size_mm, rtm_x, rtm_y)
        info.update({
            'x_max_mm':       x_max,
            'y_max_mm':       y_max,
            'theta_max_rad':  theta_max,
            'phi_max_rad':    phi_max,
            'rtm_x':          rtm_x.copy(),
            'rtm_y':          rtm_y.copy(),
        })

    info['n_accepted'] = int(mask.sum())
    info['n_rejected'] = int((~mask).sum())
    info['n_total']    = len(mask)

    if verbose:
        print(f"CCD acceptance: {info['n_accepted']}/{info['n_total']} rays "
              f"accepted, {info['n_rejected']} rejected")
        print(f"  CCD size: {ccd_size_mm[0]:.2f} x {ccd_size_mm[1]:.2f} mm")
        print(f"  Spatial bounds: x in [-{half_x:.2f}, {half_x:.2f}] mm, "
              f"y in [-{half_y:.2f}, {half_y:.2f}] mm")
        if rtm_x is not None:
            if info.get('x_max_mm') is not None:
                print(f"  Object x acceptance: +/-{info['x_max_mm']:.3f} mm")
            if info.get('y_max_mm') is not None:
                print(f"  Object y acceptance: +/-{info['y_max_mm']:.3f} mm")
            if info.get('theta_max_rad') is not None:
                print(f"  Derived theta acceptance: "
                      f"+/-{info['theta_max_rad']*1e3:.3f} mrad")
            else:
                print("  Theta: unconstrained by CCD (imaging axis, B_x ~ 0)")
            if info.get('phi_max_rad') is not None:
                print(f"  Derived phi acceptance: "
                      f"+/-{info['phi_max_rad']*1e3:.3f} mrad")
            else:
                print("  Phi: unconstrained by CCD (imaging axis, B_y ~ 0)")

    return mask, info


class Diagnostic:
    """
    Inheritable class for ray diagnostics.
    """

    # this is in mm's not metres - self.rf is converted to mm's (not sure if everything else is covered though)
    def __init__(self, rf, *, weights = None, focal_plane = 0, L = 400, R = 25, Lx = 18, Ly = 13.5, prefilter_input = False, ccd_shape_m = None):
        """
        Initialise ray diagnostic.

        Args:
            r0 (4xN float array): N rays, [x, theta, y, phi]

            L (int, optional): Length scale L. First lens is at L. Defaults to 400.
            R (int, optional): Radius of lenses. Defaults to 25.
            Lx (int, optional): Detector size in x. Defaults to 18.
            Ly (float, optional): Detector size in y. Defaults to 13.5.
            ccd_shape_m (tuple of 2 floats, optional): Physical CCD size
                ``(Lx, Ly)`` in metres, e.g. ``(18e-3, 13.5e-3)``.
                When provided, a rectangular acceptance mask is applied in
                :meth:`histogram` after the RTM solve.  If the diagnostic
                subclass stores RTMs (via its solve method), angular
                acceptance bounds are derived automatically from the
                optical arrangement.
        """     

        self.focal_plane, self.L, self.R, self.Lx, self.Ly = focal_plane, L, R, Lx, Ly
        # CCD footprint clipping — applied after the RTM solve in histogram().
        # ccd_shape_m is (Lx, Ly) in metres; stored as full-size in mm AND
        # as half-extents for backward compatibility.
        if ccd_shape_m is not None:
            self._ccd_size_mm = (ccd_shape_m[0] * 1e3, ccd_shape_m[1] * 1e3)
            self._ccd_half_x_mm = ccd_shape_m[0] * 1e3 / 2.0
            self._ccd_half_y_mm = ccd_shape_m[1] * 1e3 / 2.0
        else:
            self._ccd_size_mm = None
            self._ccd_half_x_mm = None
            self._ccd_half_y_mm = None

        # RTMs set by subclass solve methods; None until a solve is called.
        self._rtm_x = None
        self._rtm_y = None

        if rf is not None:
            assert rf.shape[0] == 4, colour.BOLD + "\nIncorrect format for rf, are you sure you passed the right variable?" + colour.END
            # forces self.rf to the last slice if rf returns multiple samples
            # also preserves the whole pass if required
            if len(rf.shape) == 3:
                # rf might be 3-dimensional if it is a series of 2D ray solution slices
                rf = rf[-1, :, :]

            self.Np = rf.shape[-1]

            self.rf = np.asarray(rf)

            self.Np_inc = self.Np
            
            if prefilter_input:
                # Compute the lens-cutoff mask explicitly so we can apply it
                # consistently to both the geometry array and the weight array.
                _xp = self.L * np.tan(self.rf[1]) + self.rf[0]
                _yp = self.L * np.tan(self.rf[3]) + self.rf[2]
                _prefilter_mask = np.sqrt(_xp ** 2 + _yp ** 2) <= self.R
                self.rf = self.rf[:, _prefilter_mask]
                if weights is not None:
                    weights = np.asarray(weights)[_prefilter_mask]
                self.Np_inc = self.rf.shape[-1]

            if self.Np == self.Np_inc:
                print("\nAll rays retained at diagnostic input.")
            else:
                print("\n{} rays received, {} retained after optional prefilter.".format(str(self.Np), str(self.Np_inc)))
                print(" --> {} % filtered before diagnostic solve.".format(str(round_to_n((1 - self.Np_inc / self.Np) * 100, 3))))
        else:
            assert "rf should not be of Noneype! diffrax clearly failed."

        self.r0 = m_to_mm(self.rf)
        # Per-ray amplitude weights (None = uniform count histogram).
        # Stored as a float64 array so np.histogram2d accepts it directly.
        self.weights = np.asarray(weights, dtype=np.float64) if weights is not None else None

    def histogram(self, *, bin_scale = 1, pix_x = 3448, pix_y = 2574, clear_mem = False, plain_plot = False, extra_info = True):
        """
        Bin data into a histogram. Defaults are for a KAF-8300.
        Outputs are H, the histogram, and xedges and yedges, the bin edges.

        Args:
            bin_scale (int, optional): bin size, same in x and y. Defaults to 1.
            pix_x (int, optional): number of x pixels in detector plane. Defaults to 3448.
            pix_y (int, optional): number of y pixels in detector plane. Defaults to 2574.
        """

        matrix = self.r0 if plain_plot else self.rf
        mask = ~np.isnan(matrix[0]) & ~np.isnan(matrix[2])

        # CCD rectangular clipping (after RTM solve, before binning).
        # Coordinates at this stage are in mm (self.rf is set by the solve method).
        if self._ccd_size_mm is not None:
            ccd_mask, self._ccd_info = build_ccd_acceptance_mask(
                matrix,
                ccd_size_mm=self._ccd_size_mm,
                rtm_x=self._rtm_x,
                rtm_y=self._rtm_y,
            )
            mask = mask & ccd_mask
        else:
            self._ccd_info = None

        x, y = matrix[0, mask], matrix[2, mask]

        weights = self.weights[mask] if self.weights is not None else None

        self.H, self.xedges, self.yedges = np.histogram2d(x, y, bins=[np.floor(pix_x / bin_scale).astype(np.int64), np.floor(pix_y / bin_scale).astype(np.int64)], range=[[-self.Lx / 2, self.Lx / 2],[-self.Ly / 2, self.Ly / 2]], weights=weights)

        self.H = self.H.T

        #Optional - clear ray attributes to save memory
        if clear_mem:
            clear_rays(self)

    def plot(self, ax, clim = None, cmap = None):
        ax.imshow(self.H, interpolation='nearest', origin='lower', clim=clim, cmap=cmap, extent = [self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]])

    def plot_rays(self, *, bin_scale = 1, pix_x = 3448, pix_y = 2574, clear_mem = False):
        self.histogram(bin_scale = bin_scale, pix_x = pix_x, pix_y = pix_y, clear_mem = clear_mem, plain_plot = True)

    def ccd_acceptance_info(self, *, verbose=False):
        """
        Return (and optionally print) a dict describing the CCD acceptance.

        The dict contains detector-plane spatial bounds and, when the
        diagnostic has stored its RTMs (via a solve method), the derived
        object-plane spatial and angular acceptance bounds.

        Call *after* :meth:`histogram` so that ray counts are available.
        """
        if self._ccd_size_mm is None:
            info = {'ccd_enabled': False}
            if verbose:
                print("CCD acceptance: disabled (ccd_shape_m=None)")
            return info

        info = dict(self._ccd_info) if self._ccd_info is not None else {}
        info['ccd_enabled'] = True

        if verbose:
            build_ccd_acceptance_mask(
                self.rf,
                ccd_size_mm=self._ccd_size_mm,
                rtm_x=self._rtm_x,
                rtm_y=self._rtm_y,
                verbose=True,
            )
        return info

class Shadowgraphy(Diagnostic):
    """
    Example shadowgraphy diagnostic. Inherits from Rays, has custom solve method.
    Implements a two lens telescope with M = 1 and a single lens system with M = 2. Both lenses have a f = L/2 focal length, where L is a length scale specified when the class is initialized.
    Each optic has a radius R, which is used to reject rays outside the numerical aperture of the optical system.
    """
    def single_lens_custom_solve(self, f = 200, obj_dist = 400, img_dist = 400):
        ## single lens - M = Variable (around ~2) (based on Detector position. Real experimental setup)
        r1 = travel(self.r0, obj_dist - self.focal_plane) #displace rays to lens. Accounts for object with depth
        r2 = circular_aperture(r1, self.R)      # cut off
        r3 = sym_lens(r2, f)             # lens 1
        r4 = travel(r3, img_dist)           # detector
        self.rf = r4

        # RTM (symmetric in x/y for a symmetric lens)
        ops = [('travel', obj_dist - self.focal_plane),
               ('lens', f), ('travel', img_dist)]
        self._rtm_x, self._rtm_y = compute_arrangement_rtm(ops)

    def single_lens_solve(self):
        ## single lens - M = Variable (around ~2) (based on Detector position. Real experimental setup)
        r1 = travel(self.r0, 3 * self.L / 4 - self.focal_plane) #displace rays to lens. Accounts for object with depth
        r2 = circular_aperture(r1, self.R)      # cut off
        r3 = sym_lens(r2, self.L / 2)             # lens 1
        r4 = travel(r3, 3*self.L / 2)           # detector
        self.rf = r4

        ops = [('travel', 3 * self.L / 4 - self.focal_plane),
               ('lens', self.L / 2), ('travel', 3 * self.L / 2)]
        self._rtm_x, self._rtm_y = compute_arrangement_rtm(ops)

    def two_lens_solve(self):
        ## 2 lens telescope, M = 1
        r1 = travel(self.r0, self.L - self.focal_plane) #displace rays to lens. Accounts for object with depth
        r2 = circular_aperture(r1, self.R)    # cut off
        r3 = sym_lens(r2, self.L / 2)           # lens 1
        r4 = travel(r3, self.L * 2)           # displace rays to lens 2.
        r5 = circular_aperture(r4, self.R)    # cut off
        r6 = sym_lens(r5, self.L / 2)           # lens 2
        r7 = travel(r6, self.L)             # displace rays to detector
        self.rf = r7

        ops = [('travel', self.L - self.focal_plane),
               ('lens', self.L / 2), ('travel', 2 * self.L),
               ('lens', self.L / 2), ('travel', self.L)]
        self._rtm_x, self._rtm_y = compute_arrangement_rtm(ops)
    
class Schlieren(Diagnostic):
    """
    Example dark field schlieren diagnostic. Inherits from Rays, has custom solve method.
    Implements a two lens telescope with M = 1. Both lenses have a f = L focal length, where L is a length scale specified when the class is initialized.
    Each optic has a radius R, which is used to reject rays outside the numerical aperture of the optical system.
    There is a circular stop placed at the focal point after the first lens which rejects rays which hit the focal planes at travel less than R [mm] from the optical axis.
    """

    def DF_solve(self, R = 1):
        ## 2 lens telescope, M = 1
        r1 = travel(self.r0, self.L - self.focal_plane) #displace rays to lens. Accounts for object with depth
        r2 = circular_aperture(r1, self.R) # cut off
    
        r3 = sym_lens(r2, self.L) #lens 1

        r4 = travel(r3, self.L) #displace rays to stop

        # this and positioning of lenses means schlieren ends up with less usable rays than other methods
        r5 = circular_stop(r4, R = R) # stop - blocker at focal point after the first lens of size R (1 mm?)

        r6 = travel(r5, self.L) #displace rays to lens 2

        r7 = circular_aperture(r6, self.R) # cut off

        r8 = sym_lens(r7, self.L) #lens 2

        r9 = travel(r8, self.L) #displace rays to detector

        self.rf = r9

        # RTM ignores non-linear stops/apertures (symmetric in x/y).
        ops = [('travel', self.L - self.focal_plane),
               ('lens', self.L), ('travel', 2 * self.L),
               ('lens', self.L), ('travel', self.L)]
        self._rtm_x, self._rtm_y = compute_arrangement_rtm(ops)
    
    """
    Example light field schlieren diagnostic. Inherits from Rays, has custom solve method.
    Implements a two lens telescope with M = 1. Both lenses have a f = L/2 focal length, where L is a length scale specified when the class is initialized.
    Each optic has a radius R, which is used to reject rays outside the numerical aperture of the optical system.
    There is a circular stop placed at the focal point afte rthe first lens which accepts only rays which hit the focal planes at travel less than R [mm] from the optical axis.
    """

    def LF_solve(self, R = 1):
        ## 2 lens telescope, M = 1
        r1 = travel(self.r0, self.L - self.focal_plane) #displace rays to lens. Accounts for object with depth
        r2 = circular_aperture(r1, self.R) # cut off
        r3 = sym_lens(r2, self.L) #lens 1

        r4 = travel(r3, self.L) #displace rays to stop
        r5 = circular_aperture(r4, R = R) # stop

        r6 = travel(r5, self.L) #displace rays to lens 2
        r7 = circular_aperture(r6, self.R) # cut off
        r8 = sym_lens(r7, self.L) #lens 2

        r9 = travel(r8, self.L) #displace rays to detector
        self.rf = r9

        # RTM ignores non-linear stops/apertures (symmetric in x/y).
        ops = [('travel', self.L - self.focal_plane),
               ('lens', self.L), ('travel', 2 * self.L),
               ('lens', self.L), ('travel', self.L)]
        self._rtm_x, self._rtm_y = compute_arrangement_rtm(ops)
        
class Refractometry(Diagnostic):
    """
    Example of Imaging Refractometer. Inherits from Rays, has custom solve method.
    Implements a spherical lens with focal length f1 = L/2 and M = 2 for the spatial axis and a cylindrical lens
    with focal length f1 and f2.
    """
    def incoherent_custom_solve(self, f1 = 200, f3 = 200, img_f1_dist = 600, img_dist = 400):
        ##
        ## Is there an efficient way to chain these so needlessly variables are not used without having 1 really long line
        ##

        ## Imaging the spatial axis
        r1 = travel(self.r0, 2*f1 - self.focal_plane) #displace rays to lens 1. Accounts for object with depth
        r2 = circular_aperture(r1, self.R)   # cut off
        r3 = sym_lens(r2, f1)                # lens 1 - spherical
        r4 = travel(r3, img_f1_dist)         # displace rays to lens 2 - hybrid
        r5 = rect_aperture(r4, 15, 30)       # rectangular lens cut-off
        r6 = circular_aperture(r5, self.R)   # cut off
        r7 = lens(r6, (2*f3)/3, f1)          # lens 2 - hybrid lens
        r8 = travel(r7, img_dist)               # displace rays to detector
        self.rf = r8

        # RTM: x and y axes have different focal lengths for the hybrid lens.
        d1 = 2 * f1 - self.focal_plane
        ops_x = [('travel', d1), ('lens', f1), ('travel', img_f1_dist),
                 ('lens', (2 * f3) / 3), ('travel', img_dist)]
        ops_y = [('travel', d1), ('lens', f1), ('travel', img_f1_dist),
                 ('lens', f1), ('travel', img_dist)]
        self._rtm_x, self._rtm_y = compute_arrangement_rtm(ops_x, ops_y)

    def incoherent_solve(self):
        ##
        ## Is there an efficient way to chain these so needlessly variables are not used without having 1 really long line
        ##

        ## Imaging the spatial axis - M = 2
        r1 = travel(self.r0, 3 * self.L / 4 - self.focal_plane) #displace rays to lens 1. Accounts for object with depth
        r2 = circular_aperture(r1, self.R)      # cut off
        r3 = sym_lens(r2, self.L/2)             # lens 1 - spherical
        r4 = travel(r3, 3*self.L/2)           # displace rays to lens 2 - hybrid
        r5 = rect_aperture(r4, 15, 30)          # rectangular lens cut-off
        r6 = circular_aperture(r5, self.R)      # cut off
        r7 = lens(r6, self.L/3, self.L/2)       # lens 2 - hybrid lens
        r8 = travel(r7, self.L)               # displace rays to detector
        self.rf = r8

        # RTM: x and y axes have different focal lengths for the hybrid lens.
        d1 = 3 * self.L / 4 - self.focal_plane
        ops_x = [('travel', d1), ('lens', self.L / 2),
                 ('travel', 3 * self.L / 2),
                 ('lens', self.L / 3), ('travel', self.L)]
        ops_y = [('travel', d1), ('lens', self.L / 2),
                 ('travel', 3 * self.L / 2),
                 ('lens', self.L / 2), ('travel', self.L)]
        self._rtm_x, self._rtm_y = compute_arrangement_rtm(ops_x, ops_y)

# ---------------------------------------------------------------------------
# Amplitude inspection helpers
# ---------------------------------------------------------------------------

def plot_amplitude_diagnostics(amp, jvec, axes=None):
    """
    Two-panel amplitude debug view.

    Panel 1: histogram / PDF of amplitude values — shows whether attenuation
             is uniform or has a spread, and how far values are from 1.
    Panel 2: amplitude vs x-position (jvec row 0) — shows where in the beam
             cross-section losses are largest.

    Args:
        amp  (array-like, shape (N,)): per-ray amplitude weights (exp(-tau)).
        jvec (array-like, shape (4, N)): geometric Jones vector [x, theta, y, phi].
        axes (sequence of 2 Axes, optional): existing axes; created if None.

    Returns:
        fig, axes
    """
    amp  = np.asarray(amp, dtype=np.float64)
    jvec = np.asarray(jvec)

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig = axes[0].get_figure()

    ax0, ax1 = axes

    # Panel 1: PDF of amplitude
    ax0.hist(amp, bins=40, density=True, color='steelblue', edgecolor='none')
    ax0.axvline(amp.mean(), color='firebrick', linestyle='--',
                label=f'mean = {amp.mean():.4f}')
    ax0.set_xlabel('amplitude  exp(−τ)')
    ax0.set_ylabel('density')
    ax0.set_title('Amplitude distribution')
    ax0.legend(fontsize=8)

    # Panel 2: amplitude vs x position
    x = jvec[0]
    ax1.scatter(x * 1e3, amp, s=6, alpha=0.6, color='steelblue')
    ax1.axhline(1.0, color='grey', linestyle=':', linewidth=0.8)
    ax1.set_xlabel('x  (mm)')
    ax1.set_ylabel('amplitude  exp(−τ)')
    ax1.set_title('Amplitude vs x position')

    fig.tight_layout()
    return fig, axes


def absorption_sanity_check(ne, Te, Z, lwl, depth, *, axes=None):
    """
    Compact diagnostic: expected inverse-brems optical depth for given plasma parameters.

    Computes kappa, optical depth tau, and amplitude exp(-tau) for a uniform
    slab, and produces a 2-panel summary plot:
      Panel 1: tau vs density for several temperatures.
      Panel 2: Coulomb log vs temperature.

    Args:
        ne (float): Electron density in m^-3.
        Te (float): Electron temperature in eV.
        Z  (float): Mean ion charge state.
        lwl (float): Laser wavelength in metres.
        depth (float): Path length in metres.
        axes (sequence of 2 Axes, optional): existing axes; created if None.

    Returns:
        info (dict): Keys ``kappa``, ``tau``, ``amplitude``, ``coulomb_log``.
        fig, axes
    """
    from scipy.constants import c as _c, e as _e, epsilon_0 as _eps0

    omega = 2 * np.pi * _c / lwl
    ne_cc = ne * 1e-6
    v_the = 4.19e5 * np.sqrt(Te)
    o_pe  = 5.64e4 * np.sqrt(ne_cc)
    o_max = max(o_pe, omega)
    b_min = Z * _e / (4.0 * np.pi * _eps0 * Te)
    CL_arg = v_the / (o_max * b_min)
    CL    = max(2.0, np.log(CL_arg))
    kappa = 3.1e-5 * Z * _c * (ne_cc / omega) ** 2 * CL * Te ** (-1.5)
    tau   = kappa * depth / _c
    amp   = np.exp(-tau)

    info = dict(kappa=kappa, tau=tau, amplitude=amp, coulomb_log=CL,
                coulomb_log_arg=CL_arg, b_min=b_min)

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig = axes[0].get_figure()

    ax0, ax1 = axes

    # Panel 1: tau vs density for several Te values
    ne_range = np.logspace(22, 27, 200)
    for Te_v in [10, 50, 100, 500]:
        ne_cc_v = ne_range * 1e-6
        v_v = 4.19e5 * np.sqrt(Te_v)
        o_pe_v = 5.64e4 * np.sqrt(ne_cc_v)
        o_max_v = np.maximum(o_pe_v, omega)
        b_min_v = Z * _e / (4.0 * np.pi * _eps0 * Te_v)
        CL_v = np.maximum(2.0, np.log(v_v / (o_max_v * b_min_v)))
        kappa_v = 3.1e-5 * Z * _c * (ne_cc_v / omega) ** 2 * CL_v * Te_v ** (-1.5)
        tau_v = kappa_v * depth / _c
        ax0.loglog(ne_range, tau_v, label=f'Te={Te_v} eV')
    ax0.axhline(1.0, color='grey', ls=':', lw=0.8, label=r'$\tau=1$')
    ax0.axvline(ne, color='firebrick', ls='--', lw=0.8, label=f'ne={ne:.1e}')
    ax0.set_xlabel(r'$n_e$  (m$^{-3}$)')
    ax0.set_ylabel(r'$\tau$')
    ax0.set_title(f'Optical depth ({lwl*1e9:.0f} nm, {depth*1e3:.1f} mm, Z={Z})')
    ax0.legend(fontsize=7)

    # Panel 2: Coulomb log vs Te
    Te_range = np.logspace(0, 3, 200)
    v_t = 4.19e5 * np.sqrt(Te_range)
    b_min_t = Z * _e / (4.0 * np.pi * _eps0 * Te_range)
    CL_t = np.maximum(2.0, np.log(v_t / (o_max * b_min_t)))
    ax1.semilogx(Te_range, CL_t, color='steelblue')
    ax1.axhline(2.0, color='grey', ls=':', lw=0.8, label='CL floor = 2')
    ax1.axvline(Te, color='firebrick', ls='--', lw=0.8, label=f'Te={Te} eV')
    ax1.set_xlabel('Te  (eV)')
    ax1.set_ylabel('Coulomb logarithm')
    ax1.set_title('Coulomb log vs temperature')
    ax1.legend(fontsize=7)

    fig.tight_layout()
    return info, fig, axes


def transmission_map(H_wt, H_plain):
    """
    Per-pixel mean transmission: ``T(i,j) = H_wt(i,j) / H_plain(i,j)``.

    This isolates the absorption effect from the ray-density (refraction) effect.
    Where many rays land, :attr:`H_wt` is larger but so is :attr:`H_plain`; the
    ratio cancels the density variation and leaves only the mean local amplitude.

    For uniform weights ``w``, every occupied pixel returns exactly ``w``.
    For spatially varying absorption the map shows where losses are strongest.
    Empty pixels (zero ray count) are set to ``nan``.

    Args:
        H_wt    (2-D array): amplitude-weighted histogram.
        H_plain (2-D array): unweighted ray-count histogram (same shape).

    Returns:
        T (2-D float64 array): per-pixel mean transmission in [0, 1]; nan where
        H_plain == 0.
    """
    H_wt    = np.asarray(H_wt,    dtype=np.float64)
    H_plain = np.asarray(H_plain, dtype=np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(H_plain > 0, H_wt / H_plain, np.nan)


def apply_ccd_mask(rf, weights=None, ccd_shape_m=(13.5e-3, 18e-3)):
    """
    Apply a rectangular CCD mask in diagnostic-space coordinates.

    Rays whose positions fall outside the CCD footprint are set to NaN
    (geometry) and zero (weights), so they are excluded from downstream
    histograms.

    Args:
        rf (ndarray, shape (4, N)): Jones vector in metres.
            Row 0 is the first transverse position (x), row 2 is the second (y).
        weights (ndarray or None): Per-ray amplitude weights, shape (N,).
        ccd_shape_m (tuple of 2 floats): Physical CCD full size ``(Lx, Ly)``
            in metres.  Default is ``(13.5e-3, 18e-3)`` (13.5 mm × 18 mm).
            Rays outside ``±Lx/2`` in x or ``±Ly/2`` in y are masked.

    Returns:
        rf_out (ndarray, (4, N)): Masked Jones vector (NaN outside CCD).
        weights_out (ndarray or None): Masked weights (0 outside CCD).
        mask (bool ndarray, (N,)): True for rays inside the CCD footprint.
    """
    rf = np.array(rf, dtype=np.float64, copy=True)
    half_x = ccd_shape_m[0] / 2.0
    half_y = ccd_shape_m[1] / 2.0

    inside = (
        (np.abs(rf[0]) <= half_x) &
        (np.abs(rf[2]) <= half_y)
    )
    rf[:, ~inside] = np.nan

    if weights is not None:
        weights = np.array(weights, dtype=np.float64, copy=True)
        weights[~inside] = 0.0

    return rf, weights, inside


def compare_diagnostics(H_plain, H_wt, xedges, yedges, axes=None):
    """
    Three-panel comparison: unweighted count image, weighted intensity image,
    and the per-pixel transmission map (weighted / unweighted).

    The unweighted and weighted panels share the same color scale so that a
    uniform 24 % attenuation is immediately visible as a darker weighted image
    rather than being hidden by independent auto-normalization.

    The transmission panel uses :func:`transmission_map` and has a fixed
    scale of [0, 1].  Bins with zero counts are shown in grey.

    Args:
        H_plain  (2-D array): unweighted histogram (count image).
        H_wt     (2-D array): amplitude-weighted histogram.
        xedges   (1-D array): x bin edges from np.histogram2d.
        yedges   (1-D array): y bin edges from np.histogram2d.
        axes     (sequence of 3 Axes, optional): created if None.

    Returns:
        fig, axes
    """
    H_plain = np.asarray(H_plain, dtype=np.float64)
    H_wt    = np.asarray(H_wt,    dtype=np.float64)

    T = transmission_map(H_wt, H_plain)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    kw = dict(origin='lower', extent=extent, interpolation='nearest', aspect='auto')

    # Shared scale for panels 0 and 1 so attenuation magnitude is visible.
    shared_vmax = max(H_plain.max(), H_wt.max())

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    else:
        fig = axes[0].get_figure()

    ax0, ax1, ax2 = axes

    im0 = ax0.imshow(H_plain, **kw, cmap='inferno', vmin=0, vmax=shared_vmax)
    ax0.set_title('Unweighted (count)')
    plt.colorbar(im0, ax=ax0)

    im1 = ax1.imshow(H_wt, **kw, cmap='inferno', vmin=0, vmax=shared_vmax)
    ax1.set_title('Weighted (intensity)')
    plt.colorbar(im1, ax=ax1)

    cmap_r = mpl.colormaps.get_cmap('RdBu_r').copy()
    cmap_r.set_bad(color='lightgrey')
    im2 = ax2.imshow(T, **kw, cmap=cmap_r, vmin=0, vmax=1)
    ax2.set_title('Transmission  (weighted / count)')
    plt.colorbar(im2, ax=ax2)

    for ax in axes:
        ax.set_xlabel('x  (mm)')
        ax.set_ylabel('y  (mm)')

    fig.tight_layout()
    return fig, axes
