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

class Diagnostic:
    """
    Inheritable class for ray diagnostics.
    """

    # this is in mm's not metres - self.rf is converted to mm's (not sure if everything else is covered though)
    def __init__(self, rf, *, weights = None, focal_plane = 0, L = 400, R = 25, Lx = 18, Ly = 13.5, prefilter_input = False):
        """
        Initialise ray diagnostic.

        Args:
            r0 (4xN float array): N rays, [x, theta, y, phi]

            L (int, optional): Length scale L. First lens is at L. Defaults to 400.
            R (int, optional): Radius of lenses. Defaults to 25.
            Lx (int, optional): Detector size in x. Defaults to 18.
            Ly (float, optional): Detector size in y. Defaults to 13.5.
        """     

        self.focal_plane, self.L, self.R, self.Lx, self.Ly = focal_plane, L, R, Lx, Ly

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

    def single_lens_solve(self):
        ## single lens - M = Variable (around ~2) (based on Detector position. Real experimental setup)
        r1 = travel(self.r0, 3 * self.L / 4 - self.focal_plane) #displace rays to lens. Accounts for object with depth
        r2 = circular_aperture(r1, self.R)      # cut off
        r3 = sym_lens(r2, self.L / 2)             # lens 1
        r4 = travel(r3, 3*self.L / 2)           # detector
        self.rf = r4

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


def compare_diagnostics(H_plain, H_wt, xedges, yedges, axes=None):
    """
    Three-panel comparison: unweighted count image, weighted intensity image,
    and their pixel-wise ratio (weighted / unweighted).

    Ratio > 1 is clipped to 1.  Bins with zero counts are shown in grey.
    This is the primary tool for checking whether absorption is measurably
    reducing intensity relative to a uniform-illumination baseline.

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

    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.where(H_plain > 0, H_wt / H_plain, np.nan)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    kw = dict(origin='lower', extent=extent, interpolation='nearest', aspect='auto')

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    else:
        fig = axes[0].get_figure()

    ax0, ax1, ax2 = axes

    im0 = ax0.imshow(H_plain, **kw, cmap='inferno')
    ax0.set_title('Unweighted (count)')
    plt.colorbar(im0, ax=ax0)

    im1 = ax1.imshow(H_wt, **kw, cmap='inferno')
    ax1.set_title('Weighted (intensity)')
    plt.colorbar(im1, ax=ax1)

    cmap_r = mpl.colormaps.get_cmap('RdBu_r').copy()
    cmap_r.set_bad(color='lightgrey')
    im2 = ax2.imshow(ratio, **kw, cmap=cmap_r, vmin=0, vmax=1)
    ax2.set_title('Ratio  weighted / unweighted')
    plt.colorbar(im2, ax=ax2)

    for ax in axes:
        ax.set_xlabel('x  (mm)')
        ax.set_ylabel('y  (mm)')

    fig.tight_layout()
    return fig, axes
