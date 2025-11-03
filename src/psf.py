import numpy as np
from scipy.ndimage import shift
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import zoom
from skimage.transform import resize
import math

import numpy as np
from scipy.ndimage import map_coordinates

def congrid(arr, new_x, new_y=None, new_z=None, center=False, interp=False,
            cubic=0.0, minus_one=False):
    """
    Python equivalent of IDL's CONGRID function.
    Resize 1D, 2D, or 3D arrays with optional linear or cubic interpolation.
    """

    arr = np.asarray(arr)
    ndim = arr.ndim
    if ndim < 1 or ndim > 3:
        raise ValueError("Array must have 1, 2, or 3 dimensions.")

    dims = np.array(arr.shape, dtype=float)

    # Handle keyword behavior
    intp = bool(interp)
    m1 = bool(minus_one)
    cubic = float(cubic)
    if cubic != 0:
        intp = True  # cubic implies interpolation
    offset = 0.5 if center else 0.0

    # Target sizes
    targets = [new_x]
    if ndim > 1:
        if new_y is None:
            raise ValueError("Y dimension required for 2D or 3D arrays.")
        targets.append(new_y)
    if ndim > 2:
        if new_z is None:
            raise ValueError("Z dimension required for 3D arrays.")
        targets.append(new_z)

    targets = np.array(targets, dtype=float)

    # Compute scale factors
    def compute_coords(d_in, d_out):
        return (float(d_in - m1) / (d_out - m1)) * (np.arange(d_out) + offset) - offset

    # Build coordinate grid
    if ndim == 1:
        srx = compute_coords(dims[0], targets[0])
        if intp:
            coords = [srx]
            arr_r = map_coordinates(arr, coords, order=3 if cubic != 0 else 1, mode='nearest')
        else:
            arr_r = arr[np.clip(np.round(srx).astype(int), 0, arr.shape[0]-1)]

    elif ndim == 2:
        srx = compute_coords(dims[0], targets[0])
        sry = compute_coords(dims[1], targets[1])
        if intp:
            xg, yg = np.meshgrid(srx, sry, indexing='ij')
            coords = [xg, yg]
            arr_r = map_coordinates(arr, coords, order=3 if cubic != 0 else 1, mode='nearest')
        else:
            # nearest neighbor (like POLY_2D)
            expand = (targets[0] > dims[0])
            xm1 = (targets[0] - 1) if (m1 or expand) else targets[0]
            ym1 = (targets[1] - 1)
            xi = ((dims[0] - m1) / xm1) * np.arange(targets[0])
            yi = ((dims[1] - m1) / ym1) * np.arange(targets[1])
            xi = np.clip(np.round(xi).astype(int), 0, arr.shape[0]-1)
            yi = np.clip(np.round(yi).astype(int), 0, arr.shape[1]-1)
            arr_r = arr[np.ix_(xi, yi)]

    elif ndim == 3:
        srx = compute_coords(dims[0], targets[0])
        sry = compute_coords(dims[1], targets[1])
        srz = compute_coords(dims[2], targets[2])
        xg, yg, zg = np.meshgrid(srx, sry, srz, indexing='ij')
        coords = [xg, yg, zg]
        arr_r = map_coordinates(arr, coords, order=1, mode='nearest')  # only linear interpolation supported

    return arr_r


class PSF:
    def __init__(self, psf_image, channel=1):
        """
        Initializes the PSF object with a PSF image.

        Parameters
        ----------
        psf_image : np.ndarray
            The 2D image of the Point Spread Function.
        """
        self.psf_image = psf_image
        self.channel = channel
        self.original_shape = psf_image.shape
        self.trimmed_shape = psf_image.shape
        self.current_shape = self.psf_image.shape
        self.scale = self.get_scale(channel)

    def get_scale(self, channel):
        """
        Returns the scale factor based on the channel.
        Parameters
        ----------
        channel : int
            The channel number (1, 2, 3, or 4).
        Returns
        -------
        float
            The scale factor.
        """
        if channel == 1:
            p_prf = 1.221
        elif channel == 2:
            p_prf = 1.213
        elif channel == 3:
            p_prf = 1.222
        elif channel == 4:
            p_prf = 1.220

        p_mosaic = 0.6
        return (p_prf / 100.0) / p_mosaic
    
    def trim_psf(self, trim_pixels):
        """
        Trims the PSF image equally from all sides.

        Parameters
        ----------
        trim_pixels : int
            The number of pixels to trim from each side (top, bottom, left, right).
        """
        if not isinstance(trim_pixels, int) or trim_pixels < 0:
            raise ValueError("trim_pixels must be a non-negative integer.")

        if 2 * trim_pixels >= self.psf_image.shape[0] or 2 * trim_pixels >= self.psf_image.shape[1]:
            raise ValueError("Trim amount is too large for the current PSF image dimensions.")

        self.psf_image = self.psf_image[trim_pixels:-trim_pixels, trim_pixels:-trim_pixels]
        self.trimmed_shape = self.psf_image.shape
    
    def compute_dimensions(self):
        """
        Computes the current dimensions of the PSF image after scaling.
        Returns
        -------
        int
            The current dimensions of the PSF image.
        """
        if self.channel == 1 or self.channel == 3:
            return math.ceil(self.psf_image.shape[0] * self.scale)
        elif self.channel == 2 or self.channel == 4:
            return math.floor(self.psf_image.shape[0] * self.scale)

    def shift_psf(self, frac_x, frac_y):
        """
        Shifts the PSF image by fractional pixel amounts.

        Parameters
        ----------
        frac_x : float
            Fractional shift in the x-direction.
        frac_y : float
            Fractional shift in the y-direction.
        """
        shift_x = int(frac_x / self.scale)
        shift_y = int(frac_y / self.scale)
        self.psf_image = np.roll(self.psf_image, shift_x, axis=1)
        self.psf_image = np.roll(self.psf_image, shift_y, axis=0)

        # prf = self.psf_image
        # prf_shifted = np.zeros_like(prf)

        # ny, nx = prf.shape
        # x1 = max(0, shift_x)
        # y1 = max(0, shift_y)
        # x2 = nx - max(0, -shift_x)
        # y2 = ny - max(0, -shift_y)

        # prf_shifted[y1:y2, x1:x2] = prf[
        #     max(0, -shift_y):ny - max(0, shift_y),
        #     max(0, -shift_x):nx - max(0, shift_x)
        # ]

        # self.psf_image = prf_shifted

    def normalize_psf(self, norm=1.0):
        """
        Normalizes the PSF image so that its sum is 1.

        Parameters
        ----------
        norm : float
            Normalization factor.
        """
        self.psf_image /= np.sum(self.psf_image)
        self.psf_image *= norm

    def congrid(self, new_shape, method='linear'):
        """
        Resample an array to a new shape using interpolation (similar to IDL's CONGRID).

        Parameters
        ----------
        arr : np.ndarray
            Input array.
        new_shape : tuple
            Target shape (ny, nx).
        method : str
            Interpolation method: 'nearest', 'linear', or 'cubic'.
        """
        # if not isinstance(new_shape, tuple) or len(new_shape) != 2:
        #     raise ValueError("new_shape must be a tuple of two integers (height, width).")

        old_rows, old_cols = self.psf_image.shape
        new_rows = new_shape 
        new_cols = new_shape

        # Create an interpolator for the original PSF image
        # row_coords = np.linspace(0, old_rows - 1, old_rows)
        # col_coords = np.linspace(0, old_cols - 1, old_cols)
        # interp_func = RectBivariateSpline(row_coords, col_coords, self.psf_image)

        # # Define new coordinates for the target shape
        # new_row_coords = np.linspace(0, old_rows - 1, new_rows)
        # new_col_coords = np.linspace(0, old_cols - 1, new_cols)

        # # Evaluate the interpolator at the new coordinates
        # self.psf_image = interp_func(new_row_coords, new_col_coords)

        zoom_factors = [n / float(o) for n, o in zip((new_rows, new_cols), self.psf_image.shape)]
        order_map = {'nearest': 0, 'linear': 1, 'cubic': 3}
        order = order_map.get(method, 1)
        self.psf_image = resize(self.psf_image, (new_rows, new_cols), order=order, mode='reflect', anti_aliasing=False)

    def get_psf_image(self):
        """
        Returns the current PSF image.

        Returns
        -------
        np.ndarray
            The 2D image of the Point Spread Function.
        """
        return self.psf_image
    
    def plot_psf_image(self):
        """
        Plots the PSF image using matplotlib.
        """
        import matplotlib.pyplot as plt
        plt.imshow(self.psf_image, cmap='viridis', origin='lower')
        plt.colorbar(label='Intensity')
        plt.title(f'PSF Image (Channel {self.channel})')
        plt.xlabel('X Pixels')
        plt.ylabel('Y Pixels')
        plt.show()
