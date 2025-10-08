import numpy as np
from scipy.ndimage import shift
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import zoom
import math

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
    
    def trim_prf(self, trim_pixels):
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
            return math.ceil(self.current_shape[0] * self.scale)
        elif self.channel == 2 or self.channel == 4:
            return math.floor(self.current_shape[0] * self.scale)

    def shift_prf(self, frac_x, frac_y):
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

    def normalize_prf(self, norm=1.0):
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

        # zoom_factors = [n / float(o) for n, o in zip((new_rows, new_cols), self.psf_image.shape)]
        # order_map = {'nearest': 0, 'linear': 1, 'cubic': 3}
        # order = order_map.get(method, 1)
        # self.psf_image = zoom(self.psf_image, zoom_factors, order=order)

        # Create an interpolator for the original PSF image
        row_coords = np.linspace(0, old_rows - 1, old_rows)
        col_coords = np.linspace(0, old_cols - 1, old_cols)
        interp_func = RectBivariateSpline(row_coords, col_coords, self.psf_image)

        # Define new coordinates for the target shape
        new_row_coords = np.linspace(0, old_rows - 1, new_rows)
        new_col_coords = np.linspace(0, old_cols - 1, new_cols)

        # Evaluate the interpolator at the new coordinates
        self.psf_image = interp_func(new_row_coords, new_col_coords)

    def get_psf_image(self):
        """
        Returns the current PSF image.

        Returns
        -------
        np.ndarray
            The 2D image of the Point Spread Function.
        """
        return self.psf_image
