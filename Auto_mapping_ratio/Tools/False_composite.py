import numpy as np
import os
import bisect

"""
Module constructed from the machine learning toolkit for CRISM data
Plebani et al 2022
"""

class False_composite:

    def __init__(self):
        pass

    def _imadjust(self, src, tol=5, vin=(0, 255), vout=(0, 255)):
        """Adjust image histogram."""
        # from: https://stackoverflow.com/a/44611551
        tol = max(0, min(100, tol))
        if tol > 0:
            hist = np.histogram(src, bins=list(range(256)), range=(0, 255))[0]
            cum = np.cumsum(hist)

            size = src.shape[0] * src.shape[1]
            lb_, ub_ = size * tol / 100, size * (100 - tol) / 100
            vin = (bisect.bisect_left(cum, lb_), bisect.bisect_left(cum, ub_))

        scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
        vs_ = src - vin[0]
        vs_[src < vin[0]] = 0
        vd_ = vs_*scale + 0.5 + vout[0]
        vd_[vd_ > vout[1]] = vout[1]

        return vd_

    def loadmat(self, fname):
        """Load matlab files.

        Load Matlab files using the scipy interfaces and falling back to 'mat73'
        for the new HDF5 format.

        Parameters
        ----------
        fname: str
            Matlab file to open

        Returns
        -------
        mat: dict
            a dictionary storing the Matlab variables
        """
        # pylint: disable=import-outside-toplevel
        try:
            from scipy.io import loadmat as _loadmat
            return _loadmat(fname)
        except NotImplementedError as ex:
            # scipy loads only files with version 7.3 or earlier
            try:
                from mat73 import loadmat as _loadmat73
                return _loadmat73(fname)
            except ImportError as mat_ex:
                raise ex from mat_ex
            
    def crism_to_mat(self, fname, flatten=False):
        """Convert a CRISM ENVI image to a Matlab-like dictionary.

        Loads an ENVI image as a Matlab-like dictionary with spectra (IF) and pixel
        coordinates (x, y). If the header (.hdr) is not found, it is automatically
        generated from a .lbl file (using the approach in the
        `CRISM spectral calculator`_); if neither is available, an error is raised.

        .. _CRISM spectral calculator: https://github.com/jlaura/crism/blob/\
            master/csas.py

        Parameters
        ----------
        fname: str
            ENVI file to open (.hdr or .img)
        flatten: bool
            flatten an image array to (npix, nchan) and saves the coordinates to
            the x,y fields (default: False)

        Returns
        -------
        mat: dict
            a dictionary storing the spectra and the pixels coordinates (if flatten
            is True)
        """
        # pylint: disable=import-outside-toplevel
        from spectral.io import envi, spyfile

        band_select = np.r_[433:185:-1, 170:-1:68]

        fbase, _ = os.path.splitext(fname)
        try:
            img = envi.open(f"{fbase}.hdr")
        except spyfile.FileNotFoundError:
            self._generate_envi_header(f"{fbase}.lbl")
            img = envi.open(f"{fbase}.hdr")

        arr = img.load()

        mdict = {'IF': arr[:, :, band_select]}
        if flatten:  # use coordinate arrays for indexing
            xx_, yy_ = np.meshgrid(np.arange(arr.shape[1]),
                                np.arange(arr.shape[0]))
            mdict.update({'x': xx_.ravel() + 1, 'y': yy_.ravel() + 1})
            mdict['IF'] = mdict['IF'].reshape((-1, len(band_select)))

        return mdict

    def _generate_envi_header(self, lbl_fname):
        """Generate a HDR file from the LBL file when the former is missing."""
        # see: https://github.com/jlaura/crism/blob/master/csas.py
        fbase, _ = os.path.splitext(lbl_fname)

        with open(lbl_fname, 'r') as fid:
            for line in fid:
                if "LINES" in line:
                    lines = int(line.split("=")[1])
                if "LINE_SAMPLES" in line:
                    samples = int(line.split("=")[1])
                if "BANDS" in line:
                    bands = int(line.split("=")[1])

        with open(f"{fbase}.hdr", 'w') as fid:
            fid.write(
                f"ENVI\nsamples = {samples}\nlines   = {lines}\nbands   = {bands}"
                "\nheader offset = 0\nfile type = ENVI Standard\ndata type = 4\n"
                "interleave = bil\nbyte order = 0")

    def load_image(self,fname):
        """Try to load a .mat file and fall back to ENVI if not found."""
        try:
            return self.loadmat(fname)
        except (FileNotFoundError, NotImplementedError, ValueError):
            return self.crism_to_mat(fname, flatten=True)


    def image_shape(self, mat):
        """Get the image shape from the pixel x and y coordinates."""
        return (np.max(mat['y']), np.max(mat['x']))

    def filter_bad_pixels(self, pixspec, copy=False):
        """Remove large, infinite or NaN values from the spectra.

        Parameters
        ----------
        pixspec: ndarray
            the set of spectra to clean
        copy: bool
            if a new array must be returned; by default the orginal is overwritten

        Returns
        -------
        pixspec: ndarray
            the cleaned spectra with bad pixels set to the mean of all channels
        rem: ndarray
            boolean mask of bad pixels, with the first n-1 dimensions of pixspec
        """
        N_BANDS = 438
        if copy:
            pixspec = pixspec.copy()

        bad = (pixspec > 1e3) | ~np.isfinite(pixspec)
        if np.any(bad):
            pixspec[bad] = np.mean(pixspec[~bad])

        rem = np.sum(bad[..., :N_BANDS], axis=-1) > 0

        return pixspec, rem.reshape(pixspec.shape[:-1])

    def norm_minmax(self, pixspec, vmin=None, vmax=None, axis=0):
        """Normalize features in the [0,1] range.

        Parameters
        ----------
        pixspec: ndarray
            the set of spectra to normalize
        vmin: float
            custom minimum value; if None, computed from data
        vmax: float
            custom maximum value; if None, computed from data
        axis: int
            dimension to normalize (default: 0)

        Returns
        -------
        res: ndarray
            the normalized spectra
        """
        if vmin is None or vmax is None:
            vmin = np.min(pixspec, axis=axis, keepdims=True)
            vmax = np.max(pixspec, axis=axis, keepdims=True)

        diff = vmax - vmin
        diff[diff == 0] = 1.0   # avoid division by 0
        return (pixspec - vmin) / diff

    def get_false_colors(self, pixspec, badpix, channels=(233, 103, 20)):
        """Return the spectra as false color image on selected bands.

        Convert a spectral cube to an RGB image, using a median filter of size 17
        around selected bands for the R, G and B channels. The pixels are
        normalized first with L2 and then with min-max; finally, histogram
        equalization is applied to each channel.

        Parameters
        ----------
        pixspec: ndarray
            spectra shaped as a (h, w, n_channels) array
        badpix: ndarray
            boolean mask of bad pixels to be excluded from equalization
        channels: tuple
            tuple of three indices specifing the R, G and B bands

        Returns
        -------
        img: ndarray
            the false color image with channels normalized between 0 and 1
        """
        shape = badpix.shape
        badpix, pixspec = badpix.ravel(), pixspec.reshape((-1, pixspec.shape[-1]))

        lsize, rsize = 8, 9  # median filter size: 17
        img = np.stack([np.median(pixspec[:, max(i - lsize, 0):i + rsize], axis=1)
                        for i in channels]).T

        goodpx = img[~badpix, :]
        img /= np.mean(np.sqrt(np.einsum('ij,ij->i', goodpx, goodpx)), axis=0,
                    keepdims=True)

        vmin = np.min(img[~badpix, :], keepdims=True)
        vmax = np.max(img[~badpix, :], keepdims=True)
        img = 255*self.norm_minmax(img, vmin=vmin, vmax=vmax).reshape(shape + (-1,))

        img = np.stack([self._imadjust(im) for im in np.rollaxis(img, 2)], axis=2)
        return img / 255.0