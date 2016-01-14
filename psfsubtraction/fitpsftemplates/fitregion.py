# Licensed under a MIT licence - see file `license`
'''Function to construct the "fit region" for a region.

The "fit region" is the region used in the fitting to find the best PSF for a
region. The "fit region" can (but usually will not) be identical to the region.
For example, if the base PSF is flat, and the region we are looking at
contains some flat area and also a source, then the fit might tell us to
subtract a constant so that the mean of the subtracted data is zero.
Instead, if we look for the best fit for the source and we use only the
surrounding flat region, then we will be left with the source after the
subtraction - exactly what we want.

These functions meant to be included into a `PSFFitter` object and this they
all take the ``self`` argument.

All functions here take three arguments:

- ``self``
- ``region``: See `region` for a description
- ``indpsf``: See `findbase`.

and return a "fit region" which has the same format as `region`.

'''
from warnings import warn

import numpy as np
from scipy.ndimage import binary_dilation

from .utils import OptionalAttributeError


def identity(self, region, indpsf):
    '''Return a input region as fit region.'''
    return region


def all_unmasked(self, region, indpsf):
    '''Here we select the maximal region.

    The region is maximal in the sense that it includes all pixels that are
    not masked in the data or any of the bases.

    Returns
    -------
    fitregion : np.array of type bool
        True for those pixels that should be included in the fit
    '''
    psfmask = np.max(np.ma.getmaskarray(self.psfbase1d[:, indpsf]), axis=1)
    datamask = np.ma.getmaskarray(self.image1d)
    fitregion = ~psfmask & ~datamask

    if fitregion.sum() <= np.asarray(indpsf).sum():
        warn('Fit underdetermined. Choose larger fit region or smaller base.')
    return fitregion


def dilated_region(self, region, indpsf):
    '''Specify a fit region that extends around the ``region``.

    This requires that the fitter has an attribute ``dilatation_region``, which can be

    - an int: In this case a square matrix of size 2 * n + 1 is used.
    - a matrix (see `scipy.ndimage.binary_dilation` for details.

    Example
    -------

    >>> from psfsubtraction.fitpsftemplates import fitters
    >>> from psfsubtraction.fitpsftemplates import fitregion
    >>> region = np.array([[True, False, False], \
                           [False, False, False], \
                           [False, False, False]])
    >>> class DilationFitter(fitters.SimpleSubtraction):
    ...     fitregion = fitregion.dilated_region
    ...     dilation_region = 1
    >>> dummy_image = np.ones((3, 3)) # boring image, but good enough for the example
    >>> dummy_psfs = np.ones((3,3,4)) # even more boring psf array.
    >>> myfitter = DilationFitter(dummy_image, dummy_psfs)
    >>> myfitter.fitregion(region, [0]).reshape((3, 3))
    array([[ True,  True, False],
           [ True,  True, False],
           [False, False, False]], dtype=bool)
    '''
    if not hasattr(self, 'dilation_region'):
        raise OptionalAttributeError('Fitter must speficy the `self.dilation_region`\n'
                                     + 'which is either and int or a square matrix.')
    if np.isscalar(self.dilation_region):
        selem = np.ones((2 * self.dilation_region + 1, 2 * self.dilation_region + 1))
    else:
        selem = self.dilation_region
    return self.dim2to1(binary_dilation(self.dim1to2(region), selem))


def around_region(self, region, indpsf):
    '''similar to `dilated_region`, but exclude all pixels in ``region`` itself.

    See `dilated_region` for options.

    Example
    -------

    >>> from psfsubtraction.fitpsftemplates import fitters
    >>> from psfsubtraction.fitpsftemplates import fitregion
    >>> region = np.array([[True, False, False], \
                           [False, False, False], \
                           [False, False, False]])
    >>> dummy_image = np.ones((3, 3)) # boring image, but good enough for the example
    >>> dummy_psfs = np.ones((3,3,4)) # even more boring psf array.
    >>> class AroundFitter(fitters.SimpleSubtraction):
    ...     fitregion = fitregion.around_region
    ...     dilation_region = 1
    >>> myfitter = AroundFitter(dummy_image, dummy_psfs)
    >>> myfitter.fitregion(region.ravel(), [0]).reshape((3, 3))
    array([[False,  True, False],
           [ True,  True, False],
           [False, False, False]], dtype=bool)
    '''
    fitreg = dilated_region(self, region, indpsf)
    return fitreg & ~np.asarray(region)


def wrapper_fitmask(func):
    '''Wrap a fitregion function to apply an additional global mask.

    This function wraps the fitregion function ``func``. Fit regions are
    determined by that function, but are then additionally filtered
    such that points that are masked as ``True`` in ``self.fitmask`` are
    not included in the fit region.

    One use case is an image with a source hidden in the PSF. Assume that
    this source is already known. We want to include it in ``region`` to make
    sure that the PSF is removed under it, but we do not want to include it
    in the fit of the PSF.
    (A better alternative might be to fit its PSF at the same time, but that
    is beyond the scope of this module.)

    Parameters
    ----------
    func : callable
        function to be wrapped.

    Returns
    -------
    func_and_fitmask : function
        wrapped ``func``
    '''
    def func_and_fitmask(self, region, indpsf):
        if not hasattr(self, "fitmask"):
            raise OptionalAttributeError('Fit object needs to define a boolean array fitmask.')
        if not hasattr(self.fitmask, "shape") or \
           not ((self.fitmask.shape == self.image.shape) or
                (self.fitmask.shape == self.image1d.shape)):
            raise OptionalAttributeError('"fitmask" must have same shape as image.')
        fitregion = func(self, region, indpsf)

        fitregion_and_fitmask = fitregion & ~self.fitmask.flatten()
        if fitregion_and_fitmask.sum() <= np.asarray(indpsf).sum():
            warn('Fit underdetermined. Ignoring fitmask.')
            return fitregion
        else:
            return fitregion_and_fitmask

    return func_and_fitmask

def wrapper_ignore_all_masked(func):
    '''Wrap a fitregion function to remove all masked pixels from fitregion.

    This function wraps the fitregion function ``func``. Fit regions are
    determined by that function, but are then additionally filtered
    such that points that are masked in either the image or and used psfbase
    are not part of the returned ``fitregion``.

    Parameters
    ----------
    func : callable
        function to be wrapped.

    Returns
    -------
    func_and_fitmask : function
        wrapped ``func``
    '''
    def func_unmasked(self, region, indpsf):
        fitregion = self.anyreg_to_mask(func(self, region, indpsf))

        psfmask = np.max(np.ma.getmaskarray(self.psfbase1d[:, indpsf]), axis=1)
        datamask = np.ma.getmaskarray(self.image1d)
        fitregion = fitregion & ~psfmask & ~datamask

        if fitregion.sum() <= np.asarray(indpsf).sum():
            warn('Fit underdetermined. Choose larger fit region or smaller base.')
        return fitregion

    return func_unmasked
