# Licensed under a MIT licence - see file `license`
'''Function to construct the "optimization region" for a region.

The "optimization region" is the region used in the fitting to find the best
PSF for a region. The "optimization region" can (but usually will not) be
identical to the region.
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

and return a "optimization region" which has the same format as `region`.

'''
from warnings import warn

import numpy as np
from scipy.ndimage import binary_dilation

from .utils import OptionalAttributeError

__all__ = ['identity', 'all_unmasked', 'dilated_region',
           'around_region', 'wrapper_optmask', 'wrapper_ignore_all_masked']


def identity(self, region, indpsf):
    '''Return a input region as optimization region.'''
    return region


def all_unmasked(self, region, indpsf):
    '''Here we select the maximal region.

    The region is maximal in the sense that it includes all pixels that are
    not masked in the data or any of the bases.

    Returns
    -------
    optregion : np.array of type bool
        True for those pixels that should be included in the fit
    '''
    psfmask = np.max(np.ma.getmaskarray(self.psfbase1d[:, indpsf]), axis=1)
    datamask = np.ma.getmaskarray(self.image1d)
    optregion = ~psfmask & ~datamask

    if optregion.sum() <= np.asarray(indpsf).sum():
        warn('Fit underdetermined. Choose larger optimization region or smaller base.')
    return optregion


def dilated_region(self, region, indpsf):
    '''Specify a optimization region that extends around the ``region``.

    This requires that the fitter has an attribute ``dilatation_region``, which can be

    - an int: In this case a square matrix of size 2 * n + 1 is used.
    - a matrix (see `scipy.ndimage.binary_dilation` for details.

    Examples
    --------

    >>> from psfsubtraction.fitpsf import fitters
    >>> from psfsubtraction.fitpsf import optregion
    >>> region = np.array([[True, False, False], \
                           [False, False, False], \
                           [False, False, False]])
    >>> class DilationFitter(fitters.SimpleSubtraction):
    ...     optregion = optregion.dilated_region
    ...     dilation_region = 1
    >>> dummy_image = np.ones((3, 3)) # boring image, but good enough for the example
    >>> dummy_psfs = np.ones((3,3,4)) # even more boring psf array.
    >>> myfitter = DilationFitter(dummy_psfs, dummy_image)
    >>> myfitter.optregion(region, [0]).reshape((3, 3))
    array([[ True,  True, False],
           [ True,  True, False],
           [False, False, False]])
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

    Examples
    --------

    >>> from psfsubtraction.fitpsf import fitters
    >>> from psfsubtraction.fitpsf import optregion
    >>> region = np.array([[True, False, False], \
                           [False, False, False], \
                           [False, False, False]])
    >>> dummy_image = np.ones((3, 3)) # boring image, but good enough for the example
    >>> dummy_psfs = np.ones((3,3,4)) # even more boring psf array.
    >>> class AroundFitter(fitters.SimpleSubtraction):
    ...     optregion = optregion.around_region
    ...     dilation_region = 1
    >>> myfitter = AroundFitter(dummy_psfs, dummy_image)
    >>> myfitter.optregion(region.ravel(), [0]).reshape((3, 3))
    array([[False,  True, False],
           [ True,  True, False],
           [False, False, False]])
    '''
    fitreg = dilated_region(self, region, indpsf)
    return fitreg & ~np.asarray(region)


def wrapper_optmask(func):
    '''Wrap an optregion function to apply an additional global mask.

    This function wraps the optregion function ``func``. Optimization regions
    are determined by that function, but are then additionally filtered
    such that points that are masked as ``True`` in ``self.optmask`` are
    not included in the optimization region.

    One use case is an image with a source hidden in the PSF. Assume that
    this source is already known. We want to include it in ``region`` to make
    sure that the PSF is removed under it, but we do not want to include when
    optimize the PSF.
    (A better alternative might be to fit its PSF at the same time, but that
    is beyond the scope of this module.)

    Parameters
    ----------
    func : callable
        function to be wrapped.

    Returns
    -------
    func_and_optmask : function
        wrapped ``func``
    '''
    def func_and_optmask(self, region, indpsf):
        if not hasattr(self, "optmask"):
            raise OptionalAttributeError('Fit object needs to define a boolean array optmask.')
        if not hasattr(self.optmask, "shape") or \
           not ((self.optmask.shape == self.image.shape) or
                (self.optmask.shape == self.image1d.shape)):
            raise OptionalAttributeError('"optmask" must have same shape as image.')
        optregion = func(self, region, indpsf)

        optregion_and_optmask = optregion & ~self.optmask.flatten()
        if optregion_and_optmask.sum() <= np.asarray(indpsf).sum():
            warn('Fit underdetermined. Ignoring fitmask.')
            return optregion
        else:
            return optregion_and_optmask

    return func_and_optmask


def wrapper_ignore_all_masked(func):
    '''Wrap a optregion function to remove all masked pixels from optregion.

    This function wraps the optregion function ``func``. Optimization regions
    are determined by that function, but are then additionally filtered
    such that points that are masked in either the image or and used psfbase
    are not part of the returned ``optregion``.

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
        optregion = self.anyreg_to_mask(func(self, region, indpsf))

        psfmask = np.max(np.ma.getmaskarray(self.psfbase1d[:, indpsf]), axis=1)
        datamask = np.ma.getmaskarray(self.image1d)
        optregion = optregion & ~psfmask & ~datamask

        if optregion.sum() <= np.asarray(indpsf).sum():
            warn('Fit underdetermined. Choose larger optimization region or smaller base.')
        return optregion

    return func_unmasked
