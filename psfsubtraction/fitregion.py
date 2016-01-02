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
from warning import warn

import numpy as np


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

        if fitregion.sum() <= (indpsf != False).sum():
            warn('Fit underdetermined. Choose larger fit region or smaller base.')
        return fitregion
