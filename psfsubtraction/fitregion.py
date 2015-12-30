'''Function to construct the "fit region" for a region.

The "fit region" is the region used in the fitting to find the best PSF for a
region. The "fit region" can (but usually will not) be identical to the region.
For example, to find the best fitting PSF for pixel A, one might want to
optimize using an annulus around the region.
If the region itself is used in the fit, the PSF might be oversubtracted.

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
