# Licensed under a MIT licence - see file `license`
'''Functions to select that bases used in a PSF fit.

These functions meant to be included into a `PSFFitter` object and this they
all take the ``self`` argument.

All functions here take two arguments:

- ``self``
- ``region``: See `region` for a description

and return a an index array that can be used to select the bases from the base
list.

'''
import numpy as np


def allbases(self, region):
    '''Return all available bases.'''
    return np.ones((self.psfbase.shape[2]), dtype=bool)


def nonmaskedbases(self, region):
    '''Return all bases that are not masked in any pixel in region'''
    indbase = ~np.ma.getmaskarray(self.psfbase1d)[region, :]
    # region could have several pixels in it.
    # region could be
    # - np.array/list/tuple of True / False
    # - np.array/list/tuple of index numbers
    check = np.asanyarray(region)
    if (check.dtype == bool and check.sum() == 0) or (len(check) == 0):
        raise ValueError('The input region selects no pixel.')
    else:
        return np.min(indbase, axis=0)
