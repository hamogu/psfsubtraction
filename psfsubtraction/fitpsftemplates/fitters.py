import numpy as np

import regions
import findbase
import fitregion
import fitpsf


class RegionError(Exception):
    '''Region does not have the right shape or dtype'''
    pass


class PSFIndexError(Exception):
    '''PSF Index array does not have the right shape or dtype'''
    pass


class BasePSFFitter(object):
    '''Base object for PSF fitting.

    Parameters
    ----------
    image : np.array of shape (n,m)
        N, M array
    psfbase : np.ndarray of shape (n, m, k)
        array of psfbases. (n, m) are the dimensions of each image
        and there are k potential elements of the PSF base.
    '''
    def __init__(self, image, psfbase):
        if image.shape != psfbase.shape[:2]:
            raise ValueError('Each PSF must have same dimension as image.')
        if len(psfbase.shape) != 3:
            raise ValueError('psfbase must have 3 dim [im_x, im_y, n]')
        self.image = image
        self.psfbase = psfbase

    ### Convenience functions and infrastructure ###

    @property
    def image1d(self):
        return self.dim2to1(self.image)

    @property
    def psfbase1d(self):
        return self.psfbase.reshape((-1, self.psfbase.shape[2]))

    def dim2to1(self, array2d):
        '''Flatten image'''
        return array2d.ravel()

    def dim1to2(self, array1d):
        '''Reshape flattened image to 2 d.'''
        return array1d.reshape(self.image.shape)

    ### Functions that should be overwritten by child classes ###

    def regions(self):
        '''This function should be overwritten by derived classes.'''
        raise NotImplementedError

    def findbase(self, region):
        '''This function should be overwritten by derived classes.'''
        raise NotImplementedError

    def fitregion(self, region, indpsf):
        '''This function should be overwritten by derived classes.'''
        raise NotImplementedError

    def fitpsfcoeff(self, img, base):
        '''This function should be overwritten by derived classes.'''
        raise NotImplementedError

    ### Some wrapper around the above classes to unify output formats, check
    ### validity etc.

    def anyreg_to_mask(self, reg):
        '''Convert any type of a region definition to a 1d boolean mask.

        Also check that the region has the correct size.

        Parameters
        ----------
        r : boolean mask of size image in 1d or 2d or 1d integer array
        '''
        r = np.asanyarray(reg)
        # Index array like [1,5,12,23]
        if (r.ndim == 1) and np.issubdtype(r.dtype, np.int64):
            region = np.zeros((self.image1d.shape), dtype=bool)
            region[r] = True
            r = region
        if r.ndim == 2:
            r = r.ravel()
        if r.shape != self.image1d.shape:
            raise RegionError("Every region must have the same shape as the image.")
        return r

    def baseind_to_mask(self, indpsf):
        '''Convert any type of psf base index to boolen mask.'''
        indpsf = np.asanyarray(indpsf)
        if (indpsf.ndim == 1) and np.issubdtype(indpsf.dtype, np.int64):
            ind = np.zeros((self.psfbase.shape[2]), dype=bool)
            ind[indpsf] = True
            indpsf = ind
        if indpsf.shape != (self.psfbase.shape[2], ):
            raise PSFIndexError("PSF index shape does not match the shape of the psf base.")
        return indpsf

    def iter_regions(self):
        '''Convert all allowed regions formats to a 1d boolean mask array'''
        for r in self.regions():
            yield self.anyreg_to_mask(r)


    ### Here the actual work is done ###

    def fit_psf(self):
        # This line triggers in bug in numpy < 1.9
        # psf = np.zeros_like(self.image1d)
        # It results in psf.mask is self.image1d.mask
        # Here is a different way to get the same result:
        psf = np.ma.zeros(len(self.image1d))

        psf[:] = np.ma.masked
        for region in self.iter_regions():
            # select which bases to use
            indpsf = self.baseind_to_mask(self.findbase(region))
            # Select which region to use in the optimization
            fitregion = self.anyreg_to_mask(self.fitregion(region, indpsf))
            # Perform fit on the fitregion
            psf_coeff = self.fitpsfcoeff(self.image1d[fitregion],
                                         self.psfbase1d[:, indpsf][fitregion, :])
            # Use psfcoeff to estimate the psf in `region`
            psf[region] = np.dot(self.psfbase1d[:, indpsf][region, :],
                                 psf_coeff)
        return self.dim1to2(psf)

    def remove_psf(self):
        psf = self.fit_psf()
        return self.dim1to2(self.image1d) - psf


class SimpleSubtraction(BasePSFFitter):
    '''Simple examples of PSF fitting.

    - The whole (unmasked) image is fit at once
    - using all bases.
    '''
    regions = regions.image_unmasked
    findbase = findbase.allbases
    fitregion = fitregion.identity
    fitpsfcoeff = fitpsf.psf_from_projection


class UseAllPixelsSubtraction(BasePSFFitter):
    regions = regions.group_by_basis
    findbase = findbase.nonmaskedbases
    fitregion = fitregion.all_unmasked
    fitpsfcoeff = fitpsf.psf_from_projection
