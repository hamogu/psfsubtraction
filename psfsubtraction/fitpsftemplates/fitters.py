import numpy as np

import regions
import findbase
import fitregion
import fitpsf


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

    ### Here the actual work is done ###

    def fit_psf(self):
        # This line triggers in bug in numpy < 1.9
        # psf = np.zeros_like(self.image1d)
        # It results in psf.mask is self.image1d.mask
        # Here is a different way to get the same result:
        psf = np.ma.zeros(len(self.image1d))

        psf[:] = np.ma.masked
        for region in self.regions():
            # select which bases to use
            indpsf = self.findbase(region)
            # Select which region to use in the optimization
            fitregion = self.fitregion(region, indpsf)
            psf_coeff = self.fitpsfcoeff(self.image1d[fitregion],
                                         self.psfbase1d[:, indpsf][fitregion, :])
            psf[region] = np.dot(self.psfbase1d[:, indpsf][region, :],
                                 psf_coeff)
        return self.dim1to2(psf)

    def remove_psf(self):
        psf = self.fit_psf()
        return self.dim1to2(self.image1d - psf)


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
