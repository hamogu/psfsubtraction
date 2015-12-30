import numpy as np

import regions
import findbase
import fitregion
import fitpsf


class PSFFitter(object):

    '''The region that is used for fitting.

    This can be different from ``region``. In fact, it is usually a good idea
    not to use the region where we want to perform the subtraction in the fit,
    otherwise the fit will be biased to oversubtract.
    For example, if the base PSF is flat, and the region we are looking at
    contains some flat area and also a source, then the fit might tell us to
    subtract a constant so that the mean of the subtracted data is zero.
    Instead, if we look for the best fit for the source and we use only the
    surrounding flat region, then we will be left with the source after the
    subtraction - exactly what we want.
    '''
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

    def __init__(self, image, psfbase):
        if image.shape != psfbase.shape[:2]:
            raise ValueError('Each PSF must have same dimension as image.')
        if len(psfbase.shape) != 3:
            raise ValueError('psfbase must have 3 dim [im_x, im_y, n]')
        self.image = image
        self.image1d = self.dim2to1(image)
        self.psfbase = psfbase
        self.psfbase1d = psfbase.reshape((-1, psfbase.shape[2]))

    def dim2to1(self, array2d):
        return array2d.ravel()

    def dim1to2(self, array1d):
        return array1d.reshape(self.image.shape)

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
        return psf

    def remove_psf(self):
        psf = self.fit_psf()
        return self.image1d - psf


class SimpleLinearSubtraction(PSFFitter):
    '''Simple examples of PSF fitting.

    - The whole (unmasked) image is fit at once
    - using all bases.
    '''
    regions = regions.image_unmasked
    findbase = findbase.allbases
    fitregion = fitregion.identity
    fitpsfcoeff = fitpsf.psf_from_projection


class ExtremeLOCI(PSFFitter):

    fitpsfcoeff = fitpsf.psf_from_projection
    findbase = findbase.nonmaskedbases
    fitregion = fitregion.all_unmasked
    regions = regions.group_by_basis
