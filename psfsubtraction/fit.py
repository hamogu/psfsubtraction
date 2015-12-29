import numpy as np


# regionlist
def regionlist_template(self):
    '''
    Returns
    -------
    regionlist : iterable
        Iterable (e.g. a list) of all regions to fit.
    '''
    raise NotImplementedError
    return regionlist  # list of 1d index arrays


def image_at_once(self):
    return [np.ones_like(self.image1d, dtype=bool)]


def image_unmasked(self):
    if hasattr(self.image, 'mask'):
        return [~np.ma.getmaskarray(self.image1d)]
    else:
        return image_at_once()


# fitregion
def fitregion_template(self, region, indpsf):
    raise NotImplementedError
    return fitregion


def fitregion_identity(self, region, indpsf):
    return region


# findbase
def findbase_template(self, region):
    raise NotImplementedError
    return bases_index


def findbase_allbases(self, region):
    return np.ones((self.psfbase.shape[2]), dtype=bool)


# fit psf_coeff
def fitpsfcoeff_template(self, image1d, psfbase):
    raise NotImplementedError
    return psf_coeff


def psf_from_projection(self, image1d, psfbase):
    '''solve a linear algebra system for the best PSF

    Parameters
    ----------
    image1d : array in 1 dim
    psfbase : array in [M,N]
        M = number of pixels in flattened image
        N = number of images that form the space of potential PSFs

    Returns
    -------
    psf_coeff : array in 1 dim
        Coefficients for a linear combination of ``psfbase`` elements that
        that give the optimal PSF.
    '''
    a = np.dot(psfbase.T, psfbase)
    b = np.dot(psfbase.T, image1d)
    psf_coeff = np.linalg.solve(a, b)
    return psf_coeff


class PSFSubtraction(object):

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
    regions = regionlist_template
    findbase = findbase_template
    fitregion = fitregion_template
    fitpsfcoeff = fitpsfcoeff_template

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
        # psf = np.zeros_like(self.image1d)
        # For numpy < 1.9 this results in psf.mask is self.image1d.mask
        psf = np.ma.zeros(len(self.image1d))
        psf[:] = np.ma.masked
        for region in self.regions():
            indpsf = self.findbase(region)  # select which bases to use
            fitregion = self.fitregion(region, indpsf)  # select which region to fit
            psf_coeff = self.fitpsfcoeff(self.image1d[fitregion],
                                         self.psfbase1d[:, indpsf][fitregion, :])
            psf[region] = np.dot(self.psfbase1d[:, indpsf][region, :],
                                 psf_coeff)
        return psf
