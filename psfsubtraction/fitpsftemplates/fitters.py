# Licensed under a MIT licence - see file `license`
from warnings import warn

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

    '''Regions with fewer than ``min_pixels_in_region`` will be ignored for speed-up.'''
    min_pixels_in_region = 1

    '''Can this fitter deal with masked data?

    This attribute is not fool-proof; it is set by hand for the pre-defined fitters.
    If you define your own fitter, you will ahve to check yourself if if works with
    masked data.'''
    _allow_masked_data = True

    def __init__(self, image, psfbase):
        if image.shape != psfbase.shape[:2]:
            raise ValueError('Each PSF must have same dimension as image.')
        if len(psfbase.shape) != 3:
            raise ValueError('psfbase must have 3 dim [im_x, im_y, n]')
        if not self._allow_masked_data and \
           (np.ma.getmask(image).sum() > 0 or np.ma.getmask(psfbase).sum() > 0):
            raise ValueError('This fitter cannot deal with masked data.')
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
            reg = self.anyreg_to_mask(r)
            if reg.sum() >= self.min_pixels_in_region:
                yield reg
            else:
                warn('Skipping region that includes no pixels.')


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

            # Check for consistency
            self.check_fittable(region, fitregion, indpsf)

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

    def check_fittable(self, region, fitregion, indpsf):
        n_data = (fitregion & ~np.ma.getmaskarray(self.image1d)).sum()
        n_pars = indpsf.sum()
        if n_data.sum() <= n_pars.sum():
            raise RegionError('Fit region contains only {0} data points to fit {1} coefficients.'.format(n_data.sum(), n_pars.sum()))


class SimpleSubtraction(BasePSFFitter):
    '''Simple examples of PSF fitting.

    - The whole (unmasked) image is fit at once
    - using all bases.
    '''
    regions = regions.image_unmasked
    findbase = findbase.allbases
    fitregion = fitregion.all_unmasked
    fitpsfcoeff = fitpsf.psf_from_projection


class UseAllPixelsSubtraction(BasePSFFitter):
    '''Use all available pixels of the image.

    For unmasked image pixel the maximal set of PSF templates that are
    unmasked at that position is used.
    Pixels are then group in regions that make use of the same PSF templates.
    '''
    regions = regions.group_by_basis
    findbase = findbase.nonmaskedbases
    fitregion = fitregion.all_unmasked
    fitpsfcoeff = fitpsf.psf_from_projection


class LOCI(BasePSFFitter):
    '''LOCI fitter (locally optimized combination of images)

    The loci algorithm was introduced in the following paper
    `Lafreniere et al. 2007, ApJ, 660, 770 <http://adsabs.harvard.edu/abs/2007ApJ...660..770L>`_.

    The default parameters in this fitter are chosen similar to the shape of
    the regions used in that paper.
    '''

    '''Can this fitter deal with masked data?

    No, in this case, because it is possible that sectors defines a region
    where no unmasked bases exist.'''
    _allow_masked_data = False

    regions = regions.sectors

    @property
    def sector_radius(self):
        return np.logspace(self.sector_radius_inner,
                           np.log10(self.image.shape[1]),
                           self.sector_radius_n)

    sector_radius_inner = 0
    sector_radius_n = 10
    sector_phi = 12

    findbase = findbase.nonmaskedbases

    fitregion = fitregion.wrapper_ignore_all_masked(fitregion.around_region)
    dilation_region = 10

    fitpsfcoeff = fitpsf.psf_from_projection


class LOCIAllPixelsSubtraction(LOCI):
    '''LOCI fitter that splits LOCI regions according to the available PSF bases.

    For unmasked image pixel the maximal set of PSF templates that are
    unmasked at that position is used.
    Pixels are then group in regions that make use of the same PSF templates.
    '''

    '''Can this fitter deal with masked data?'''
    _allow_masked_data = True

    regions = regions.sectors_by_basis
