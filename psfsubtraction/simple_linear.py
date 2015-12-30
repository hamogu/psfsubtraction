from itertools import imap, ifilterfalse
from collections import defaultdict
from warnings import warn
import numpy as np

from fit import PSFSubtraction
from fit import image_unmasked, findbase_allbases, fitregion_identity, \
    psf_from_projection


class SimpleLinearSubtraction(PSFSubtraction):
    '''Simple examples of PSF fitting.

    - The whole (unmasked) image is fit at once
    - using all bases.
    '''
    regions = image_unmasked
    findbase = findbase_allbases
    fitregion = fitregion_identity
    fitpsfcoeff = psf_from_projection


def mask_except_pixel(self, pix):
    '''Helper function - make True/False mask that is True at pix

    Parameters
    ----------
    pix : int
        Index in flattened image

    Returns
    -------
    m : np.array of bool
        Array that is True at and only at position ``pix``.
    '''
    m = self.dim2to1(np.zeros_like(self.image, dtype=bool))
    m[pix] = True
    return m


def pixel_by_pixel(self):
    '''Each pixel it its own region.

    This is an extreme LOCI variant, where each region is made up of a single
    pixel only.
    This function returns one region per unmasked image pixel.

    Note
    ----
    Even for images with just a few thousand pixels this method is
    too expensive in run time.

    Returns
    -------
    regions : iterator
        True/False index arrays
    '''
    # image_unmasked returns a list with one element.
    # All pixels to be used are marked True.
    mask = image_unmasked(self)[0]

    return imap(mask_except_pixel,
                ifilterfalse(lambda x: mask[x], range(len(mask))))


def group_by_basis(self):
    '''Group pixels with the same valid bases into one region.

    For each valid pixel in the image, this function checks which bases
    are valid at that pixel.
    It then groups pixels with the same valid bases into one region.

    Theoretically, there could be 2^60 combinations of bases for 60 bases,
    but in practice the valid pixels in the bases are not randomly
    distributed, so that typically a much smaller number of regions is
    generated.

    Returns
    -------
    regions : list of index arrays
    '''
    imagemask = np.ma.getmaskarray(self.image1d)
    basemask = np.ma.getmaskarray(self.psfbase1d)

    D = defaultdict(list)
    for i in range(imagemask.shape[0]):
        # Add to the dict UNLESS the image itself is masked.
        if not imagemask[i]:
            D[tuple(basemask[i, :])].append(i)

    return D.values()


class ExtremeLOCI(PSFSubtraction):

    fitpsfcoeff = psf_from_projection
    regions = group_by_basis
    min_usable_bases = 35

    def findbase(self, region):
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

    def fitregion(self, region, indpsf):
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
