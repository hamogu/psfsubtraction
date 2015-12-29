from itertools import imap, ifilterfalse
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
    m = self.dim2to1(np.zeros_like(self.image, dtype=bool))
    m[pix] = True
    return m


def pixel_by_pixel(self):
    '''Each pixel it its own region.

    This is an extreme LOCI variant, where each region is made up of a single
    pixel only. However, for images with even a few thousand pixels that is
    prohibitive in run time.
    '''
    # image_unmasked returns a list with one element.
    # All pixels to be used are marked True.
    mask = image_unmasked(self)[0]

    return imap(mask_except_pixel, ifilterfalse(lambda x: mask[x], range(len(mask))))


# Since pixel_by_pixel is far too slow, we need a more intelligent scheme to divide
# the image into regions.
# The vast majority of pixels is NOT masked in any image. Process those in one chunk.

def all_plus_pixel_by_pixel(self):
    '''Simple all-at-one treatment with masked pixel special handling.

    All pixels that are unmasked in every image are treated at once (for speed),
    pixels that are masked in some of the images receive special treatment.

    reads self.min_usable_bases
    '''
    masked_somewhere = self.image1d.mask | (np.sum(self.psfbase1d.mask, axis=1) >=1)
    masked_in_too_many_bases = np.sum(self.psfbase1d.mask, axis=1) > self.min_usable_bases

    regionlist = [~masked_somewhere]

    pixelstoadd = masked_somewhere & ~self.image1d.mask & ~masked_in_too_many_bases
    for pix in np.nonzero(pixelstoadd)[0]:
        regionlist.append(mask_except_pixel(self, pix))

    return regionlist


class ExtremeLOCI(PSFSubtraction):

    fitpsfcoeff = psf_from_projection
    regions = all_plus_pixel_by_pixel
    min_usable_bases = 35

    def findbase(self, region):
        '''Return all bases that are not masked in any pixel in region'''
        indbase = ~np.ma.getmaskarray(self.psfbase1d)[region, :]
        # region could have several pixels in it.
        if region.sum() == 0:
            raise ValueError('The input region selects no pixel.')
        else:
            return np.min(indbase, axis=0)

    def fitregion(self, region, indpsf):
        '''Here we select the maximal region.

        The region is maximal in the sense that it includes all pixels that are
        not masked in the data or any of the bases.
        '''
        psfmask = np.ma.getmaskarray(self.psfbase1d.mask[:, indpsf])
        fitregion = np.max(psfmask, axis=1)
        fitregion *= np.ma.getmaskarray(self.image1d)
        # Go from masked=True, to an index array that is True where no mask is set
        fitregion = ~fitregion
        if fitregion.sum() <= (indpsf != False).sum():
            warn('Fit underdetermined. Choose larger fit region or smaller base.')
        return fitregion

        fitpsfcoeff = psf_from_projection
