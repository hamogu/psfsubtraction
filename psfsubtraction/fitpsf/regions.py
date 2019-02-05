# Licensed under a MIT licence - see file `license`
'''Functions that construct an iterable to iterate over regions.

These functions meant to be included into a `PSFFitter` object and thus they
all take the ``self`` argument.
The function should return an iterable, e.g. an iterator or a list of regions.
A "region" should be be a True/False index array of the same size as the
flattend (1-dimensional representation) image.
'''
from collections import defaultdict
from itertools import filterfalse
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle

from .utils import OptionalAttributeError

__all__ = ['image_at_once', 'image_unmasked', 'pixel_by_pixel',
           'group_by_basis', 'sectors', 'sectors_by_basis',
           'mask_except_pixel']


def image_at_once(self):
    '''Fit whole image at one.

    Returns
    -------
    regions: list
        List of one element (the image)
    '''
    return [np.ones_like(self.image1d, dtype=bool)]


def image_unmasked(self):
    '''Return the unmasked part of an image.

    Returns
    -------
    regions: list
        List of one element (the image)
    '''
    if hasattr(self.image, 'mask'):
        return [~np.ma.getmaskarray(self.image1d)]
    else:
        return image_at_once(self)


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

    .. note::

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

    return map(mask_except_pixel,
               filterfalse(lambda x: mask[x], range(len(mask))))


def group_by_basis(self):
    '''Group pixels with the same valid bases into one region.

    For each valid pixel in the image, this function checks which bases
    are valid at that pixel.
    It then groups pixels with the same valid bases into one region.

    If ``self.min_number_of_bases`` is set to an integer, only regions with
    at least that many valid bases are returned (default is 1).

    Theoretically, there could be 2^60 combinations of bases for 60 bases,
    but in practice the valid pixels in the bases are not randomly
    distributed, so that typically a much smaller number of regions is
    generated.

    Returns
    -------
    regions : list of index arrays
    '''
    imagemask = np.ma.getmaskarray(self.image1d)
    for r in _by_basis(self, imagemask):
        yield r


def _by_basis(self, imagemask):
    '''part of group_by_bases, refactored'''
    basemask = np.ma.getmaskarray(self.psfbase1d)

    min_bases = getattr(self, "min_number_of_bases", 1)

    D = defaultdict(list)
    for i in range(imagemask.shape[0]):
        # Add to the dict UNLESS the image itself is masked or TOO FEW bases
        # are valid
        if not imagemask[i] and ((~basemask[i, :]).sum() >= min_bases):
            D[tuple(basemask[i, :])].append(i)

    return D.values()


def sectors(self):
    '''Generate a function that generates sector regions

    A pixel is included in a region, if the pixel center falls within the
    region boundaries.

    This function makes use of the following fitter attributes, which have to
    be set to use this function:

    fitter.sector_radius : np.array
        boundaries for sector elements in pixels.
    fitter.sector_phi : int or `~astropy.quantity`
        If this is an int it sets the number of sectors that make up a
        full circle.
        If this is an `astropy.quantity` it is interpreted as the
        boundaries of the angular bins. It should cover the range from
        0 to 2 pi (or 360 deg, if units is degrees).
    fitter.sector_center : tuple or None
        x, y position of the center of all sectors (in pixel coordinates).
        ``None`` selects the center of the input image.

    Returns
    -------
    regions : generator
        sector regions
    '''
    try:
        phi = self.sector_phi
    except AttributeError:
        raise OptionalAttributeError('Fitter must speficy the `self.sector_phi`')
    if np.isscalar(phi):
        phi = np.linspace(0, 2 * np.pi, int(phi) + 1) * u.radian

    try:
        radius = self.sector_radius
    except AttributeError:
        raise OptionalAttributeError('Fitter must speficy the `self.sector_radius`')

    image_center = getattr(self, 'sector_center', None)
    if image_center is None:
        center = np.array(self.image.shape) / 2.
    else:
        center = image_center
    indices = np.indices(self.image.shape)
    x = indices[0, ...] - center[0]
    y = indices[1, ...] - center[1]
    r = np.sqrt(x**2 + y**2)
    # express as complex number and then take angle
    phiarr = np.angle(x + y * 1j)
    # Now turn into astro.coordinates.Angle object
    # because that has wrap_at and compares with astropy.quantities.
    phiarr = Angle(phiarr * u.rad).wrap_at('360d')

    for ri in range(len(radius) - 1):
        for phii in range(len(phi) - 1 ):
            sector = (r >= radius[ri]) & (r < radius[ri + 1]) \
                & (phiarr >= phi[phii]) & (phiarr < phi[phii + 1])
            if sector.sum() > 0:
                yield sector


def sectors_by_basis(self):
    '''Combines `sectors` and `group_by_basis`.

    The pixels in every sector will be grouped by their bases.
    '''
    for reg in sectors(self):
        for r in _by_basis(self, ~self.anyreg_to_mask(reg)):
            yield r


def wrapper_min_size(func):
    '''Wrap a region generator to remove all regions smaller than a limit

    This function wraps the region generator ``func``. Regions are
    generated by that function, but are then additionally filtered
    such that regions with fewer than ``self.region_min_size`` are skipped.
    (All regions are returned if the fitter does not has a
    ``region_min_size`` attribute.)

    Parameters
    ----------
    func : callable
        function to be wrapped.

    Returns
    -------
    func_min_size : function
        wrapped ``func``
    '''
    def func_min_size(self):
        for r in func(self):
            r = self.anyreg_to_mask(r)
            if r.sum() >= getattr(self, "region_min_size", 1):
                yield r
    return func_min_size
