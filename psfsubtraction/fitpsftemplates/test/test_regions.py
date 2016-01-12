import numpy as np
import pytest

import astropy.units as u

from .. import fitters
from .. import regions
from ..utils import OptionalAttributeError, bool_indarray

from .small_examples import image, psfarray


def test_image_at_once():
    class psf(fitters.SimpleSubtraction):
        regions = regions.image_at_once

    f = psf(image, psfarray)
    regs = f.regions()
    assert len(regs) == 1
    assert regs[0].shape == (9, )
    assert np.all(regs[0] == ~image.mask.flatten())


def test_image_unmasked():
    class psf(fitters.SimpleSubtraction):
        regions = regions.image_unmasked

    # image with mask
    f = psf(image, psfarray)
    regs = f.regions()
    assert len(regs) == 1
    assert np.all(regs[0] == ~image.mask.flatten())

    # image without mask
    f = psf(np.ones((3, 3)), psfarray)
    regs = f.regions()
    assert len(regs) == 1
    assert regs[0].shape == (9, )
    assert np.all(regs[0])


def test_group_by_basis():
    class psf(fitters.SimpleSubtraction):
        regions = regions.group_by_basis

    f = psf(image, psfarray)
    regs = f.regions()
    assert len(regs) == 3
    # order is given by implementation, but does not matter at all.
    r1 = np.array([True, True, False, True, True, False, True, False, False])
    r2 = np.array([False, False, False, False, False, True, False, False, False])
    r3 = np.array([False, False, True, False, False, False, False, False, False])
    # We don't know the order of regs, so check if any of the three matches
    # and keep a list of which one matches which.
    regfound = []
    for r in regs:
        for i, ri in enumerate([r1, r2, r3]):
            if np.all(bool_indarray(9, r) == ri):
                regfound.append(i)
                break
    assert set(regfound) == set([0, 1, 2])

    f.min_number_of_bases = 2
    assert len(f.regions()) == 1


def test_sector_regions():
    im = np.arange(1200).reshape((30, 40))
    psfs = np.ones((30, 40, 15))
    for center in [(1,7), None]:
        for r, phi in zip([np.arange(55), np.array([0, 1, 5, 55])],
                      [5, np.linspace(0., 360., 5.) * u.degree]):
            class psf(fitters.SimpleSubtraction):
                regions = regions.sectors(r, phi, center)

            f = psf(im, psfs)
            regs = np.dstack(list(f.regions()))
            regs = regs.reshape((1200, -1))
            # Test that each pixel is part of one and only one regions
            assert np.all(regs.sum(axis=1) == 1)

    # test a region that has a hole in the middle
        class psf(fitters.SimpleSubtraction):
            regions = regions.sectors([5, 10, 50], phi, center)

        f = psf(im, psfs)
        regs = np.dstack(list(f.regions()))
        regs = regs.reshape((1200, -1))
        # Test that each pixel is part of one and only one regions
        assert regs.sum() < 1200
