# Licensed under a MIT licence - see file `license`
import numpy as np

import astropy.units as u

from .. import fitters
from .. import regions
from ..utils import OptionalAttributeError, bool_indarray


def test_image_at_once(example3_3):
    psfarray, image = example3_3

    class psf(fitters.BasePSFFitter):
        regions = regions.image_at_once

    f = psf(psfarray, image)
    regs = f.regions()
    assert len(list(regs)) == 1
    assert regs[0].shape == (9, )
    assert np.all(regs[0] == ~image.mask.flatten())


def test_image_unmasked(example3_3):
    psfarray, image = example3_3

    class psf(fitters.BasePSFFitter):
        regions = regions.image_unmasked

    # image with mask
    f = psf(psfarray, image)
    regs = list(f.regions())
    assert len(regs) == 1
    assert np.all(regs[0] == ~image.mask.flatten())

    # image without mask
    f = psf(psfarray, np.ones((3, 3)))
    regs = list(f.regions())
    assert len(regs) == 1
    assert regs[0].shape == (9, )
    assert np.all(regs[0])


def test_group_by_basis(example3_3):
    psfarray, image = example3_3

    class psf(fitters.BasePSFFitter):
        regions = regions.group_by_basis

    f = psf(psfarray, image)
    regs = list(f.regions())
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
    assert len(list(f.regions())) == 1


def test_sector_regions():
    im = np.arange(1200).reshape((30, 40))
    psfs = np.ones((30, 40, 15))
    for center in [(1, 7), None]:
        for r, phi in zip([np.arange(55), np.array([0, 1, 5, 55])],
                          [5, np.linspace(0., 360., 5.) * u.degree]):
            class PSF(fitters.BasePSFFitter):
                regions = regions.sectors
                sector_radius = r
                sector_phi = phi
                sector_center = center

            f = PSF(psfs, im)
            regs = np.dstack(list(f.regions()))
            regs = regs.reshape((1200, -1))
            # Test that each pixel is part of one and only one region
            assert np.all(regs.sum(axis=1) == 1)

    # test a region that has a hole in the middle
        class PSF2(PSF):
            sector_radius = [5, 10, 50]

        f = PSF2(psfs, im)
        regs = np.dstack(list(f.regions()))
        regs = regs.reshape((1200, -1))
        # Test that each pixel is part of one and only one region
        assert regs.sum() < 1200
