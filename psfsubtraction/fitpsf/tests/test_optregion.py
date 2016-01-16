# Licensed under a MIT licence - see file `license`
import numpy as np
import pytest

from .. import fitters
from .. import optregion
from ..utils import OptionalAttributeError


def test_identity(example3_3):
    image, psfarray = example3_3
    class PSF(fitters.BasePSFFitter):
        optregion = optregion.identity

    f = PSF(image, psfarray)
    assert np.all(f.optregion(~image.mask.flatten(), [True, True]) == ~image.mask.flatten())
    assert np.all(f.optregion(~image.mask.flatten(), [False, True])== ~image.mask.flatten())


def test_unmasked(example3_3):
    image, psfarray = example3_3
    class psf(fitters.BasePSFFitter):
        optregion = optregion.all_unmasked

    f = psf(image, psfarray)

    reg1 = np.array([[True, True, False],
                     [True, True, False],
                     [True, False, False]])
    reg2 = np.array([[True, True, False],
                     [True, True, True],
                     [True, False, False]])
    assert np.all(f.optregion(~image.mask.flatten(), np.array([True, True])) == reg1.flatten())
    assert np.all(f.optregion(~image.mask.flatten(), np.array([False, True])) == reg2.flatten())


def test_wrapper_fitmask(example3_3):
    image, psfarray = example3_3

    class psf(fitters.BasePSFFitter):
        optregion = optregion.wrapper_fitmask(optregion.all_unmasked)

        fitmask = np.array([[True, True, False],
                            [False, False, False],
                            [False, False, False]])

    f = psf(image, psfarray)
    reg1 = np.array([[False, False, False],
                     [True, True, False],
                     [True, False, False]])

    assert np.all(f.optregion(~image.mask.flatten(), np.array([True, True])) == reg1.flatten())

    f.fitmask = False
    with pytest.raises(OptionalAttributeError) as e:
        temp = f.optregion(~image.mask.flatten(), np.array([True, True]))
    assert "must have same shape" in str(e.value)


def test_dilated_region_int(example3_3):
    image, psfarray = example3_3

    region = np.array([[True, False, False],
                       [False, False, False],
                       [False, False, False]])

    class DilationFitter(fitters.BasePSFFitter):
        optregion = optregion.dilated_region
        dilation_region = 1

    myfitter = DilationFitter(image, psfarray)
    fitreg = myfitter.optregion(region, [0])
    expected = np.array([[True, True, False],
                         [True, True, False],
                         [False, False, False]], dtype=bool)
    assert np.all(fitreg == expected.ravel())


def test_dilated_region_array(example3_3):
    image, psfarray = example3_3

    region = np.array([[True, False, False],
                       [False, False, False],
                       [False, False, False]])

    class DilationFitter(fitters.BasePSFFitter):
        optregion = optregion.dilated_region
        dilation_region = np.array([[False, True, False],
                                    [True, True, True],
                                    [False, True, False]])

    myfitter = DilationFitter(image, psfarray)
    fitreg = myfitter.optregion(region, [0])
    expected = np.array([[True, True, False],
                         [True, False, False],
                         [False, False, False]], dtype=bool)
    assert np.all(fitreg == expected.ravel())


def test_region_around_array(example3_3):
    image, psfarray = example3_3

    region = np.array([[True, False, False],
                       [False, False, False],
                       [False, False, False]])

    class DilationFitter(fitters.BasePSFFitter):
        optregion = optregion.around_region
        dilation_region = np.array([[False, True, False],
                                    [True, True, True],
                                    [False, True, False]])

    myfitter = DilationFitter(image, psfarray)
    fitreg = myfitter.optregion(region.ravel(), [0])
    expected = np.array([[False, True, False],
                         [True, False, False],
                         [False, False, False]], dtype=bool)
    assert np.all(fitreg == expected.ravel())
