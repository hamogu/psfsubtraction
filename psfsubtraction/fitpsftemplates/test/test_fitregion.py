import numpy as np
import pytest

from .. import fitters
from .. import fitregion
from ..utils import OptionalAttributeError

from .small_examples import image, psfarray

def test_identity():
    f = fitters.SimpleSubtraction(image, psfarray)
    assert np.all(f.fitregion(~image.mask.flatten(), [True, True]) ==  ~image.mask.flatten())
    assert np.all(f.fitregion(~image.mask.flatten(), [False, True])==  ~image.mask.flatten())

def test_unmasked():
    class psf(fitters.SimpleSubtraction):
        fitregion = fitregion.all_unmasked

    f = psf(image, psfarray)

    reg1 = np.array([[True, True, False],
                     [True, True, False],
                     [True, False, False]])
    reg2 = np.array([[True, True, False],
                     [True, True, True],
                     [True, False, False]])
    assert np.all(f.fitregion(~image.mask.flatten(), np.array([True, True])) == reg1.flatten())
    assert np.all(f.fitregion(~image.mask.flatten(), np.array([False, True])) == reg2.flatten())

def test_wrapper_fitmask():
    class psf(fitters.SimpleSubtraction):
        fitregion = fitregion.wrapper_fitmask(fitregion.all_unmasked)

        fitmask = np.array([[ True, True, False],
                            [ False, False, False],
                            [ False, False, False]])

    f = psf(image, psfarray)
    reg1 = np.array([[False, False, False],
                     [True, True, False],
                     [True, False, False]])

    assert np.all(f.fitregion(~image.mask.flatten(), np.array([True, True])) == reg1.flatten())

    f.fitmask = False
    with pytest.raises(OptionalAttributeError) as e:
        temp = f.fitregion(~image.mask.flatten(), np.array([True, True]))
    assert "must have same shape" in str(e.value)
