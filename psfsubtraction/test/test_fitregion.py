import numpy as np
import pytest

from .. import fitters
from .. import fitregion
from ..utils import OptionalAttributeError

image = np.ma.array([[1., 2., 3.],
                     [4., 5., 6.],
                     [7., 8., 9.]],
                    mask = [[False, False, False],
                            [False, False, False],
                            [False, True, True]]
                    )

psf1 = np.ma.array([[1., 2., 3.],
                    [4., 5., 100.],
                    [7., 8., 9.]],
                    mask = [[False, False, False],
                            [False, False, True],
                            [False, False, False]]
                    )

psf2 = np.ma.array([[1., 2., 100.],
                    [4., 5., 6.],
                    [7., 8., 100.]],
                   mask = [[False, False, True],
                           [False, False, False],
                           [False, False, True]]
                    )
psfarray = np.ma.dstack((psf1, psf2))

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
