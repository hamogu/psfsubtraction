# Licensed under a MIT licence - see file `license`
import numpy as np
from scipy.stats import multivariate_normal
import pytest


@pytest.fixture()
def example3_3():
    image = np.ma.array([[1., 2., 3.],
                         [4., 5., 6.],
                         [7., 8., 9.]],
                        mask=[[False, False, False],
                              [False, False, False],
                              [False, True, True]]
                        )

    psf1 = np.ma.array([[1., 2., 3.],
                        [4., 5., 100.],
                        [7., 8., 9.]],
                       mask=[[False, False, False],
                             [False, False, True],
                             [False, False, False]]
                       )

    psf2 = np.ma.array([[1., 2., 100.],
                        [4., 5., 6.],
                        [7., 8., 100.]],
                       mask=[[False, False, True],
                             [False, False, False],
                             [False, False, True]]
                       )
    psfarray = np.ma.dstack((psf1, psf2))
    return image, psfarray


@pytest.fixture()
def example40_40():
    x, y = np.mgrid[-1:1:.05, -1:1:.05]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    psf1 = multivariate_normal([0, 0.], [[2.0, 0.3], [0.3, 0.5]]).pdf(pos)
    psf2 = multivariate_normal([0, 0.], [[1.0, 0.3], [0.3, 0.7]]).pdf(pos)
    psf3 = multivariate_normal([0, 0.], [[1.0, 0], [0, 1.]]).pdf(pos)
    psfarray = np.ma.dstack((psf1, psf2, psf3))

    image = 1 * psf1 + 2 * psf2 + 3 * psf3
    np.random.seed(0)
    image += 0.3 * np.random.rand(*image.shape)

    return image, psfarray

@pytest.fixture()
def example40_40_masked(example40_40):

    image = example40_40[0]
    psf = np.ma.array(example40_40[1])

    # mask a couple of points
    for i in range(10):
        ind = np.random.choice(40, 2)
        image[ind[0], ind[1]] = 1e5

    image = np.ma.masked_greater(image, 1e4)

    for i in range(10):
        for j in range(3):
            ind = np.random.choice(40, 2)
            indj = np.random.choice(3)
            psf[ind[0], ind[1], indj] = 1e5

    psf = np.ma.masked_greater(psf, 1e4)

    return image, psf
