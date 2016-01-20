import numpy as np
from scipy.stats import multivariate_normal

def gaussian_PSFs():
    '''Provide a simple set of PSFs and image for testing and example.

    Returns
    -------
    psfbase : np.array
        Three Gaussians with different width and covariance as PSF base
        functions.
    image, image2 : np.array
        Two images generated from the Gaussian PSFs with some added noise.
    '''
    x, y = np.mgrid[-1:1:.05, -1:1:.05]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    psf1 = multivariate_normal([0, 0.], [[2.0, 0.3], [0.3, 0.5]]).pdf(pos)
    psf2 = multivariate_normal([0, 0.], [[1.0, 0.3], [0.3, 0.7]]).pdf(pos)
    psf3 = multivariate_normal([0, 0.], [[1.0, 0], [0, 1.]]).pdf(pos)
    psfbase = np.ma.dstack((psf1, psf2, psf3))
    # Make an image as a linear combination of PSFs plus some noise
    image = 1 * psf1 + 2 * psf2 + 3 * psf3
    image += 0.3 * np.random.rand(*image.shape)
    # Add a faint companion
    image += 0.1 * multivariate_normal([0, 0.05], [[0.2, 0.], [0., 0.05]]).pdf(pos)
    image2 =  2. * psf1 + 2.3 * psf2 + 2.6 * psf3
    image2 += 0.3 * np.random.rand(*image.shape)
    return psfbase, image, image2
