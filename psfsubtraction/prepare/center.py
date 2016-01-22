# Licensed under a MIT licence - see file `license`
import numpy as np
from scipy import ndimage, stats
from astropy.nddata.utils import extract_array, _round

__all__ = ['fit_diffraction_spike',
           'center_from_spikes',
           'guess_center_nested']


def guess_center_nested(image, halfwidth=50):
    '''Guess the position of the central object as two-step process

    First, this function calculates the center of mass of an image.
    This works well if the central object is the only bright source, however
    even a moderately bright source that is far away can shift the center of
    mass of an image by a few pixels. To improve the first guess the function
    selects a subimage with the halfwidth ``halfwidth`` in a second step
    and calculates the center of mass of that subimage.

    Parameters
    ----------
    image : 2d np.array
        input image
    halfwidth : int
        half width of the subimage selected in the second step.

    Returns
    -------
    xm, ym : float
        x and y coordinates estimated position of the central object
    '''
    xm, ym = ndimage.center_of_mass(np.ma.masked_invalid(image))
    n = 2 * halfwidth + 1
    subimage, xmymsmall = extract_array(image, (n, n), (xm, ym),
                                        return_position=True)
    x1, y1 = ndimage.center_of_mass(np.ma.masked_invalid(subimage))
    # xmymsmall is the xm, ym position in the coordinates of subimage
    # So, correct the initial (xm, ym) by delta(xmsmall, x1)
    return xm + (x1 - xmymsmall[0]), ym + (y1 - xmymsmall[1])


def fit_diffraction_spike(image, fac=1, r_inner=50, r_outer=250, width=25,
                          initial_guess=ndimage.center_of_mass):
    '''fit a diffraction spike with a line

    Regions with low signal, i.e. regions where the maximum is not well
    determined (standard deviation < 40% percentile indicates that all pixels
    in that strip are very similar, meaning that these pixels contain mostly
    sky and not signal).


    Parameters
    ----------
    image : np.array
        2 dimensional image
    fac : float
        Initial guess for the parameter m in the equation y = m*x+b.
    r_inner, r_outer : float
        The fit is done on the left side of the image from ``-r_outer`` to
        ``-r_inner`` and on the right side from ``+r_inner`` to ``+r_outer``
        (all coordinates in pixels, measured from the center of the image).
        Use these parameters to contrain the fit to a region with a strong, but
        not saturated signal.
    width : float
        For each column in the image, a ``width`` strip of pixels centered on the
        initial guess is extracted and the position of the maximum is recorded.
    initial guess : tuple or callable
        This can either be a numeric value for an initial guess such as
        ``(54, 67.7)`` for a function that accepts and 2 d array (the image)
        and returns an tuple.

    Returns
    -------
    m, b : float
        coefficients for a line of the form y = m x + b
    '''
    if callable(initial_guess):
        xm, ym = initial_guess(image)
    else:
        xm, ym = initial_guess
    s = np.hstack([np.arange(-r_outer, -r_inner), np.arange(r_inner, r_outer)])
    x = xm + s
    y = ym + fac * s

    ytest = np.zeros((len(x), 2 * width + 1))
    ymax = np.zeros((len(x)))

    for i in range(len(x)):
        ytest[i, :] = image[_round(x[i]), _round(y[i] - width):
                            _round(y[i] + width + 1)]
    ymax = np.argmax(ytest, axis=1) - width

    # identify where there is actually a signal
    st = np.std(ytest, axis=1)
    ind = st > np.percentile(st, 40.)
    m, b, r_value, p_value, std_err = stats.linregress(x[ind],
                                                       y[ind] + ymax[ind])
    if np.abs(r_value) < 0.99:
        raise Exception("No good fit to spike found")
    return m, b


def center_from_spikes(image, **kwargs):
    '''Fit both diffraction spikes and return the intersection point

    .. note:: Direction of diffraction spikes.

       Currently this function assumes X shaped diffraction spikes.

    Parameters
    ----------
    image : np.array
        2 dimensional image

    All keyword arguments will be passed to `fit_diffraction_spike`.
    '''
    m1, b1 = fit_diffraction_spike(image, 1., **kwargs)
    m2, b2 = fit_diffraction_spike(image, -1., **kwargs)

    xmnew = (b1 - b2) / (m2 - m1)
    ymnew = m1 * xmnew + b1

    return xmnew, ymnew


# Masking streaks and spikes


def mask_spike(image, m, xm, ym, width=3):
    # make normalized vector normal to spike in (x,y)
    n = np.array([1, -1. / m])
    n = n / np.sqrt(np.sum(n**2))
    # Distance separately for x and y, because I did not find a matrix form
    # to write this dot product for each x in closed form
    x, y = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    dx = (x - xm) * n[0]
    dy = (y - ym) * n[1]
    r = np.abs(dx + dy)
    return r < width


def mask_spikes(image, m1, m2, maskwidth=3, **kwargs):
    xmnew, ymnew = center_from_spikes(image, **kwargs)
    mask1 = mask_spike(image, m1, xmnew, ymnew, width=maskwidth)
    mask2 = mask_spike(image, m2, xmnew, ymnew, width=maskwidth)
    return mask1 | mask2, xmnew, ymnew


def mask_readoutstreaks(image):
    '''logarithmic image to edge detect faint features

    This requires the `scikit-image <http://scikit-image.org/>`_ package.
    '''

    from skimage import filters as skfilter
    from skimage.morphology import convex_hull_image

    logimage = np.log10(np.clip(image, 1, 1e5)) / 5
    # Mask overexposed area + sobel edge detect
    mask = (skfilter.sobel(logimage) > 0.1) | (image > 0.6 * np.max(image))
    # pick out the feature that contain the center
    # I hope that this always is bit enough
    mask, lnum = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))

    i = mask[ndimage.center_of_mass(image)]
    mask = (mask == i)
    # fill any holes in that region
    return convex_hull_image(mask)
