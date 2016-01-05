import numpy as np
from scipy import ndimage, stats

from skimage import filters as skfilter
from skimage.morphology import convex_hull_image


def fit_diffraction_spike(image, fac=1, r_inner=50, r_outer=250, width=25):
    '''fit a diffraction spike with a line

    The fit is done on the left side of the image from ``-r_outer`` to
    ``-r_inner`` and on the right side from ``+r_inner`` to ``+r_outer``
    (all coordinates in pixels, measured from the center of the image).
    ``fac`` is an initial guess for the parameter m in the equation y = m*x+b.
    For each column in the image, a ``width`` strip of pixels centered on the
    initial guess is extracted and the position of the maximum is recorded.
    Regions with low signal, i.e. regions where the maximum is not well
    determined (standard deviation < 40% percentile indicates that all pixels
    in that strip are very similar -> there is not signal).
    '''
    xm, ym = ndimage.center_of_mass(image)
    s = np.hstack([np.arange(-r_outer, -r_inner), np.arange(r_inner, r_outer)])
    x = xm + s
    y = ym + fac * s

    ytest = np.zeros((len(x), 2 * width + 1))
    ymax = np.zeros((len(x)))

    for i in range(len(x)):
        ytest[i, :] = image[np.int(x[i]), np.int(y[i] - width):
                            np.int(y[i] + width + 1)]
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
    '''Fit both diffraction spikes and return the intersection point'''
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
    # logarithmic image to edge detect faint features
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
