# Licensed under a MIT licence - see file `license`
import numpy as np
import pytest

from .. import fitters


@pytest.fixture(params=[fitters.SimpleSubtraction,
                        fitters.UseAllPixelsSubtraction,
                        fitters.LOCI,
                        fitters.LOCIAllPixelsSubtraction])
def myfitter(request, example40_40):
    return request.param(*example40_40)


def test_fitters_are_functinoal(example40_40, myfitter):
    '''Test is the fitters run as expected.

    The point here is not to benchmark the fitters, but only to see
    that they improve the image at all.
    This tests that all the intermediate arrays and indices that are used
    in the fitting have compatible shapes and data types,
    that there are no simple typos in the code etc.
    '''
    psf = myfitter.fit_psf()

    residual_im = myfitter.remove_psf()

    # There could always be a few pixels where the errors are bigger.
    # We want to check that at least 90 % of the pixels are subtracted well.
    # Detailed testing for individual algorithms is beyond this function.

    n = example40_40[0].size
    assert (np.abs(residual_im) < 0.2 * example40_40[0]).sum() > 0.9 * n
    assert (np.isclose(psf, example40_40[0], 0.2, 0.05)).sum() > 0.9 * n

@pytest.fixture(params=[fitters.SimpleSubtraction,
                        fitters.UseAllPixelsSubtraction,
                        fitters.LOCIAllPixelsSubtraction])
def myfitter_masked(request, example40_40_masked):
    return request.param(*example40_40_masked)


def test_fitters_are_functinoal_masked(example40_40_masked, myfitter_masked):
    '''Test is the fitters run as expected.

    The point here is not to benchmark the fitters, but only to see
    that they improve the image at all.
    This tests that all the intermediate arrays and indices that are used
    in the fitting have compatible shapes and data types,
    that there are no simple typos in the code etc.
    '''
    psf = myfitter_masked.fit_psf()

    residual_im = myfitter_masked.remove_psf()

    image = example40_40_masked[0]
    psfs = example40_40_masked[1]

    # There could always be a few pixels where the errors are bigger.
    # We want to check that at least 90 % of the pixels are subtracted well.
    # Detailed testing for individual algorithms is beyond this function.

    n = image.size - image.mask.sum()
    assert (np.abs(residual_im) < 0.2 * image).sum() > 0.9 * n
    assert (np.isclose(psf, image, 0.2, 0.05)).sum() > 0.9 * n
    # Exact numbers depend on the fitter an exact placement of masked pixels
    # Here, we test for integration. Detailed tests for individual methods are
    # elsewhere.
    assert residual_im.mask.sum() >= 10
    assert residual_im.mask.sum() <= 40


def test_error_on_masked():

    class PSF(fitters.BasePSFFitter):
        _allow_masked_data = False

    image = np.ones((4, 4))
    psf = np.ones(( 4, 4, 3))

    mimage = np.ma.array(image)
    mpsf = np.ma.array(psf)

    # masked array with no mask set works
    f = PSF(mimage, mpsf)

    # but if mask is set it raises an exception
    mimage[2, 2] = np.ma.masked
    mpsf[1, 3, 1] = np.ma.masked

    with pytest.raises(ValueError) as e:
        f = PSF(mimage, psf)
    assert 'cannot deal with masked' in str(e)

    with pytest.raises(ValueError) as e:
        f = PSF(image, mpsf)
    assert 'cannot deal with masked' in str(e)
