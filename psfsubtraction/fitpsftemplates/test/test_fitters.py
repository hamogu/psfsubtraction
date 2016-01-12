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
