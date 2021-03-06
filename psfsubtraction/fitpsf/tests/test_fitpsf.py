# Licensed under a MIT licence - see file `license`
import pytest

from .. import fitters
from .. import regions
from .. import findbase
from .. import optregion
from .. import fit


class CannotDealWithMasked(fitters.BasePSFFitter):
    '''

    We set _allow_masked_data to True and also chose a combination
    of functions that does **not** deal with masked values, because we
    want to check that the fitpsf function catches that mistake.
    '''
    _allow_masked_data = True
    regions = regions.image_unmasked
    findbase = findbase.allbases
    optregion = optregion.identity
    fitpsfcoeff = fit.psf_from_projection


def test_cannotdealwithmasked(example3_3):
    '''check that psf_from_projection raises an erro if given masked data.'''
    f = CannotDealWithMasked(*example3_3)
    with pytest.raises(ValueError) as e:
        psf = f.fit_psf()
    assert 'cannot deal with masked' in str(e)
