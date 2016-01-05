import numpy as np

class OptionalAttributeError(Exception):
    '''Problem with optional attribute in the PSF fitter class.

    Some functions require that more than the usual attributes
    ``image``, ``image1d``, ``psfbase``, and ``psfbase1d`` are defined.
    These other attributes are *optional* in the sense that most fitter
    objects will not need them, but they might be *required* for e.g.
    certain ``fitregion`` functions.
    '''
    pass


def bool_indarray(shape, index):
    '''Convert index list to boolean index array.

    Parameters
    ----------
    shape : tuple
        shape of output array
    index : index array
        can be a list of index values or an array of True/False

    Returns
    -------
    indarr : np.array
        ``indarr`` is a boolean array of shape ``shpe``.
    '''
    indarr = np.zeros(shape, dtype=bool)
    indarr[index] = True
    return indarr
