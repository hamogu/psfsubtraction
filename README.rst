The ``psfsubtraction`` package
==============================

This is a package for fitting the PSF of a very bright object in CCD images with templates. Often, this approach allows for a better description of the PSF than the fit with an analytical model, specifically in the wings. This technique is important for planet detection in images or any other close object with a larger luminosity contrast, e.g. main-sequence companions of giant or supergiant stars.

Currently the package is not far enough developed to be useful except for developers. More information will be added here later.

dependencies
------------
numpy
scipy
astropy >= 1.1
photutils
