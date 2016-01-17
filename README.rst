The ``psfsubtraction`` package
==============================

.. image:: https://travis-ci.org/hamogu/psfsubtraction.svg?branch=master
    :target: https://travis-ci.org/hamogu/psfsubtraction
    :alt: Travis CI status

.. image:: https://coveralls.io/repos/hamogu/psfsubtraction/badge.svg?branch=master&service=github
    :target: https://coveralls.io/github/hamogu/psfsubtraction?branch=master 
    :alt: Percentage of code covered in tests.	 

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org/
    :alt: Powered by astropy

.. image:: https://readthedocs.org/projects/psfsubtraction/badge/?version=latest
    :target: http://psfsubtraction.readthedocs.org/en/latest/
    :alt: Documentation Status for master branch
	   
This is a package for fitting the PSF of a very bright object in CCD images with templates. Often, this approach allows for a better description of the PSF than the fit with an analytical model, specifically in the wings. This technique is important for planet detection in images or any other close object with a larger luminosity contrast, e.g. main-sequence companions of giant or supergiant stars.

Documentation can be found at http://psfsubtraction.readthedocs.org/en/latest/

