Fit and remove the PSF (`psfsubtraction.fitpsf`)
================================================
This is the core of the package.


Conventions
-----------

+ All positions and distances are given in pixels.
+ Positions are given as floats where the unit of pixels is implied.
+ Angles require a `~astropy.units.Quantity` object.

Data with missing or bad values
-------------------------------

We often need a template-based PSF subtraction when the central object is very bright, at least much brighter than the companion we want to detect. Thus, we often have pixels in the image that are saturated or bleed columns or other artifacts that should not be used in the fitting. There might also be cosmic-rays or hot pixels on the detector. While it is often possibly to minimize the number of unusable pixels with a clever observing strategy, this package implements mechanisms to deal with bad or missing values.

If any pixel has a bad or missing value, all data (image and PSF base) should be expressed as `numpy.ma.masked_array`, where the mask is set to ``True`` for bad or missing values.

Most functions in this package accept masked arrays and those that do not will raise an exception when they detected masked pixels.

Examples
--------

For the examples below, we will generate some artifical data here (TODO: bundle some real data with the package for use in examples and testing)::

  >>> from psfsubtraction import data
  >>> psfbase, image, image2 = data.gaussian_PSFs()
  
We can now instanciate a fitter object and use it to remove the PSF from ``image``, looking for what's left::

  >>> from psfsubtraction.fitpsf import fitters
  >>> my_fitter = fitters.SimpleSubtraction(psfbase, image)
  >>> residual = my_fitter.remove_psf()

Let's now compare the initial image and the PSF subtracted image (note the different scales on both images):

.. plot::

  from psfsubtraction import data
  psfbase, image, image2 = data.gaussian_PSFs()

  from psfsubtraction.fitpsf import fitters
  my_fitter = fitters.SimpleSubtraction(psfbase, image)
  residual = my_fitter.remove_psf()

  import matplotlib.pylab as plt
  fig = plt.figure(figsize=(8, 4))
  ax1 = fig.add_subplot(121)
  im1 = ax1.imshow(image, interpolation='nearest', cmap=plt.cm.hot)
  ax1.set_title('image')
  plt.colorbar(im1, ax=ax1)
  ax2 = fig.add_subplot(122)
  im2 = ax2.imshow(residual, interpolation='nearest', cmap=plt.cm.hot)
  plt.colorbar(im2, ax=ax2)
  ax2.set_title('Subtracted image')

  
The fitter object
-----------------
As shown in the example above, the PSF fit is done through a fitter object. This object specifies the algorithm (e.g. is the entire image fitted at once? If not, how are regions split up? How is the fit optimized? What about masked values?).

A fitter is initialized with a (masked) numpy array of PSF bases in the shape *(m, n, k)* where *(n, m)* are the dimensions of the images and *k* is the number of bases::

  >>> my_fitter = fitters.SimpleSubtraction(psfbase)

This package comes with a selection of fitters which are listed below in the API docs and :ref:`sect-newfitters` shows how to define variants of those or completely new fitters.

Specifying an image for the fitter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In order to do anything useful, a fitter object also needs an *(n, m)* (masked) numpy array for the image. This can either be set when the fitter object is created::

  >>> my_fitter = fitters.SimpleSubtraction(psfbase, image)

or set as an attribute later::

  >>> my_fitter.image = image

or be passed as an argument to the `~psfsubtraction.fitpsf.fitters.BasePSFFitter.remove_psf` or `~psfsubtraction.fitpsf.fitters.BasePSFFitter.fit_psf` functions::

  >>> fitted_psf = my_fitter.fit_psf(image)

See :ref:`sect-two-ways-to-fit-image-list` for an example where each of these possibilities might be useful.


Specifying parameters for a fitter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In some cases, attributes of the fitter object control the parameters of the fit, e.g. a `~psfsubtraction.fitpsf.fitters.LOCI` fitter splits an image into concentric rings. It has an attribute `fitter.sector_radius_n` that determines the number of such rings::

  >>> my_fitter = fitters.LOCI(psfbase)
  >>> my_fitter.sector_radius_n = 5
  
If a fitter has options, they are listed in the docstring.

Masked data
^^^^^^^^^^^
Not all fitters can deal with bad or missing data. However, all fitters accept
`~numpy.ma.masked_array` objects as input, provided there are no maked
values. This is done so that the same code can be used to read in the data and
prepare the arrays for both fitters that treat masked data and those that do
not. Most fitters that do **not** accept masked data will issue a warning or an
exception when they encounter a mask.


Ingredients for a fitter
-----------------------

algorithm - how do the fitters work?
4 basic functions...


.. toctree::
   :maxdepth: 1

   newfitters.rst
   advanced_fitter_use.rst

.. automodapi:: psfsubtraction.fitpsf
