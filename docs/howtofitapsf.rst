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

  >>> import numpy as np
  >>> from scipy.stats import multivariate_normal
  >>> x, y = np.mgrid[-1:1:.05, -1:1:.05]
  >>> pos = np.empty(x.shape + (2,))
  >>> pos[:, :, 0] = x
  >>> pos[:, :, 1] = y
  >>> psf1 = multivariate_normal([0, 0.], [[2.0, 0.3], [0.3, 0.5]]).pdf(pos)
  >>> psf2 = multivariate_normal([0, 0.], [[1.0, 0.3], [0.3, 0.7]]).pdf(pos)
  >>> psf3 = multivariate_normal([0, 0.], [[1.0, 0], [0, 1.]]).pdf(pos)
  >>> psfbase = np.ma.dstack((psf1, psf2, psf3))
  >>> # Make an image as a linear combination of PSFs plus some noise
  >>> image = 1 * psf1 + 2 * psf2 + 3 * psf3
  >>> image += 0.3 * np.random.rand(*image.shape)
  >>> # Add a faint companion
  >>> image += 0.1 * multivariate_normal([0, 0.05], [[0.2, 0.], [0., 0.05]]).pdf(pos)
  >>> image2 =  2. * psf1 + 2.3 * psf2 + 2.6 * psf3
  >>> image2 += 0.3 * np.random.rand(*image.shape)
  >>> images = np.dstack([image, image2])

We can now instanciate a fitter object and use it to remove the PSF from ``image``, looking for what's left::

  >>> from psfsubtraction.fitpsf import fitters
  >>> my_fitter = fitters.SimpleSubtraction(image, psfbase)
  >>> residual = my_fitter.remove_psf()

Let's now compare the initial image and the PSF subtracted image (note the different scales on both images):

.. plot::

  import numpy as np
  from scipy.stats import multivariate_normal
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
  image += 0.1 * multivariate_normal([0, 0.3], [[0.05, 0.], [0., 0.05]]).pdf(pos)
  from psfsubtraction.fitpsf import fitters
  my_fitter = fitters.SimpleSubtraction(image, psfbase)
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
As shown in the example above, the PSF fit is done through a fitter object. This object determines which algorithm is used in the fit and it sets the parameters.

Two ways to fit several images to the same base
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first paradigm is to make a single fitter object which is given the psf base upon initialization and to set all required parameters on this object. We can then loop over this one object::

  >>> my_fitter = fitters.LOCI(psfbase)   # doctest: +SKIP
  >>> my_fitter.sector_radius_n = 5   # doctest: +SKIP
  >>> my_fitter.sector_phi = np.linspace(0.1, 6.1, 10) # not quite 2 pi.   # doctest: +SKIP 
  >>> subtracted = []   # doctest: +SKIP
  >>> for im in images:   # doctest: +SKIP
  ...     subtracted.append(my_fitter.remove_psf(im)))   # doctest: +SKIP

Alternatively, we can make a class that encapsualtes all the properties that we need and make a new fitter objects for each image. This requires a little more memory, but it makes each fit entirely independend and thus the whole process is easily paralizable. On the flipside, we have to write a little more code to to the same thing:

.. doctest-requires:: ipyparallel
		      
  >>> # get the clients for parallel execution ready.
  >>> from ipyparallel import Client
  >>> rc = Client()
  >>> dview = rc[:] # use all engines
  >>> # Prepare for the fit
  >>> class MyFitter(fitters.LOCI):
  ...     sector_radius_n = 5
  ...     sector_phi =  np.linspace(0.1, 6.1, 10) # not quite 2 pi.
  >>> fitterlist = [MyFitter(psfbase, im) for im in images]
  >>> # Do the fit
  >>> subtracted = dview.map_sync(lambda x: x.remove_psf, fitterlist)
  
Ingedients for a fitter
-----------------------

algorithm - how do the fitters work?
4 basic functions...


.. 
   toctree::
   :maxdepth: 1

   fitter.rst
   regions.rst
   optregion.rst
   findbase.rst
   fitpsf.rst
   utils.rst

.. automodapi:: psfsubtraction.fitpsf
