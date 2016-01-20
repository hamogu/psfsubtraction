Examples for fitter use
=======================

.. _sect-two-ways-to-fit-image-list:

Two ways to fit several images to the same base
-----------------------------------------------

The first paradigm is to make a single fitter object which is given the psf base upon initialization and to set all required parameters on this object. We can then loop over this one object::

  >>> from psfsubtraction.fitpsf import fitters
  >>> from psfsubtraction import data
  >>> psfbase, image, image2 = data.gaussian_PSFs()
  >>> my_fitter = fitters.LOCI(psfbase)
  >>> my_fitter.sector_radius_n = 5
  >>> my_fitter.sector_phi = 12
  >>> subtracted = []
  >>> for im in [image, image2]:
  ...     subtracted.append(my_fitter.remove_psf(im))

Alternatively, we can make a class that encapsualtes all the properties that we need and make a new fitter objects for each image. This requires a little more memory, but it makes each fit entirely independend and thus the whole process is easily paralizable. On the flipside, we have to write a little more code to to the same thing:

.. doctest-requires:: ipyparallel
		      
  >>> # get the clients for parallel execution ready.
  >>> from ipyparallel import Client
  >>> rc = Client()
  >>> dview = rc[:] # use all engines
  >>> # Prepare for the fit
  >>> class MyFitter(fitters.LOCI):
  ...     sector_radius_n = 5
  ...     sector_phi = 12
  >>> fitterlist = [MyFitter(psfbase, im) for im in [image, image2]]
  >>> # Do the fit
  >>> subtracted = dview.map_sync(lambda x: x.remove_psf, fitterlist)
