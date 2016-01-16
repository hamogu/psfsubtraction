Fit and remove the PSF (`psfsubtraction.fitpsf`)
================================================
This is the core of the package.


Conventions
-----------

+ All positions and distances are given in pixels. 

Data with missing or bad values
-------------------------------

Example
-------

with images.


The fitter object
-----------------
As shown in the example above, the PSF fit is done through a fitter object. This object determines which algorithm is used in the fit and it sets the parameters.

Two ways to fit several images to the same base
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first paradigm is to make a single fitter object which is given the psf base upon initialization and to set all required parameters on this object. We can then loop over this one object::

  >>> my_fitter = psfsubtraction.fitpsf.LOCI(psfbase)
  >>> my_fitter.sector_radius_n = 5
  >>> my_fitter.sector_phi = np.linspace(0.1, 6.1, 10) # not quite 2 pi. 
  >>> subtracted = []
  >>> for im in images:
  ...     subtracted.append(my_fitter.remove_psf(im)))

Alternatively, we can make a class that encapsualtes all the properties that we need and make a new fitter objects for each image. This requires a little more memory, but it makes each fit entirely independend and thus the whole process is easily paralizable. On the flipside, we have to write a little more code to to the same thing:

.. doctest-requires:: ipyparallel
		      
  >>> # get the clients for parallel execution ready.
  >>> from ipyparallel import Client
  >>> rc = Client()
  >>> dview = rc[:] # use all engines
  >>> # Prepare for the fit
  >>> class MyFitter(psfsubtraction.fitpsg.LOCI):
  ...     sector_radius_n = 5
  ...     sector_phi =  np.linspace(0.1, 6.1, 10) # not quite 2 pi.
  >>> fitterlist = [MyFitter(psfbase, im) for im in images]
  >>> # Do the fit
  >>> subtracted = dview.map_sync(lambda fitter: fitter.remove_psf, fitterlist)
  
Ingedients for a fitter
-----------------------

algorithm - how do the fitters work?
4 basic functions...


.. toctree::
   :maxdepth: 1

   fitter.rst
   regions.rst
   optregion.rst
   findbase.rst
   fitpsf.rst
   utils.rst

.. automodapi:: psfsubtraction.fitpsf
