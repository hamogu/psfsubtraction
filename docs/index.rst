
Welcome to the psfsubtration documentation!

psfsubtraction is is closely related to the `AstroPy`_
package. The psfsubtraction package provides a tool box for the subtraction of
the point-spread-function (PSF) from an image using template images.
This technique is widely used to find and characterize close companions around
bright objects. The most famous example is probably exoplanet imaging, but the
same ideas are applied to other problems where a high contrast is required
e.g. to find main-sequence companions to Cepheids.

Many different variants of the same basic idea "fit a combination of templates"
are described in the literature. This python package provides tools for several
different approaches that can be combined and adjusted as needed.

Contributions are welcome, see https://github.com/hamogu/psfsubtraction .


Documentation
=============

The documentation for this package is here:

.. toctree::
  :maxdepth: 2

  install.rst
  prepare.rst
  howtofitapsf.rst
  
.. toctree::
  :maxdepth: 1

  changelog
