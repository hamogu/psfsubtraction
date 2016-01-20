************
Installation
************

Requirements
============

Ccdproc has the following requirements:

- `Astropy`_ v1.1 or later
- `Numpy <http://www.numpy.org/>`_
- `Scipy <http://www.scipy.org/>`_
- (optional) `scikit-image <http://scikit-image.org/>`_

One easy way to get these dependencies is to install a python distribution like `anaconda <http://continuum.io/>`_.

Installing psfsubtraction
=========================

.. comment NOT on PIPY yet

   Using pip
   -------------

   To install ccdproc with `pip <http://www.pip-installer.org/en/latest/>`_, simply run::

       pip install --no-deps psfsubtraction

   .. note::

       The ``--no-deps`` flag is optional, but highly recommended if you already
       have Numpy installed, since otherwise pip will sometimes try to "help" you
       by upgrading your Numpy installation, which may not always be desired.

Building from source
====================

Obtaining the source packages
-----------------------------

Source packages
^^^^^^^^^^^^^^^

At this early stage of development not source packages are available.

.. comment Not on PiPy yet
   The latest stable source package for ccdproc can be `downloaded here
   <https://pypi.python.org/pypi/psfsubtraction>`_.

Development repository
^^^^^^^^^^^^^^^^^^^^^^

The latest development version of ccdproc can be cloned from github
using this command::

   git clone git://github.com/hamogu/psfsubtraction.git

Building and Installing
-----------------------

To build psfsubtraction (from the root of the source tree)::

    python setup.py build

To install psfsubtraction (from the root of the source tree)::

    python setup.py install

Testing a source code build of ccdproc
--------------------------------------

The easiest way to test that your psfsubtraction built correctly (without
installing it) is to run this from the root of the source tree::

    python setup.py test

