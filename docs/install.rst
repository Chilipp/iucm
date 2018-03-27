.. _install:

How to install
==============

.. highlight:: bash

Installation using conda
------------------------
We highly recommend to use conda for the installation of IUCM. Packages have
been built for python 2.7 and 3.6 for windows, OSX and Linux.

Just download a `miniconda installer`_, add the `conda-forge channel`_ to your
configurations and install iucm from the `chilipp channel`_::

    conda config --add channels conda-forge
    conda install -c chilipp iucm

.. _miniconda installer: https://conda.io/miniconda.html
.. _conda-forge channel: https://conda-forge.org/
.. _chilipp channel: https://anaconda.org/chilipp

Installation via pip
--------------------
After having installed the necessary :ref:`requirements`, install iucm from
PyPi.org_ via::

    $ pip install iucm

.. _PyPi.org: https://pypi.org/

Installation from scratch
-------------------------

After having installed the necessary :ref:`requirements`, clone the
`Github repository` and install it via::

    $ python setup.py install

.. _Github repository: https://github.com/Chilipp/iucm

.. _requirements:

Requirements
------------
This package depends on

- python >= 2.7
- Cython_
- numpy_
- scipy_
- xarray_
- psyplot_
- netCDF4_
- funcargparse_
- model-organization_

To install all the necessary Packages in a conda environment *iucm*, type::

    $ conda create -n iucm -c conda-forge cython psyplot netCDF4 scipy
    $ conda activate iucm
    $ pip install model-organization

.. _python: https://www.python.org/
.. _Cython: http://docs.cython.org/en/latest/
.. _numpy: http://www.numpy.org/
.. _scipy: https://scipy.org/
.. _xarray: http://xarray.pydata.org/
.. _psyplot: http://psyplot.readthedocs.io/
.. _netCDF4: http://unidata.github.io/netcdf4-python/
.. _model-organization: http://model-organization.readthedocs.io/en/latest/
.. _funcargparse: http://funcargparse.rtfd.io/
