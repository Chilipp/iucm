============================================
IUCM - The Integrated Urban Complexity Model
============================================

.. start-badges

.. list-table::
    :stub-columns: 1
    :widths: 10 90

    * - docs
      - |docs|
    * - package
      - |version| |conda| |supported-versions| |supported-implementations|

.. |docs| image:: http://readthedocs.org/projects/iucm/badge/?version=latest
    :alt: Documentation Status
    :target: http://iucm.readthedocs.io/en/latest/?badge=latest

.. |version| image:: https://img.shields.io/pypi/v/iucm.svg?style=flat
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/iucm

.. |conda| image:: https://anaconda.org/chilipp/iucm/badges/installer/conda.svg
    :alt: conda
    :target: https://conda.anaconda.org/chilipp

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/iucm.svg?style=flat
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/iucm

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/iucm.svg?style=flat
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/iucm

.. end-badges

This model simulates urban growth and transformation with the objective of
minimising the energy required for transportation.

The full documentation and installation instructions are available on
http://iucm.readthedocs.io.


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

Then run pip::

    $ pip install iucm

to install the package from pypi, or::

    $ python setup.py install

for an installation from a local source. We also provide a conda-package via::

    $ conda install -c chilipp iucm

More detailed installation instructions can be found in the `installation docs`_.

.. _python: https://www.python.org/
.. _Cython: http://docs.cython.org/en/latest/
.. _numpy: http://www.numpy.org/
.. _scipy: https://scipy.org/
.. _xarray: http://xarray.pydata.org/
.. _psyplot: http://psyplot.readthedocs.io/
.. _netCDF4: http://unidata.github.io/netcdf4-python/
.. _funcargparse: http://funcargparse.rtfd.io/
.. _model-organization: http://model-organization.readthedocs.io/en/latest/
.. _installation docs: http://iucm.readthedocs.io/en/latest/install.html
