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
      - |version| |conda| |supported-versions| |supported-implementations| |zenodo|

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

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1982564.svg
   :alt: Zenodo
   :target: https://doi.org/10.5281/zenodo.1982564

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

How to cite IUCm
-------------------

When using the IUCm, you should at least cite the publication in
`the Journal of Geoscientific Model Developments`_:

Cremades, R. and Sommer, P. S.: Computing climate-smart urban land use with the
Integrated Urban Complexity model (IUCm 1.0), *Geosci. Model Dev.*, 12, 525-539,
https://doi.org/10.5194/gmd-12-525-2019, 2019.

:download:`BibTex <iucm_entry.bib>` - :download:`EndNote <iucm_entry.enw>`

Furthermore, each release of iucm is
associated with a DOI using zenodo.org_.

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1982565.svg
    :alt: zenodo
    :target: https://doi.org/10.5281/zenodo.1982565

If you want to cite a specific version, please refer to the `releases page of iucm`_.


.. _the Journal of Geoscientific Model Developments: https://www.geoscientific-model-development.net/index.html
.. _zenodo.org: https://zenodo.org/
.. _releases page of iucm: https://github.com/Chilipp/iucm/releases/
