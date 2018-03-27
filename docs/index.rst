.. iucm documentation master file

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
minimising the energy required for transportation. This user manual
describes its technical implementation as the ``iucm`` python package.

Here we provide the principal steps on how to :ref:`install <install>` and
:ref:`use <getting_started>` the model, as well as a complete documentation of
the :ref:`python API <iucm-api-reference>` and the
:ref:`command line interface <Command.Line.API.Reference>`.

The scientific background will be published in a separate journal article.


Documentation
-------------

.. toctree::
    :maxdepth: 1

    install
    getting_started
    command_line/iucm.rst
    api/iucm.rst

License
-------
IUCM is published under license GPL-3.0 or any later version under the
copyright of Philipp S. Sommer and Roger Cremades, 2016

Acknowledgements
----------------
The authors thank Florent Le Néchet for his comments and for the provision of
further statistical details about his publications. The authors thank Walter
Sauf for his support on using the facilities of the German Supercomputing
Center (DKRZ). The authors also wish to express their gratitude to Wolfgang
Lucht, Hermann Held, Andreas Haensler, Diego Rybski and Jürgen P. Kropp for
their helpful comments. PS gratefully acknowledges funding from the Swiss
National Science Foundation ((ACACIA, CR10I2\_146314)). RC gratefully
acknowledges support from the Earth-Doc programme of the Earth League.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
