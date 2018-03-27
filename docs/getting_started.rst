.. _getting_started:

Getting started
===============

The iucm package uses the model-organization_ framework and thus can be used
from the command line. The corresponding subclass of the
:class:`model_organization.ModelOrganizer` is the
:class:`iucm.main.IUCMOrganizer` class.

In this section, we provide a small starter example that transforms a
fictitious city by moving 125'000 inhabitants. Additional to the already
mentioned :ref:`requirements <requirements>`, this tutorial needs the
psy-simple_ plugin and the pyshp_ package.

After :ref:`having installed the package <install>` you can
setup a new project with the :ref:`iucm.setup` command via

.. ipython::

    @suppress
    In [1]: import os
       ...: os.environ['PYTHONWARNINGS'] = "ignore"

    In [1]: !iucm setup . -p my_first_project

To create a new experiment inside the project, use the :ref:`iucm.init`
command:

.. ipython::

    In [3]: !iucm -id my_first_experiment init -p my_first_project

Running the model, only requires a netCDF file with absolute population data.
The x-coordinates and y-coordinates must be in metres.

.. _pyshp: https://github.com/GeospatialPython/pyshp
.. _psy-simple: http://psy-simple.readthedocs.io/
.. _model-organization: http://model-organization.readthedocs.io/en/latest/

Transforming a city
-------------------

For the purpose of demonstration, we simply create a random input file with
2 city centers on a 25km x 25km grid at a resolution of 1km.

.. ipython::

    In [4]: import numpy as np
       ...: import xarray as xr
       ...: import matplotlib.pyplot as plt
       ...: import psyplot.project as psy
       ...: np.random.seed(1234)

    @suppress
    In [4]: _plotter = psy.plot.plot2d.plotter_cls()
       ...: _plotter.rc.trace = True
       ...: _plotter.rc['xlim'] = _plotter.rc['ylim'] = 'minmax'

    In [5]: sine_vals = np.sin(np.linspace(0, 2 * np.pi, 25)) * 5000
       ...: x2d, y2d = np.meshgrid(sine_vals, sine_vals)
       ...: data = np.abs(x2d + y2d) + np.random.randint(0, 7000, (25, 25))

    In [5]: population = xr.DataArray(
       ...:     data,
       ...:     name='population',
       ...:     dims=('x', 'y'),
       ...:     coords={'x': xr.Variable(('x', ), np.arange(25, dtype=float),
       ...:                              attrs={'units': 'km'}),
       ...:             'y': xr.Variable(('y', ), np.arange(25, dtype=float),
       ...:                              attrs={'units': 'km'})},
       ...:     attrs={'units': 'inhabitants', 'long_name': 'Population'})

    @savefig docs_getting_started.png width=100%
    In [6]: population.plot.pcolormesh(cmap='Reds');

    In [7]: population.to_netcdf('input.nc')

    @suppress
    In [7]: plt.close('all')

Now we create a new scenario where we transform the city by moving stepwise
500 inhabitants around. For this, we need a forcing file which we can create
using the :ref:`iucm.preproc.forcing` command:

.. ipython::

    In [8]: !iucm -v preproc forcing -steps 50 -trans 500

This now did create a new netCDF file with two variables

.. ipython::

    In [9]: xr.open_dataset(
       ...:     'my_first_project/experiments/my_first_experiment/input/forcing.nc')

that is also registered as forcing file in the experiment configuration

.. ipython::

    In [10]: !iucm info -nf

The change variable in this forcing file describes the number of people that
are moving within each step. In our case, this is just an alternating series of
500 and -500 since we take 500 inhabitants from one grid cell and move it to
another.


Having prepared this input file, we can run our experiment with the
:ref:`iucm.run` command:

.. ipython::

    In [9]: !iucm -id my_first_experiment configure -s run -i input.nc -t 50 -max 15000

The options here in detail:

-id my_first_experiment
    Tells iucm the experiment to use. The ``-id`` option is optional. If
    omitted, iucm uses the last created experiment.
configure -s
    This subcommand modifies the configuration to run our model in serial
    (see :ref:`iucm.configure`)
run
    The :ref:`iucm.run` command which tells iucm to run the experiment. The
    options here are

    -t 50
        Tells to model to make 50 steps
    -max 15000
        Tells the model that the maximum population is 15000 inhabitants per
        grid cell

The output now is a netCDF file with 50 steps:

.. ipython::

    In [10]: ds = xr.open_dataset(
       ....:     'my_first_project/experiments/my_first_experiment/'
       ....:     'outdata/my_first_experiment_1-50.nc')

    In [11]: ds

With the output for the population, energy consumption and other variables.
In the last step we also see, that the new population has mainly be added to
the city centers in order to minimize the transportation energy:

.. ipython::

    In [11]: fig = plt.figure(figsize=(14, 6))
       ....: fig.subplots_adjust(hspace=0.5)

    # plot the energy consumption
    In [12]: ds.cons.psy.plot.lineplot(
       ....:     ax=plt.subplot2grid((4, 2), (0, 0), 1, 2),
       ....:     ylabel='{desc}', xlabel='%(name)s');

    In [13]: ds.population[-1].psy.plot.plot2d(
       ....:     ax=plt.subplot2grid((4, 2), (1, 0), 3, 1),
       ....:     cmap='Reds', clabel='Population');

    @savefig docs_getting_started_final.png width=100%
    In [14]: (ds.population[-1] - population).psy.plot.plot2d(
       ....:     ax=plt.subplot2grid((4, 2), (1, 1), 3, 1),
       ....:     bounds='roundedsym', cmap='RdBu_r',
       ....:     clabel='Population difference');

    @suppress
    In [7]: plt.close('all')
       ...: ds.close()

As we can see, the model did move the population of sparse cells to locations
where the population is higher, mainly to decrease the average distance between
two individuals within the city.

The run settings are now stored in the configuration of the experiment, which
can be seen via the :ref:`iucm.info` command:

.. ipython::

    In [11]: !iucm info -nf

.. _probabilistic:

Probabilistic model
-------------------
The default IUCM settings use a purely deterministic methodology based on
the regression by [LeNechet2012]_. However, to take the errors of this model
into account, there exists a probabilistic version that is simply enabled via
the ``-prob`` (or ``--probabilistic``) argument, e.g. via

.. ipython::

    In [9]: !iucm run -nr -prob 1000 -t 50

Instead of simply moving population from one cell to another, it distributes
the population to multiple cells based on their probability to lower the
energy consumption for the city.

.. ipython::

    In [10]: ds = xr.open_dataset(
       ....:     'my_first_project/experiments/my_first_experiment/'
       ....:     'outdata/my_first_experiment_1-50.nc')

    @suppress
    In [11]: def plot_result():
       ....:     fig = plt.figure(figsize=(14, 6))
       ....:     fig.subplots_adjust(hspace=0.5)
       ....:     ax_cons = plt.subplot2grid((4, 2), (0, 0), 1, 2)
       ....:     pl1 = ds.cons.psy.plot.lineplot(
       ....:         ax=ax_cons, legendlabels='probabilistic',
       ....:         legend={'loc': 'lower center', 'ncol': 3,
       ....:                 'bbox_to_anchor': (0.5, 1.1)});
       ....:     rolling_cons = ds.cons.rolling(step=10, center=True)
       ....:     da = xr.concat([rolling_cons.mean(), rolling_cons.std()],
       ....:                    dim='variable')
       ....:     pl2 = da.psy.plot.lineplot(
       ....:         ax=ax_cons, legendlabels='10-steps running mean');
       ....:     pl3 = ds.cons_det.psy.plot.lineplot(
       ....:         ax=ax_cons, legendlabels='deterministic')
       ....:     pl1.share([pl2, pl3], keys=['legend', 'xlim', 'ylim'])
       ....:     pl1.update(ylabel='{desc}', xlabel='%(name)s')
       ....:     ds.population[-1].psy.plot.plot2d(
       ....:         ax=plt.subplot2grid((4, 2), (1, 0), 3, 1),
       ....:         cmap='Reds', clabel='Population');
       ....:     (ds.population[-1] - population).psy.plot.plot2d(
       ....:         ax=plt.subplot2grid((4, 2), (1, 1), 3, 1),
       ....:         bounds='roundedsym', cmap='RdBu_r',
       ....:         clabel='Population difference');

    @savefig docs_getting_started_final_prob.png width=100%
    In [15]: plot_result()

    @suppress
    In [7]: plt.close('all')
       ...: ds.close()

As we can see, the results are not as smooth as the deterministic results,
because now the energy consumption is based on a probabilistic set of
regression weights (see :func:`iucm.energy_consumption.random_weights`).
On the other hand, the deterministic energy consumption (stored as variable
`cons_det` in the output file) corresponds pretty much to the deterministic
version of our experiment setup above, as well as the running mean. And indeed,
if we would drastically increase the number of probabilistic scenarios, we
would approximate this energy consumption curve.

.. note::

    The energy consumption in the output file is for the probabilistic setting
    determined by the mean energy consumption for all random scenarios.
    The `cons_det` variable on the other hand is always determined by the
    weights in [LeNechet2012]_ (see
    :attr:`iucm.energy_consumption.weights_LeNechet`)


.. _masks:

Masking areas
-------------
Each city has several areas that should not be filled with population, such as
rivers, parks, etc. For example we assume a river, a lake and a forest in our
city (see :download:`the zipped shapefile <masking_shapes.zip>`)

.. ipython::

    In [12]: population.plot.pcolormesh(cmap='Reds');

    In [13]: from shapefile import Reader
       ....: reader = Reader('masking_shapes.shp')

    @savefig docs_getting_started_mask.png width=100%
    In [14]: from matplotlib.patches import Polygon
       ....: ax = plt.gca()
       ....: for shape_rec in reader.iterShapeRecords():
       ....:     color = 'forestgreen' if shape_rec.record[0] == 'Forest' else 'aqua'
       ....:     poly = Polygon(shape_rec.shape.points, facecolor=color, alpha=0.75)
       ....:     ax.add_patch(poly)

    @suppress
    In [14]: plt.close('all')

IUCM has three options, how to handle these areas:

ignore
    The cells and the population that are touched by these shapes are
    completely ignored
mask
    The cells are masked for keeping their population constant
max-pop
    The maximum population in the cells that are touched by the shapes is
    lowered by the fraction that the shape cover in each cell

All three methods can easily be applied using the :ref:`iucm.preproc.mask`
command.

.. note::

    Using this feature requires pyshp_ to be installed and the shapefile must
    be defined on the same coordinate system as the input data!

Ignoring the areas
******************
Ignoring the shapes will set the grid cells that are touched by the given
shapefiles to ``NaN``, i.e. `not a number`. Input cells that contain this
value are completely ignored in the simulation. For our shapefile and input
data here, the result would look like

.. ipython::

    In [15]: fig, axes = plt.subplots(1, 2)

    In [16]: plotter = population.psy.plot.plot2d(
       ....:     ax=axes[0], cmap='Reds', cbar='')

    In [17]: !iucm preproc mask masking_shapes.shp -m ignore

    @savefig docs_getting_started_mask_ignore.png width=100%
    In [18]: sp = psy.plot.plot2d('input.nc', name='population', ax=axes[1],
       ....:                      cmap='Reds', cbar='fb', miss_color='0.75')
       ....: sp.share(plotter, keys='bounds')

    @suppress
    In [18]: psy.close('all')
       ....: del sp, plotter

Masking the areas
*****************
Masking the areas means, that the population data in the grid cells that touch
the given cells is not changed but it is considered in the calculation of the
energy consumption. The input file for the model has a designated variable
named `mask` for that. The population data for non-zero grid cells in this
variable will be kept constant. In our case, the resulting `mask` variable in
looks like this

.. ipython::

    In [19]: !iucm preproc mask masking_shapes.shp -m mask

    @savefig docs_getting_started_mask_mask.png width=100%
    In [20]: sp = psy.plot.plot2d('input.nc', name='mask', cmap='Reds')

    @suppress
    In [20]: psy.close('all')
       ....: del sp


Adjusting the maximum population
********************************
This is the default method and is the best method represent the shape files in
the model. Instead of masking the data, we lower the amount of the maximum
possible population in the grid cells. For this, the shapefile is rasterized
at high resolution (by default 100-times the resolution of the input file) and
the we calculate the percentage for each coarse grid cell that is covered by
the shape. The result will then be stored in the `max_pop` variable in the
input dataset which defines the maximum population for each grid cell. In our
case, this variable looks like

.. ipython::

    @verbatim
    In [21]: !iucm preproc mask masking_shapes.shp

    @suppress
    In [21]: # we do not make the calculation here to speed up the build
       ....: xr.open_dataset('max_pop.nc').to_netcdf('input.nc', mode='a')

    @savefig docs_getting_started_mask_max_pop.png width=100%
    In [22]: sp = psy.plot.plot2d('input.nc', name='max_pop', cmap='Reds',
       ....:                      clabel='{desc}')

    @suppress
    In [23]: psy.close('all')
       ....: del sp

.. note::

    This method is a pure python implementation that does not have any other
    dependencies than matplotlib and pyshp. Due to this, it might be slow
    for large shapefiles or large input files. In this case, we recommend to
    use gdal_rasterize_ for creating the high resolution rastered shape file
    and gdalwarp_ for interpolating it to the input grid. In our case here,
    this would look like

    .. code-block:: bash

        gdal_rasterize -burn 1.0 -tr 0.01 0.01 masking_shapes.shp hr_rastered_shapes.tif
        gdalwarp -tr 1.0 1.0 -r average hr_rastered_shapes.tif covered_fraction.tif
        gdal_calc.py -A covered_fraction.tif --outfile=max_pop.nc --format=netCDF --calc="(1-A)*15000"

    And then merge the file ``'max_pop.nc'`` into ``'input.nc'`` as variable
    ``'max_pop'``.



    .. _gdal_rasterize: http://www.gdal.org/gdal_rasterize.html
    .. _gdalwarp: http://www.gdal.org/gdalwarp.html

.. ipython::
    :suppress:

    # delete the project and the input.nc
    In [10]: !rm input.nc
       ....: !iucm remove -p my_first_project -ay
