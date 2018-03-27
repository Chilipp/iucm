"""Main module of the iucm package

This module defines the :class:`IUCMOrganizer` class that is used to create
a command line parser and to manage the configuration of the experiments
"""
from __future__ import print_function, division
import os
import copy
import os.path as osp
import six
import datetime as dt
from itertools import cycle
from argparse import Namespace
import logging
from model_organization import OrderedDict
from psyplot.config.rcsetup import safe_list
import iucm.utils as utils
from iucm.utils import docstrings
from model_organization import ModelOrganizer


class IUCMOrganizer(ModelOrganizer):
    """
    A class for organizing a model

    This class is indended to have hold the basic functions for organizing a
    model. You can subclass the functions ``setup, init`` to fit to your model.
    When using the model from the command line, you can also use the
    :meth:`setup_parser` method to create the argument parsers"""

    commands = ModelOrganizer.commands

    commands.insert(commands.index('archive'), 'preproc')
    commands.insert(commands.index('archive'), 'run')
    commands.insert(commands.index('archive'), 'postproc')

    paths = ModelOrganizer.paths + ['indir']

    name = 'iucm'

    # -------------------------------------------------------------------------
    # -------------------------- Preprocessing --------------------------------
    # -------------- Preprocessing functions for the experiment ---------------
    # -------------------------------------------------------------------------

    @property
    def preproc_funcs(self):
        """A mapping from preproc commands to the corresponding function"""
        return {'forcing': self.preproc_forcing,
                'mask': self.preproc_mask,
                }

    @docstrings.dedent
    def preproc(self, **kwargs):
        """
        Preprocess the data

        Parameters
        ----------
        ``**kwargs``
            Any keyword from the :attr:`preproc` attribute with kws for the
            corresponding function, or any keyword for the :meth:`main` method
        """
        funcs = self.preproc_funcs
        sp_kws = {key: kwargs.pop(key) for key in set(kwargs).intersection(
            funcs)}
        self.app_main(**kwargs)
        exp_config = self.fix_paths(self.exp_config)
        outdir = exp_config.setdefault('indir', osp.join(
            exp_config['expdir'], 'input'))
        if not osp.exists(outdir):
            os.makedirs(outdir)

        preproc_config = exp_config.setdefault('preproc', OrderedDict())

        for key, val in sp_kws.items():
            if isinstance(val, Namespace):
                val = vars(val)
            info = funcs[key](**val)
            if info:
                preproc_config[key] = info

    def _modify_preproc(self, parser):
        self._modify_app_main(parser)
        sps = parser.add_subparsers(title='Preprocessing tasks', chain=True)

        # forcing
        sp = sps.add_parser(
            'forcing',
            help='Create a forcing file from a predescribed population path')
        sp.setup_args(self.preproc_forcing)
        sp.update_arg('development_file', short='df')
        sp.update_arg('output', short='o')
        sp.update_arg('date_cols', short='t', metavar='col1,col2,col3,...',
                      type=lambda s: s.split(','))
        sp.update_arg('steps', type=int)
        sp.update_arg('movement', short='m', type=float)
        sp.update_arg('trans_size', short='trans', type=float)
        sp.update_arg('population_col', short='p')
        sp.update_arg('no_date_parsing', short='nd')

        # max_pop
        sp = parser.setup_subparser(self.preproc_mask, name='mask',
                                    return_parser=True)
        sp.update_arg('ifile', short='i')
        sp.update_arg('method', short='m')
        sp.unfinished_arguments['method']['help'] = (
            'Default: %(default)s. ' +
            sp.unfinished_arguments['method']['help'])
        sp.update_arg('hr_res', short='r')
        sp.update_arg('max_pop', short='max')
        sp.update_arg('vname', short='v')

    @docstrings.dedent
    def preproc_forcing(self, development_file=None, output=None,
                        date_cols=None, steps=1, movement=0, trans_size=0,
                        population_col=None, no_date_parsing=False):
        """
        Create a forcing file from a predescribed population path

        Parameters
        ----------
        development_file: str
            The path to a csv file containing (at least) one column with the
            projected population development
        output: str
            The name of the ouput forcing netCDF file. By default:
            ``'<expdir>/input/forcing.nc'``
        date_cols: list of str
            The names of the date columns in the `development_file` that shall
            be used to generate the date-time information. If not given, the
            date will simply be a range from 1 to `steps` times the length of
            the projected population development from `development_file`
        steps: int
            The numbers of model steps between on step of the projected
            development in `development_file`
        movement: float
            The people moving randomly during on model step
        trans_size: float
            The forced size of the transformation additionally to the
            development from `development_file`
        population_col: str
            The name of the column with population data. If not given, the last
            one is used
        no_date_parsing: bool
            If True, then `date_cols` is simply interpreted as an index and no
            date-time information is parsed
        """
        import numpy as np
        import pandas as pd
        import xarray as xr
        logger = self.logger
        logger.info('Creating forcing data...')

        idx_name = None
        if development_file is not None:
            kws = {}
            if not no_date_parsing and date_cols is not None:
                kws['parse_dates'] = {'time': date_cols}
                idx_name = 'time'
            elif date_cols is not None:
                idx_name = safe_list(date_cols)[0]
            df = pd.read_csv(development_file, **kws)
            if idx_name is not None:
                df.set_index(idx_name, inplace=True)
        else:
            df = pd.DataFrame(np.zeros((2, 1)),
                              columns=[population_col or 'population'])

        if population_col is None:
            population_col = df.columns[-1]
        steps_between = steps if not trans_size else steps * 2
        total_steps = steps_between * (len(df) - 1) + 1
        if idx_name:
            idx = np.zeros(total_steps)
            idx[:] = np.nan
            idx[::steps_between] = df.index.values
            idx = pd.Series(idx, name=idx_name)
            idx = pd.to_datetime(idx.interpolate())
        else:
            idx = None
        var = np.zeros(total_steps)
        var[:] = np.nan
        var[::steps_between] = df[population_col].values
        change = pd.Series(var, name='change', index=idx)
        change.interpolate(inplace=True)
        # only keep the population change
        change.values[1:] = change.values[1:] - change.values[:-1]
        # drop initial state
        change.drop(change.index.values[0], inplace=True)
        if trans_size:
            # only make population development every second ("add") step
            change.values[1::2] += change.values[:-1:2] + trans_size
            change.values[::2] = -trans_size
        forcing_df = change.to_frame()
        forcing_df['movement'] = movement
        if not forcing_df.index.name:
            forcing_df.index.name = 'step'
        ds = xr.Dataset.from_dataframe(forcing_df)
        ds.variables['change'].attrs.update(
            {'long_name': "population change", 'units': 'inhabitants'})
        ds.variables['movement'].attrs.update(
            {'long_name': "randomly moving population",
             'units': 'inhabitants'})
        if output is None:
            output = output = self.exp_config.get(
                'forcing', osp.join(self.exp_config['indir'], 'forcing.nc'))
        logger.debug('Saving output to %s', output)
        ds.to_netcdf(output)
        self.exp_config['forcing'] = output

        config = OrderedDict([
            ('development_file', development_file),
            ('development_steps', list(range(
                    0, total_steps, steps_between))[1:]),
            ('trans_size', trans_size),
            ('total_steps', total_steps - 1),
            ('movement', movement)])
        if logger.isEnabledFor(logging.DEBUG):
            for key, val in config.items():
                logger.debug('    %s: %s', key, val)

        return config

    @docstrings.dedent
    def preproc_mask(self, shapefile, method='max-pop', ifile=None, vname=None,
                     overwrite=False, max_pop=None, hr_res=100):
        """
        Mask grid cells based on a shape file

        This command calculates the maximum population for the model based on
        a masking shape file. The given shape file is rasterized at high
        resolution (by default, 100 times the resolution of the input file) and
        the fraction for each grid cell that is covered by that shape file is
        calculated.

        Parameters
        ----------
        shapefile: str
            The path to a shapefile
        method: {'max-pop', 'mask', 'ignore'}
            Determines how to handle the given shapes.

            max-pop
                The maximum population per grid cell is lowered by the fraction
                of the cell that is covered by the given shape. This will
                adjust the `max_pop` variable in the input file `ifile`
            mask
                The population of the grid cells in the input data that are
                touched by the given shapes will be kept constant during the
                simulation. This will adjust the `mask` variable in the input
                file `ifile`
            ignore
                The grid cells in the input data that are touched by the
                given grid cells are put to NaN and their population is not
                considered during the simulation. This will adjust the input
                population data (i.e. variable `vname`) directly
        ifile: str
            The path of the input file. If not specified, the value of the
            configuration is used
        vname: str
            The variable name to use. If not specified and only one variable
            exists in the dataset, this one is used. Otherwise, the
            ``'run.vname'`` key in the experiment configuration is used
        overwrite: bool
            If True and the target variable exists already in the input file
            `ifile` (and method is not 'ignore'), this variable is overwritten
        max_pop: float
            The maximum population. If not specified, the value in the
            configuration is used (only necessary if ``method=='max-pop'``)
        hr_res: int
            The resolution of the high resolution file, relative to the
            resolution of `ifile` (only necessary if ``method=='max-pop'``)

        Notes
        -----
        Note that the shapefile and the input file have to be defined on the
        same coordinate system! This function is not super efficient, for large
        data files we recommend using gdal_rasterize_ and gdalwarp_.

        .. _gdal_rasterize: http://www.gdal.org/gdal_rasterize.html
        .. _gdalwarp: http://www.gdal.org/gdalwarp.html"""
        from shapefile import Reader
        import matplotlib.path as mplp
        import psyplot.data as psyd
        import numpy as np
        import netCDF4 as nc
        exp_config = self.exp_config
        run_config = exp_config.setdefault('run', OrderedDict())
        if max_pop is None:
            max_pop = run_config.get('max_pop')
        if ifile is None:
            ifile = exp_config.get('input')
        if method == 'max-pop' and max_pop is None:
            raise ValueError(
                "The maximum population per grid cell has not yet been "
                "specified! See `iucm preproc mask --help`")
        if ifile is None:
            raise ValueError(
                "The input file needs to be specified! "
                "See `iucm preproc mask --help`")
        dsi = psyd.open_dataset(ifile, decode_times=False)
        vname = vname or run_config.get('vname') or self.get_population_vname(
            dsi)

        reader = Reader(shapefile)
        paths = list(mplp.Path(np.asarray(shape.points)[..., :2])
                     for shape in reader.shapes())

        x = dsi.psy[vname].psy.get_coord('x').values
        y = dsi.psy[vname].psy.get_coord('y').values
        dims = dsi.psy[vname].dims[-2:]
        ny, nx = dsi[vname].shape[-2:]

        dsi.close()
        del dsi

        if method == 'max-pop':
            hr_x = np.linspace(y[0], x[-1], len(x) * hr_res)
            hr_y = np.linspace(y[0], y[-1], len(y) * hr_res)

            X, Y = np.meshgrid(hr_x, hr_y)
        else:
            X, Y = np.meshgrid(x, y)

        points = np.array((X.flatten(), Y.flatten())).T
        del X, Y

        mask = np.zeros(points.shape[:-1], dtype=bool)
        for p in paths:
            mask |= p.contains_points(points)

        if method == 'max-pop':
            arr = mask.reshape(ny, hr_res, nx, -1).astype(float).mean(
                axis=(1, 3))
        else:
            mask = mask.reshape((ny, nx))

        nco = nc.Dataset(ifile, 'a')
        vnames = {'max-pop': 'max_pop', 'mask': 'mask', 'ignore': vname}
        target_vname = vnames[method]
        create = target_vname not in nco.variables
        if create:
            var = nco.createVariable(
                target_vname, int if method == 'mask' else float, dims)
            if method == 'max-pop':
                attrs = {'long_name': 'Maximum population',
                         'units': 'inhabitants'}
            else:
                attrs = {'long_name': 'Cells with constant population',
                         'comments': ('Cells with a non-zero value are not '
                                      'modified by the IUCM simulation.')}
            var.setncatts(attrs)
        else:
            var = nco.variables[target_vname]
        if method == 'max-pop' and (create or overwrite):
            var[:] = max_pop
        elif method == 'mask' and (create or overwrite):
            var[:] = 0
        if method == 'max-pop':
            var[:] *= (1 - arr)
        elif method == 'mask':
            var[:] = (mask | var[:].astype(bool)).astype(int)
        else:
            arr = var[:]
            arr[mask] = np.nan
            var[:] = arr

        nco.close()

    # -------------------------------------------------------------------------
    # ------------------------------- Run -------------------------------------
    # --------------------------- Run the experiment --------------------------
    # -------------------------------------------------------------------------

    @docstrings.dedent
    def run(self, ifile=None, forcing=None, vname=None, steps=50,
            selection_method=None, update_method=None, ncells=None,
            categories=None, use_pctls=False, no_restart=False,
            output_step=None, seed=None, stop_en_change=None,
            agg_stop_steps=100, agg_steps=1, probabilistic=None,
            max_pop=None, coord_transform=None, copy_from=None, **kwargs):
        """
        Run the model for the given experiment

        Parameters
        ----------
        ifile: str
            The input file. If not specified, the `input` key in the
            experiment configuration is used
        forcing: str
            The forcing file (necessary if ``update_method=='forced'``). If not
            specified, the `forcing` key in the experiment configuration is
            used
        vname: str
            The variable name to use. If not specified and only one variable
            exists in the dataset, this one is used. Otherwise, the
            ``'run.vname'`` key in the experiment configuration is used
        steps: int
            The number of time steps
        selection_method: { 'consecutive' | 'random' }
            The name of the method on how the data is selected.
            The default is consecutive.
            Possible selection methods are

            consecutive:
                Always the `ncells` consecutive cells are selected.
            random:
                `ncells` random cells in the field are updated.
        update_method: { 'categorical' | 'random' | 'forced' }
            The name of the update method on how the selected cells (see
            `selection_method` are updated). The default is categorical.
            Possible update methods are

            categorical:
                The selected cells are updated to the lower level of the next
                category.
            random:
                The selected cells are updated to a randum number within the
                next category.
            forced:
                A forcing file is used (see the `forcing` parameter).
        ncells: int
            The number of cells that shall be changed during 1 step. The
            default value is 4
        categories: list of floats
            The values for the categories to use within the models
        use_pctls: bool
            If True, interprete `categories` as percentiles instead of real
            population density
        no_restart: bool
            If True, and the run has already been conducted, restart it.
            Otherwise the previous run is continued
        output_step: int
            Make an output every `output_step`. If None, only the final result
            is written to the output file
        seed: int
            The random seed for numpy to use. Specify this parameter for the
            experiment to guarantee reproducability
        stop_en_change: float
            The minimum of required relative energy consumption change. If the
            mean relative energy consumption change over the last
            `agg_stop_steps` steps is below this number, the run is stopped
        agg_stop_steps: int
            The number of steps to aggregate over when calculating the mean
            relative energy consumption change. Does not have an effect if
            `stop_en_change` is None
        agg_steps: int
            Use only every `agg_steps` energy consumption for the aggregation
            when checking the `stop_en_change` criteria
        probabilistic: int
            The number of probabilistic scenarios. For each scenario the energy
            consumption is calculated and the final population is distributed
            to the cells with the ideal energy consumption. Set this to 0 to
            only use the weights by [LeNechet2012]_. If this option is None,
            the value will be taken from the configuration with a default of 0
            (i.e. no probabilistic run).
        max_pop: float
            The maximum population for each cell. If None, the last
            value in `categories` will be used or what is stored in the
            experiment configuration
        coord_transform: float
            The transformation factor to transform the coordinate values into
            kilometres
        copy_from: str
            If not None, copy the run settings from the other given experiment
        ``**kwargs``
            Any other keyword argument that is passed to the :meth:`main`
            method"""
        import numpy as np
        import psyplot.data as psyd
        from iucm.model import PopulationModel

        def get_files(imin, imax):
            if output_step is None:
                return [osp.join(outdir, exp + '_%i-%i.nc' % (imin, imax))]
            else:
                return [osp.join(outdir, exp + '_%i-%i.nc' % (i, min(
                            i + output_step - 1, imax)))
                        for i in range(imin, imax + 1, output_step)]

        def log_state():
            for key, val in six.iteritems(model.state._asdict()):
                logger.debug('    %s: %s', key, val)

        self.app_main(**kwargs)
        exp = self.experiment
        logger = self.logger
        debug = logger.isEnabledFor(logging.DEBUG)
        exp_config = self.fix_paths(self.exp_config)
        global_conf = self.config.global_config
        run_config = exp_config.setdefault('run', OrderedDict())
        if copy_from is not None:
            run_config.update(
                self.config.experiments[copy_from].get('run', {}))
            try:
                exp_config['forcing'] = self.config.experiments[copy_from][
                    'forcing']
            except KeyError:
                pass
        logger.debug('Run experiment...')

        seed = seed or exp_config['run'].get('seed')
        if seed is not None:
            np.random.seed(int(seed))
            run_config['seed'] = seed

        # ----------------------- Model Configuration -------------------------
        defaults = {'ncells': 4, 'max_pop': None,
                    'selection_method': 'consecutive',
                    'update_method': 'forced',
                    'probabilistic': 0,
                    'categories': None,
                    'coord_transform': 1.,
                    }
        run_settings = {'ncells': ncells,
                        'selection_method': selection_method,
                        'update_method': update_method,
                        'categories': categories,
                        'probabilistic': probabilistic,
                        'max_pop': max_pop,
                        'coord_transform': coord_transform,
                        }
        for key, val in run_settings.items():
            run_settings[key] = val = val if val is not None else \
                run_config.get(key, defaults[key])
            run_config[key] = val
        # get the actual steps value. The run_config will be updated below,
        # depending on whether we continue a run or not

        # ------------------------- output configuration ----------------------
        continued = 'run' in exp_config['timestamps']
        outdir = exp_config.setdefault('outdir', osp.join(
            exp_config['expdir'], 'outdata'))
        if not osp.exists(outdir):
            os.makedirs(outdir)
        if no_restart or not continued:
            for f in exp_config.get('outdata', []):
                if osp.exists(f):
                    os.remove(f)
            exp_config['outdata'] = []
            outfiles = get_files(1, steps)
            run_config['steps'] = 0
            run_config['step_date'] = OrderedDict()
            old_steps = 0
        elif continued:
            old_steps = run_config['steps']
            outfiles = get_files(old_steps + 1, old_steps + steps)
            run_config['steps'] += steps
            ifile = exp_config['outdata'][-1]
            run_config.setdefault('step_date', OrderedDict())
            if categories is None:  # percentiles have already been computed
                use_pctls = False
        else:
            raise ValueError("Cannot restart the experiment because no run "
                             "was ever invoked")
        logger.debug('    Output files: %s', outfiles)
        if output_step is None:
            output_step = steps
        outfiles = iter(outfiles)
        output_step = iter(cycle(safe_list(output_step)))

        # -------------------------- input file -------------------------------
        indir = exp_config.get('indir',
                               osp.join(exp_config['expdir'], 'input'))
        if not osp.exists(indir):
            os.makedirs(indir)
        if ifile is None:
            ifile = exp_config.get('input', self.project_config.get(
                'input',  self.global_config.get('input')))
        if ifile is None:
            raise ValueError("No input file specified!")
        if not continued:
            ifile_conf = osp.join(indir, 'input.nc')
            exists = osp.exists(ifile_conf)
            if exists and not osp.samefile(ifile_conf, ifile):
                os.remove(ifile_conf)
                os.symlink(osp.relpath(ifile, indir), ifile_conf)
            elif not exists:
                os.symlink(osp.relpath(ifile, indir), ifile_conf)
            exp_config['input'] = osp.abspath(ifile_conf)
        logger.debug('    Input file: %s', ifile)

        # -------------------------- forcing file -----------------------------
        if forcing is None:
            forcing = exp_config.get('forcing', self.project_config.get(
                'forcing',  self.global_config.get('forcing')))
        logger.debug('    Forcing file: %s', forcing)
        if forcing is not None:
            exp_config['forcing'] = osp.abspath(forcing)
            forcing = psyd.open_dataset(forcing)

        # --------------------------- Load data -------------------------------
        dsi = psyd.open_dataset(ifile)
        vname = vname or run_config.get('vname') or self.get_population_vname(
            dsi)
        try:
            data = dsi[vname]
        except KeyError:
            vnames = set(dsi.variables) - set(dsi.coords) - {'mask', 'max_pop'}
            if vname == run_config.get('vname'):
                msg = ('The variable %s as stored in the configuration is not '
                       'found in the dataset. Possible names are {%s}. Please '
                       'use the `vname` parameter!') % (
                           vname, ', '.join(vnames))
            else:
                msg = ('The variable %s is not found in the dataset. Possible '
                       'names are {%s}.') % (vname, ', '.join(vnames))
            raise KeyError(msg)
        run_config['vname'] = data.name

        if (coord_transform is None and
                data[data.dims[-1]].attrs.get('units', None) in [
                        'm', 'meter', 'metres', 'metre', 'meters']):
            run_config['coord_transform'] = run_settings['coord_transform'] = \
                coord_transform = 1000.

        logger.debug('    Input variable: %s, shape %s', vname, data.shape)

        logger.debug('Initialize model...')
        model = PopulationModel.from_da(
            data, forcing=forcing, ofiles=outfiles, osteps=output_step,
            use_pctls=use_pctls, dsi=dsi, last_step=old_steps, **run_settings)
        if model.categories is not None:
            run_config['categories'] = model.categories.tolist()
        if debug:
            logger.debug('Initial state:')
            log_state()

        # ------------------ Start parallel processes -------------------------
        if not global_conf.get('serial'):
            import multiprocessing as mp
            nprocs = global_conf.get('nprocs', 'all')
            if nprocs == 'all':
                nprocs = mp.cpu_count()
            model.start_processes(nprocs)
        logger.debug('Start main loop')
        t = [dt.datetime.now()] * (steps + 1)
        # --------------------------- Start loop ------------------------------
        last_output = None
        if stop_en_change is not None:
            consumptions = np.zeros(steps + 1)
            consumptions[0] = model.consumption
            # to use exactly the specified number of `agg_stop_steps` steps
            # when calculating the changes, we have to subtract 1
            agg_stop_steps -= 1
        for step in range(1, steps + 1):
            logger.debug('Step %i', step)
            model.step()
            if debug:
                log_state()
                t[step] = dt.datetime.now()
            if model.output_written and not self.no_modification:
                exp_config['outdata'].append(model.output_written)
                run_config['steps'] = old_steps + step
                if last_output is not None:
                    run_config['step_date'].pop(old_steps + last_output, None)
                last_output = step
                run_config['step_date'][old_steps + last_output] = str(t[step])
                exp_config['timestamps']['run'] = str(t[step])
                copied = copy.deepcopy(self.config)
                self.rel_paths(copied.experiments[self.experiment])
                copied.save()
            if stop_en_change is not None:
                consumptions[step] = model.consumption
                arr = consumptions[step - agg_stop_steps:step + 1:agg_steps]
                any_inf = np.isinf(arr).any()
                mean_change = np.abs((arr[1:] - arr[:-1]) / arr[:-1]).mean()
                if any_inf or mean_change <= stop_en_change:
                    logger.info("Stopping after %i steps with mean change %s",
                                step, mean_change)
                    break
        try:  # save the last step
            model.write_output(False)
        except StopIteration:  # last step already saved
            pass
        else:
            exp_config['outdata'].append(model.output_written)
            run_config['steps'] = old_steps + steps
        if debug:
            diffs = np.array([(t[i+1] - t[i]).seconds for i in range(steps)])
            logger.debug('Average step time: %s seconds' % diffs.mean())
            logger.debug('Fastest step time: %s seconds' % diffs.min())
            logger.debug('Slowest step time: %s seconds' % diffs.max())

        # --------------------------- Finish loop -----------------------------
        if not global_conf.get('serial'):
            model.stop_processes()

        logger.debug('Run Done.')

    def get_population_vname(self, ds):
        vnames = list(filter(
            lambda v: ds[v].ndim >= 2,
            set(ds.variables) - set(ds.coords) - {'mask', 'max_pop'}))
        if len(vnames) == 0:
            raise ValueError("Found no variables in the input dataset!")
        if len(vnames) > 1:
            raise ValueError(
                "Found %i possible variable names. Please specify one of "
                "{%s}" % (len(vnames), ', '.join(vnames)))
        return vnames[0]

    def _modify_run(self, parser):
        parser.update_arg('ifile', short='i')
        parser.update_arg('forcing', short='f')
        parser.update_arg('vname', short='v')
        parser.update_arg('steps', short='t', type=int)
        parser.update_arg('selection_method', short='sm')
        parser.update_arg('update_method', short='um')
        parser.update_arg('ncells', short='n', type=int)
        parser.update_arg('categories', short='c', metavar='float1,float2,...',
                          type=lambda s: list(map(int, s.split(','))))
        parser.pop_key('categories', 'nargs', None)
        parser.update_arg('no_restart', short='nr')
        parser.update_arg('output_step', short='ot', type=int)
        parser.update_arg('seed', type=int)
        parser.update_arg('use_pctls', short='pctls')
        parser.update_arg('probabilistic', short='prob')
        parser.update_arg('copy_from', short='cp')
        parser.update_arg('max_pop', short='max')
        parser.update_arg('coord_transform', short='ct')

    # -------------------------------------------------------------------------
    # -------------------------- Postprocessing -------------------------------
    # ------------ Postprocessing functions for the experiment ----------------
    # -------------------------------------------------------------------------

    @property
    def postproc_funcs(self):
        """A mapping from postproc commands to the corresponding function"""
        return OrderedDict([('rolling', self.rolling_mean),
                            ('map', self.make_map),
                            ('movie', self.make_movie),
                            ('evolution', self.plot_evolution),
                            ])

    @docstrings.dedent
    def postproc(self, no_input=False, **kwargs):
        """
        Postprocess the data

        Parameters
        ----------
        no_input: bool
            If True/set, the initial input file is ignored"""
        import xarray as xr
        funcs = self.postproc_funcs
        sp_kws = {key: kwargs.pop(key) for key in set(kwargs).intersection(
            funcs)}
        self.app_main(**kwargs)
        exp_config = self.fix_paths(self.exp_config)
        outdir = exp_config.setdefault('postprocdir', osp.join(
            exp_config['expdir'], 'postproc'))
        if not osp.exists(outdir):
            os.makedirs(outdir)
        files = exp_config['outdata']
        postproc_config = exp_config.setdefault('postproc', OrderedDict())

        ds = xr.open_mfdataset(files)

        if not no_input:
            from iucm.model import PopulationModel
            dsi = xr.open_dataset(exp_config['input'])
            vname = exp_config['run']['vname']
            forcing = exp_config.get('forcing')
            if forcing is not None:
                forcing = xr.open_dataset(forcing)
            dsi = PopulationModel.get_input_ds(
                dsi[vname], dsi, forcing=forcing)
            ds = xr.concat([dsi, ds], dim=dsi[vname].dims[0])

        for key, func in funcs.items():
            if key not in sp_kws:
                continue
            val = sp_kws[key]
            if isinstance(val, Namespace):
                val = vars(val)
                val.pop('no_input', None)
            info = func(ds, **val)
            if info:
                postproc_config[key] = info

    def _modify_postproc(self, parser):
        self._modify_app_main(parser)
        parser.update_arg('no_input', short='ni')
        sps = parser.add_subparsers(title='Postprocessing tasks', chain=True)

        # rolling mean
        sp = sps.add_parser(
            'rolling',
            help='Calculate the rolling mean for the energy consumption')
        sp.setup_args(self.rolling_mean)
        sp.pop_arg('ds')
        sp.update_arg('window', short='w')
        sp.update_arg('output', short='o')
        sp.update_arg('odir', short='od')

        # map
        sp = sps.add_parser(
            'map', help='Make a 2D-plot of the population')
        sp.setup_args(self.make_map)
        sp.pop_arg('ds')
        sp.pop_arg('close')
        sp.update_arg('output', short='o')
        sp.update_arg('odir', short='od')
        sp.update_arg('project_file', short='p')
        sp.update_arg('save_project', short='save')
        sp.update_arg('simple_plot', short='simple')
        sp.update_arg('time', short='t', type=int)
        sp.update_arg('t0', type=int)

        # movie
        sp = sps.add_parser(
            'movie', help='Make a movie of the population development')
        sp.setup_args(self.make_movie)
        sp.pop_arg('ds')
        sp.pop_arg('close')
        sp.update_arg('output', short='o')
        sp.update_arg('odir', short='od')
        sp.update_arg('project_file', short='p')
        sp.update_arg('save_project', short='save')
        sp.update_arg('simple_plot', short='simple')
        sp.update_arg('t0', type=int)
        sp.update_arg('time', short='t', type=utils.str_ranges,
                      help=docstrings.dedents("""
                          The time steps to use. %(str_ranges.s_help)s"""),
                      metavar='t1[;t2[;t31,t32,[t33]]]')

        sp.add_argument('-fps', help=(
            "The number of frames per second. Default: 10"), type=int)
        sp.add_argument('-dpi', help=("The dots per inch"), type=int)
        # evolution
        sp = sps.add_parser('evolution', help=(
            'Plot the evolution of energy consumption, DIST, RSS and ENTROP'))
        sp.setup_args(self.plot_evolution)
        sp.pop_arg('ds')
        sp.pop_arg('close')
        sp.update_arg('output', short='o')
        sp.update_arg('odir', short='od')
        sp.update_arg('time', short='t', type=utils.str_ranges,
                      help=docstrings.dedents("""
                          The time steps to use. %(str_ranges.s_help)s"""),
                      metavar='t1[;t2[;t31,t32,[t33]]]')
        sp.update_arg('use_rolling', short='roll')

    @docstrings.get_sectionsf('IUCMOrganizer.rolling_mean')
    @docstrings.dedent
    def rolling_mean(self, ds, window=None, output=None, odir=None):
        """
        Calculate the rolling mean for the energy consumption

        This postprocessing function calculates the rolling mean for the
        energy consumption

        Parameters
        ----------
        ds: xarray.Dataset
            The dataset with the *cons* and *cons_std* variables
        window: int
            Size of the moving window. This is the number of observations used
            for calculating the statistic. Each window will be a fixed size. If
            None, it will be taken from the experiment configuration with a
            default value of 50.
        output: str
            A filename where to save the output. If not given, it is not saved
            but may be later used by the :meth:`evolution` method
        odir: str
            The name of the output directory
        """
        if window is None:
            window = self.exp_config.get('postproc', {}).get(
                'rolling', {}).get('window', 50)
        self.logger.debug('Calculating rolling mean with window of %i...',
                          window)
        rolling = ds.cons.to_pandas().rolling(window)
        ds.variables['cons'][:] = rolling.mean()
        ds.variables['cons_std'][:] = rolling.std()
        if odir is None:
            odir = osp.join(self.exp_config['expdir'], 'postproc')
        if output:
            output = osp.join(odir, output)
            ds[['cons', 'cons_std']].to_netcdf(output)
            return {'output': output, 'window': window}
        return {'window': window}

    @docstrings.get_sectionsf('IUCMOrganizer.make_map')
    @docstrings.dedent
    def make_map(self, ds, output='map.pdf', odir=None, time=-1, diff=False,
                 t0=0, project_file=None, save_project=None, simple_plot=False,
                 close=True, **kwargs):
        """
        Make a movie of the post processing

        Parameters
        ----------
        ds: xarray.Dataset
            The dataset for the plot or a list of filenames
        output: str
            The name of the output file
        odir: str
            The name of the output directory
        time: int
            The timestep to plot. By default, the last timestep is used
        diff: bool
            If True/set, visualize the difference to the `t0` (by default, the
            first step) is used, instead of the entire data
        t0: int
            If `diff` is set, the reference step for the difference
        project_file: str
            The path to a filename containing a file that can be loaded via
            the :meth:`psyplot.project.Project.load_project` method
        save_project: str
            The path to a filename where to save the psyplot project
        simple_plot: bool
            If True/set, use a non-georeferenced plot. Otherwise, we use the
            cartopy module to plot it
        close: bool
            If True, close the project at the end

        Other Parameters
        ----------------
        ``**kwargs``
            Any other keyword that is passed to the
            :meth:`psyplot.project.Project.export` method"""
        import numpy as np
        import psyplot.project as psy
        from psyplot.data import _infer_interval_breaks

        self.logger.debug('Create plot of step %s...', time)

        config = self.exp_config['postproc'].get('map', OrderedDict())

        kwargs.update({key: config[key] for key in set(config) - {
            'plot_output', 'project_output'}})

        if odir is None:
            odir = osp.join(self.exp_config['expdir'], 'postproc')

        #: name of the variable in the above netCDF files
        vname = self.exp_config['run']['vname']

        if diff:
            ds = ds.copy(True)
            arr = ds[vname].values
            mask = np.zeros_like(arr, dtype=bool)
            diff = arr[time] - arr[0]
            mask[time] = ~np.isnan(arr[time]) & (diff == 0)
            arr[mask] = 0

        if project_file is not None:
            sp = psy.Project.load_project(project_file, datasets=[ds])
        else:
            #: categories of the experiment (will also be used as colorbar
            #: ticks)
            categories = np.array([0] + self.exp_config['run']['categories'])
            #: boundaries of the colormap
            bounds = _infer_interval_breaks(categories)
            bounds[0] = 0

            # ---- make the plot
            #: psyplot project of the plot
            if simple_plot:
                sp = psy.plot.plot2d(
                    ds, name=vname, mf_mode=True, t=time,
                    bounds=bounds, cmap='afmhot_r', cticks=categories,
                    title='step %(time)s', titleprops={'ha': 'left'})
            else:
                sp = psy.plot.mapplot(
                    ds, name=vname, mf_mode=True, t=time,
                    bounds=bounds, transform='moll', cmap='afmhot_r',
                    cticks=categories, projection='moll',
                    title='step %(time)s', titleprops={'ha': 'left'},
                    xgrid=False, ygrid=False)

        #: export the plot
        if output:
            output = osp.join(odir, output)
            self.logger.debug('Saving plot to %s', output)
            sp.export(output, **kwargs)
            kwargs['plot_output'] = output

        if save_project is None:
            save_project = config.get('project_output')
        if save_project:
            save_project = osp.join(odir, save_project)
            sp.save_project(save_project, dump=False,
                            paths=[self.exp_config['outdata']])
            kwargs['project_output'] = save_project

        if close:
            sp.close(True, True)
            return kwargs
        return kwargs, sp

    docstrings.delete_params('IUCMOrganizer.make_map.parameters', 'time')

    @docstrings.dedent
    def make_movie(self, ds, output='movie.gif', odir=None, diff=False,
                   t0=None, project_file=None, save_project=None,
                   simple_plot=False, close=True, time=None, **kwargs):
        """
        Make a movie of the post processing

        Parameters
        ----------
        %(IUCMOrganizer.make_map.parameters.no_time)s
        time: list of int
            The time steps to use for the movie

        Other Parameters
        ----------------
        ``**kwargs``
            Any other keyword for the
            :class:`matplotlib.animation.FuncAnimation` class that is used to
            make the plot"""
        import numpy as np
        from matplotlib.animation import FuncAnimation

        self.logger.debug('Making movie...')

        if odir is None:
            odir = osp.join(self.exp_config['expdir'], 'postproc')

        task_names = {True: 'diff', False: 'regular'}
        task_name = task_names[diff]

        config = self.exp_config['postproc'].get('movie', OrderedDict()).get(
            task_name, OrderedDict())

        defaults = {'fps': 10}

        for key in set(config) - {'plot_output', 'project_output'}:
            if kwargs[key] is None:
                kwargs.setdefault(key, config[key])

        # update with defaults for the movie writer
        for key, val in defaults.items():
            if kwargs.get(key) is None:
                kwargs[key] = val

        #: name of the variable in the above netCDF files
        vname = self.exp_config['run']['vname']
        tname = ds[vname].dims[0]

        if diff and t0 is not None:
            ref = ds.isel(**{tname: t0})[vname].values

        orig_time = range(ds[tname].size)

        if time is not None:
            if diff and t0 is None:
                ref = ds.isel(**{tname: safe_list(time)[0]})[
                    vname].values
                t0 = 0
            if isinstance(time, slice):
                orig_time = time.indices(ds[tname].size)
            else:
                orig_time = time[:]
            ds = ds.isel(**{tname: time})
        elif diff and t0 is None:
            ref = ds.isel(**{tname: 0})[vname].values
            t0 = 0

        time = range(ds[tname].size)

        if diff:
            ds = ds.copy(True)
            arr = ds[vname].values
            for i, orig in zip(range(0, ds[tname].size), orig_time):
                if orig != t0:
                    mask = np.zeros_like(arr, dtype=bool)
                    diffs = arr[i] - ref
                    mask[i] = ~np.isnan(arr[i]) & (diffs == 0)
                    arr[mask] = 0

        _, sp = self.make_map(
            ds, odir=odir, diff=False, output=None, project_file=project_file,
            save_project=False, simple_plot=simple_plot, close=False,
            time=0)

        def update(t):
            """Update function for the movie"""
            sp.update(t=t)

        #: animation which create the movie
        ani = FuncAnimation(
            next(iter(sp.figs)), update, time, sp.draw)
        #: save the movie to a gif file
        if output:
            output = osp.join(odir, output)
            self.logger.debug('Saving movie to %s', output)
            self.logger.debug('Movie config: %s', kwargs)
            ani.save(output, **kwargs)
            kwargs['plot_output'] = output

        save_project = save_project or config.get('project_output')
        if save_project:
            save_project = osp.join(odir, save_project)
            sp.save_project(save_project, dump=False,
                            paths=[self.exp_config['outdata']])
            kwargs['project_output'] = save_project

        info = {task_name: kwargs}
        if self.exp_config['postproc'].get('movie', OrderedDict()).get(
                task_names[not diff], OrderedDict()):
            info[task_names[not diff]] = self.exp_config['postproc']['movie'][
                task_names[not diff]]

        if close:
            sp.close(True, True)
            return info

        return info, sp

    docstrings.keep_params('IUCMOrganizer.make_map.parameters', 'ds',
                           'output', 'odir')

    @docstrings.dedent
    def plot_evolution(self, ds, output='plots.pdf', odir=None, close=True,
                       time=None, use_rolling=False):
        """
        Plot the evolution of DIST, RSS, ENTROP and Energy consumption

        Parameters
        ----------
        %(IUCMOrganizer.make_map.parameters.ds|output|odir)s
        close: bool
            If True, the created figures are closed in the end
        time: list of int
            The time steps to use for the movie
        use_rolling: bool
            If True, use the rolling mean for the energy consumption

        Returns
        -------
        dict
            Information on the output"""
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.backends.backend_pdf import PdfPages
        import iucm.energy_consumption as en

        self.logger.debug('Plot timeseries...')

        if odir is None:
            odir = osp.join(self.exp_config['expdir'], 'postproc')

        if use_rolling:
            import xarray as xr
            if 'output' in self.exp_config.get('postproc', {}).get(
                    'rolling', {}):
                with xr.open_dataset(self.exp_config['postproc'][
                        'rolling']['output']) as ds2:
                    ds.variables['cons'] = ds2.variables['cons']
                    ds.variables['cons_std'] = ds2.variables['cons_std']
            else:
                self.rolling_mean(ds)

        if time is not None:
            ds = ds.isel(**{ds.dist.dims[0]: time})

        ds_plot = ds[['cons', 'rss', 'dist', 'entrop', 'cons_std']].copy(True)
        ds_plot2 = ds_plot.copy(True)

        # ----- area plot
        ds_plot['entrop'].values *= en.wENTROP
        ds_plot['rss'].values *= en.wRSS
        ds_plot['dist'].values *= en.wDIST
        summed = np.abs(ds_plot.dist) + np.abs(ds_plot.rss) + np.abs(
            ds_plot.entrop)
        for vname in ['dist', 'rss', 'entrop']:
            ds_plot[vname].values /= summed

        #: List of matplotlib figures created in this function
        figs = []

        figs.append(plt.figure())

        plt.stackplot(
            ds_plot.time.values, ds_plot.dist.values, ds_plot.entrop.values,
            labels=[ds.dist.long_name, ds.entrop.long_name], baseline='zero')
        plt.stackplot(ds_plot.time.values, ds_plot.rss.values,
                      labels=[ds.rss.long_name])

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.2)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

        # ----- single line plot for each variable
        # --- population
        figs.append(plt.figure())
        pop_da = ds[self.exp_config['run']['vname']]
        pop_da.sum(axis=(1, 2)).plot()
        plt.ylabel('Population [inhabitants]')
        # --- energy consumption
        units = ds.cons.units
        figs.append(plt.figure())
        ds_plot.cons.plot()
        plt.fill_between(ds_plot[ds_plot.cons.dims[0]],
                         ds_plot.cons - ds_plot.cons_std,
                         ds_plot.cons + ds_plot.cons_std,
                         facecolors=plt.gca().lines[0].get_c(),
                         edgecolors='None',
                         alpha=0.5)
        plt.ylabel('%s [%s]' % (ds.cons.long_name, units))

        ds_plot2['dist'].values *= en.wDIST
        ds_plot2['rss'].values *= en.wRSS
        ds_plot2['entrop'].values *= en.wENTROP
        for name in ['dist', 'rss', 'entrop']:
            figs.append(plt.figure())
            ds_plot2[name].plot()
            plt.ylabel('weighted %s [%s]' % (ds_plot2[name].long_name, units))

        ret = {}

        if output:
            output = osp.join(odir, output)
            self.logger.debug('Saving plots to %s', output)
            with PdfPages(output) as pdf:
                for fig in figs:
                    pdf.savefig(fig, bbox_inches='tight')
            ret['plot_output'] = output
        if close:
            for fig in figs:
                plt.close(fig.number)
        return ret


def _get_parser():
    """Function returning the iucm parser, necessary for sphinx documentation
    """
    return IUCMOrganizer.get_parser()


def main():
    """Call the :meth:`~model_organization.ModelOrganizer.main` method of the
    :class:`IUCMOrganizer` class"""
    return IUCMOrganizer.main()


if __name__ == '__main__':
    main()
