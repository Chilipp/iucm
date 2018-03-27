from __future__ import division
from collections import namedtuple, OrderedDict
import six
import numpy as np
import xarray as xr
import logging
from itertools import chain
import iucm.energy_consumption as en
from iucm.utils import docstrings
import iucm.utils as utils
import psyplot.data as psyd


logger = logging.getLogger(__name__)


def _nothing():
    return None, None


#: Meta information for the variables in the 1D-output of the
#: :class:`PopulationModel`
fields = OrderedDict([
    ('cons', {
        'units': 'MJ / inh',
        'long_name': 'Energy consumption',
        }),
    ('dist', {
        'units': 'km',
        'long_name': 'Average distance between two individuals',
        }),
    ('entrop', {
        'units': '-',
        'long_name': 'Entropy',
        }),
    ('rss', {
        'units': '-',
        'long_name': 'Rank-Size Slope',
        }),
    ('cons_det', {
        'units': 'MJ / inh',
        'long_name': 'Deterministic energy consumption',
        'comments': ('Energy consumption calculated based on based on '
                     'Le Nechet, 2012'),
        }),
    ('cons_std', {
        'units': 'MJ / inh',
        'long_name': 'std. deviation of Energy consumption',
        }),
    ('left_over', {
        'units': 'inhabitants',
        'long_name': 'Left over population that could not be subtracted',
        }),
    ('nscenarios', {
        'units': '-',
        'long_name': (
            'The number of scenarios that have been changed during this step'),
        }),
    ])

Output = utils.append_doc(
    namedtuple('Output', list(fields)),
    """The state of the model

    The :func:`collections.namedtuple` defining the state of the model during
    a single step. Each field of this class corresponds to one output variable
    in the output netCDF of the :class:`PopulationModel`. Meta information are
    taken from the :attr:`fields` dictionary

    Parameters
    ----------
    cons: float
        Energy consumption
    dist: float
        Average distance between two individuals
    entrop: float
        Entropy
    rss: float
        Rank-Size Slope
    cons_det: float
        The deterministic energy consumption based on
        :attr:`iucm.energy_consumption.weights_LeNechet`
    cons_std: float
        The standard deviation of the energy consumption
    left_over: float
        The left over inhabitants that could not be subtracted in the last step
    nscenarios: float
        The number of scenarios that have been changed during this step

    See Also
    --------
    Output2D
    PopulationModel.state
    PopulationModel.allocate_output
    """)


#: Meta information for the variables in the 2D-output of the
#: :class:`PopulationModel`
fields2D = fields.copy()
fields2D['scenarios'] = {'long_name': 'Scenario identifier'}


Output2D = utils.append_doc(
    namedtuple('Output2D', list(fields2D)),
    """The 2D state variables of the model

    2D output variables. Each variable corresponds to the same variable as in
    0-d :class:`Output` objects, but instead of saving only the best state of
    the model, this output saves all scenarios

    Parameters
    ----------
    cons: np.ndarray of dtype float
        Energy consumption
    dist: np.ndarray of dtype float
        Average distance between two individuals
    entrop: np.ndarray of dtype float
        Entropy
    rss: np.ndarray of dtype float
        Rank-Size Slope
    cons_det: np.ndarray of dtype float
        The deterministic energy consumption based on
        :attr:`iucm.energy_consumption.weights_LeNechet`
    cons_std: np.ndarray of dtype float
        The standard deviations within each probabilistic scenario
    left_over: float
        The left over inhabitants that could not be subtracted in the last step
    nscenarios: float
        The weight of each scenario that has been used during this step
    scenarios: np.ndarray of dtype int
        The number of the scenario

    See Also
    --------
    Output
    PopulationModel.state2d""")


class PopulationModel(object):
    """
    Class that runs and manages the population model

    This class represents one instance of the model for one experiment and is
    responsible for all the computation, parallelization and input-output
    coordination.
    The major important features of this class are

    :meth:`from_da` method
        A classmethod to construct the model from a :class:`xarray.DataArray`
    :meth:`step` method
        The method that is brings the model to the next step
    :attr:`init_step_methods` and :attr:`update_methods`
        The available update methods how to bring the model to the next change
    :attr:`selection_methods`
        The available selection methods that define the available scenarios
    :attr:`update_methods`
        The available update methods that define how the population change
        for the given selection is pursued
    :attr:`state` attribute
        The current state of the model. It's an instance of the :class:`Output`
        class containing all 1D-variables of the current :attr:`data` attribute

    Most of the other routines are related to input/output and parallelization
    """

    @property
    def dist(self):
        """Denumerator of average distance between two individuals"""
        return self.state.dist

    @property
    def consumption(self):
        """Energy consumption"""
        return self.state.cons

    @property
    def population_change(self):
        """The population change during this time step"""
        return self._change[self.total_step]

    @property
    def movement(self):
        """The population movement during this time step"""
        return self._movement[self.total_step]

    @property
    def total_change(self):
        """The total population change during this time step"""
        return self.movement + self.population_change + self.left_over

    @property
    def state(self):
        """The state as a namedtuple. You may set it with an iterable defined
        by the :attr:`Output` class"""
        return self._state

    @state.setter
    def state(self, value):
        self._state = Output(*value)

    @property
    def state2d(self):
        """The different values from the :attr:`state` of the model for each
        of the scenarios"""
        return self._state2d

    @state2d.setter
    def state2d(self, value):
        self._state2d = Output2D(*value)

    @property
    def state_dict(self):
        """The state as a dictionary. Mapping from state variable to the
        corresponding value. You may also set it with a dictionary"""
        return self._state._asdict()

    @state_dict.setter
    def state_dict(self, kwargs):
        self._state = Output(**kwargs)

    @property
    def state2d_dict(self):
        """The state as a dictionary. Mapping from state variable to the
        corresponding value. You may also set it with a dictionary"""
        return self._state2d._asdict()

    @state2d_dict.setter
    def state2d_dict(self, kwargs):
        self._state2d = Output(**kwargs)

    @property
    def left_over(self):
        """Left over population that could not have been subtracted for within
        the :meth:`PopulationModel.value_update` method and should be
        considered in the next step"""
        return self.state.left_over

    #: Left over population that could not have been subtracted for within the
    #: :meth:`PopulationModel.value_update` method
    _left_over = 0

    #: :class:`int`. The number of cells that are modified for one scenario
    ncells = 4

    #: The current step since the last output. -1 means, output has just been
    #: written or the model has been initialized
    current_step = -1

    #: The current step since the initialization of the model. -1 means, the
    #: model has been initialized
    total_step_run = -1

    #: The absolute current step in case the model has been restarted. If -1,
    #: the model has not yet been started
    total_step = -1

    #: np.ndarray. The current simulation data of the model
    data = None

    #: np.ndarray. The x-coordinates of the points in :attr:`data`
    x = None

    #: np.ndarray. The y-coordinates of the points in :attr:`data`
    y = None

    #: np.ndarray. The positions of the points in :attr:`data` that can be
    #: modified
    data2modify = None

    #: Attribute that is either False or the name of the output file where the
    #: model data just has been written to
    output_written = False

    #: The selection method from :attr:`selection_methods` we use to define the
    #: scenarios
    select = None

    #: The step initialization method from :attr:`init_step_methods` we use to
    #: define the scenarios
    init_step = None

    #: The update method from :attr:`update_methods` we use to compute that
    #: changes for each scenario
    update = None

    @docstrings.get_sectionsf('PopulationModel')
    def __init__(self, data, x, y, selection_method='consecutive',
                 update_method='categorical', ncells=4, categories=None,
                 state=None, forcing=None, probabilistic=0, max_pop=None,
                 use_pctls=False, last_step=0, data2modify=None):
        """
        Parameters
        ----------
        data: np.ndarray
            The 1D-data array (without NaN!) of the model
        x: np.ndarray
            The x-coordinates in *km* of each point in `data` (same shape as
            `data`)
        y: np.ndarray
            The y-coordinates in *km* of each point in `data` (same shape as
            `data`)
        selection_method: { 'consecutive' | 'random' }
            The avaiable selection scenarios (see :attr:`selection_methods`)
        update_method: { 'categorical' | 'random' | 'forced' }
            The avaiable update methods (see :attr:`update_methods`)
        ncells: int
            The number of cells that shall be modified for one scenario. The
            higher the number, the less computationally expensive is the
            computation
        categories: list of str
            The categories to use. If `update_method` is ``'categorical'``, it
            describes the categories and if `use_pctls` is True, it the
            each category is interpreted as a quantile in `data`
        state: list of float
            The current state of `data`. Must be a list corresponding to the
            :class:`Output` class
        forcing: xarray.Dataset
            The input dataset for the model containing variables with
            population evolution information. Possible variables in the netCDF
            file are *movement* containing the number of people to move and
            *change* containing the population change (positive or negative)
        probabilistic: int or tuple
            The number of probabilistic scenarios. For each scenario the energy
            consumption is calculated and the final population is distributed
            to the cells with the ideal energy consumption. Set this to 0 to
            only use the weights by [LeNechet2012]_. If tuple, then they are
            considered as the weights
        max_pop: np.ndarray
            A 1d-array with the maximum population for each cell in `data`. If
            None, the last value in `categories` will be used
        use_pctls: bool
            If True, values given in `categories` are interpreted as quantiles
        last_step: int
            If the model is restarted, the total number of already made steps
            (see :attr:`total_step` attribute)
        data2modify: np.ndarray
            The indices of points in `data` which are allowed to be modified.
            If None, all points are allowed to be modified

        See Also
        --------
        from_da: A more convenient initialization method using a xarray.Dataset
        """
        self.data = data
        self.x = x
        self.y = y
        if data2modify is not None:
            data2modify = data2modify.astype(np.int64)
        self.data2modify = data2modify

        self.selection_method = selection_method

        # set the maximum population
        max_pop = np.asarray(max_pop)
        if max_pop.ndim == 0:
            if not max_pop and categories is None:
                raise ValueError(
                    "The categories must be specified if max_pop is None!")
            elif not max_pop:  # max_pop is None
                max_pop = categories[-1]
            else:
                categories = np.append(categories, max_pop)
            self.max_pop = np.empty_like(data)
            self.max_pop[:] = max_pop
        else:
            assert max_pop.shape == data.shape, (
                "Shape of max_pop must %r equal the shape of the data %r" % (
                    max_pop.shape, data.shape))
            self.max_pop = max_pop

        if use_pctls:
            categories = np.percentile(
                self.data[~np.isnan(self.data) & (self.data > 0)],
                categories)
        self.categories = np.asarray(categories)
        try:
            self.select = self.selection_methods[selection_method]
        except KeyError:
            raise ValueError(
                "Unknown selection method %s! Possibilities are {%s}" % (
                    selection_method, ', '.join(self.selection_methods)))
        self.ncells = ncells
        self.update_method = update_method
        try:
            self.update = self.update_methods[update_method]
        except KeyError:
            raise ValueError(
                "Unknown update method %s! Possibilities are {%s}" % (
                    update_method, ', '.join(self.update_methods)))
        self.init_step = self.init_step_methods[update_method]
        if update_method == 'forced' and forcing is None:
            raise ValueError(
                "Need forcing information for forced update method! Please "
                "use the forcing keyword!")
        # determine the weights
        self.nprobabilistic = probabilistic
        if probabilistic:
            if update_method != 'forced':
                raise ValueError(
                    "Can only do probabilistic runs for a forced model! "
                    "Set probabilistic to 0 or update_method to 'forced'!")
            self.weights = en.EnVariables(
                *(np.zeros(probabilistic) for var in en.EnVariables._fields))
        else:  # use the weights from LeNechet, 2012
            self.weights = en.EnVariables(
                *np.reshape(en.weights_LeNechet,
                            (len(en.EnVariables._fields), 1)))
        # set the state
        if state is None:
            # calculate data
            logger.debug('Calculating state')
            en_info = en.energy_consumption(self.data, self.x, self.y,
                                            weights=self.weights)
            self.state = Output(
                en_info[0].mean(), *en_info[1:],
                cons_det=en._calculate_en(en_info[1:]),
                cons_std=en_info[0].std(), left_over=0, nscenarios=0)
            logger.debug('Done')
        else:
            self.state = state
        self.state2d = [np.zeros_like(self.data) for _ in range(
                            len(self.state) + 1)]
        for arr in self.state2d:
            arr[:] = np.nan
        self.forcing = forcing
        if forcing is not None:
            self._change = forcing.variables['change'].values
            self._movement = forcing.variables['movement'].values
        self.current_step = -1
        self.total_step = last_step - 1
        self.total_step_run = -1

    @docstrings.get_sectionsf('PopulationModel.initialize_model')
    @docstrings.dedent
    def initialize_model(self, da, dsi, ofiles, osteps, mask=None):
        """
        Initialize the model on the I/O processor

        Parameters
        ----------
        data: xr.DataArray
            The dataarray during the initialization
        dsi: xr.Dataset
            The base dataset of the `da`
        ofiles: list of str
            The name of the output files
        osteps: list of int
            Steps when to make the output
        mask: np.ndarray
            A boolean array that maps from the :attr:`data` attribute into the
            2D output data array
        """
        if isinstance(ofiles, six.string_types):
            ofiles = [ofiles]
        ofiles = iter(ofiles)
        try:
            osteps = iter(osteps)
        except TypeError:
            osteps = iter([osteps])
        self._ofiles = ofiles
        self._osteps = osteps
        self._next_ostep = next(self._osteps)
        if mask is None:
            mask = np.zeros(da.shape, dtype=bool)
        self.mask = mask
        self.allocate_output(da, steps=self._next_ostep, dsi=dsi)

    docstrings.delete_params('PopulationModel.parameters',
                             'data', 'x', 'y', 'data2modify')
    docstrings.delete_params('PopulationModel.initialize_model.parameters',
                             'mask')

    @classmethod
    @docstrings.dedent
    def from_da(cls, da, dsi, ofiles=None, osteps=None,
                coord_transform=1, **kwargs):
        """
        Construct the model from a :class:`psyplot.data.InteractiveArray`

        Parameters
        ----------
        %(PopulationModel.initialize_model.parameters.no_mask)s
        coord_transform: float
            The transformation factor to transform the coordinate values into
            kilometres

        Other Parameters
        ----------------
        %(PopulationModel.parameters.no_data|x|y|data2modify)s

        Returns
        -------
        PopulationModel
            The model created ready to use"""
        def get_val(name):
            var = dsi.variables[name]
            return var.values[() if var.ndim == 0 else -1]
        current = da if da.ndim < 3 else da[-1]
        data = current.values.ravel().copy()
        x = current.coords[current.dims[-1]].values
        y = current.coords[current.dims[-2 if current.ndim > 1 else -1]].values
        if x.shape != data.shape:
            x, y = [arr.ravel() for arr in np.meshgrid(x, y)]
        # apply the mask to filter out NaN values
        mask = ~np.isnan(data)
        data = data[mask].ravel().astype(np.float64)
        x = x[mask.ravel()]
        y = y[mask.ravel()]
        if not set(fields).difference(dsi.variables):
            state = map(get_val, fields)
        else:
            state = None

        # check for values that should not be modified
        if 'mask' in dsi.variables and 'data2modify' not in kwargs:
            modi_mask = dsi.mask.values.ravel()[mask].astype(bool)
            data2modify = np.arange(data.size, dtype=np.int64)[~modi_mask]
            kwargs['data2modify'] = data2modify
        if 'max_pop' in dsi.variables and not kwargs.get('max_pop'):
            kwargs['max_pop'] = dsi.variables['max_pop'].values.ravel()[
                mask.ravel()]

        ret = cls(data, x * coord_transform, y * coord_transform, state=state,
                  **kwargs)
        # initialize the model and allocate the output
        if ofiles is not None:
            ret.initialize_model(da, ofiles=ofiles, osteps=osteps, dsi=dsi,
                                 mask=mask.reshape(current.shape))
        return ret

    def best_scenario(self, all_slices, all_indices):
        """
        Compute the best scenario

        This method computes the best scenario for the given scenarios defined
        through the given `slices` and `indices`

        Parameters
        ----------
        slices: list of ``None``, :class:`slice` or boolean arrays
            The slicing objects for each scenario that allow us to create a
            view of the :attr:`data` attribute that we modify in place. If list
            of ``None``, it is computed using `indices`
        indices: list of list of :class`int`
            The numpy array containing the integer position in :attr:`data` of
            the cells modified for each scenario

        Returns
        -------
        1-dim np.ndarray of dtype float
            The consumptions of the best scenarios for each set of weights used
        2-dim np.ndarray of dtype float with shape ``(nprob, len(self.state))``
            The state of the best scenario for each probabilistic scenario
            which can be used for the :attr:`state` attribute.
        list of :class:`slice` or boolean array
            The slicing object from `slices` that corresponds to the best
            scenario and can be used to create a view on the :attr:`data`
        list of list of float
            The numbers of the modified cells for the best scenario
        list of 2d-np.ndarrays
            The 2d variables of the :attr:`state2d` attribute
        """
        self.current_step += 1
        self.total_step += 1
        self.total_step_run += 1
        data = self.data
        x = self.x
        y = self.y
        update = self.update
        to_slice = self._bool_or_slice
        dist = self.dist
        nprobabilistic = self.nprobabilistic or 1
        consumptions = np.empty((len(all_slices), nprobabilistic))
        consumptions.fill(np.inf)
        others = np.empty((len(all_slices), len(self.state) - 1))
        ret_cells = [[] for _ in range(len(all_slices))]
        state2d = self.state2d
        weights = self.weights
        i_left_over = state2d._fields.index('left_over') - 1
        i_cons_std = state2d._fields.index('cons_std') - 1
        i_cons_det = state2d._fields.index('cons_det') - 1
        for arr in state2d:
            arr[:] = np.nan
        # loop through the scenarios
        for i, (sl, ind) in enumerate(zip(all_slices, all_indices)):
            all_slices[i] = sl = to_slice(data, sl, ind)
            #: store the old values
            old = data[sl].copy()
            #: The cells that are modified for this scenario
            cell_values = update(old.copy(), sl)
            #: compute the energy consumption for this scenario if (if we have
            #: any changes to the initial state)
            if (old != cell_values).any():
                en_info = en.energy_consumption(
                    data, x, y, indices=ind, increase=cell_values - old,
                    dist0=dist, slicer=sl, weights=weights)
                consumptions[i] = en_info[0]
                others[i][:len(en_info) - 1] = en_info[1:]
                others[i][i_cons_det] = en._calculate_en(en_info[1:])
                others[i][i_cons_std] = en_info[0].std()
                others[i][i_left_over] = self._left_over
                ret_cells[i] = cell_values
                state2d[0][sl] = consumptions[i].mean()
                for j, val in enumerate(others[i], 1):
                    state2d[j][sl] = val
                state2d.scenarios[sl] = i
        # get the index of the scenario with the least energy consumption
        argmins = consumptions.argmin(axis=0)
        others[:, state2d._fields.index('nscenarios') - 1] = np.unique(
            argmins).size
        return (
            # the consumptions (1-d array)
            consumptions[argmins, np.arange(nprobabilistic)],
            # the other state variables
            others[argmins],
            # the slicers
            [all_slices[i] for i in argmins],
            # the cell values
            [ret_cells[i] for i in argmins],
            # the 2d state
            state2d)

    def step(self):
        """
        Bring the model to the next step and eventually write the output

        This method is the core of the entire :class:`PopulationModel` API
        connecting the necessary functions to compute the next best scenario.
        The general structure is

        1. initialize the step (see :attr:`init_step_methods`)
        2. define the scenarios (see :attr:`selection_methods`)
        3. choose the best scenario (see :attr:`best_scenario` and
           :attr:`update_methods`)
        4. write the output (see :meth:`write` method)

        Depending on whether the :meth:`start_processes` method has been called
        earlier, this is either done serial or in parallel"""
        self.output_written = False

        # ---- initialize the step
        current = self.data.copy()
        init_sl, indices = self.init_step()
        if init_sl is not None:
            en_info = en.energy_consumption(
                self.data, self.x, self.y, indices=indices,
                dist0=self.dist, slicer=init_sl,
                increase=self.data[init_sl] - current[init_sl],
                weights=self.weights)
            self.state = Output(en_info[0].mean(), *en_info[1:],
                                cons_det=en._calculate_en(en_info[1:]),
                                cons_std=en_info[0].std(),
                                left_over=self._left_over, nscenarios=0)
        nprobabilistic = self.nprobabilistic

        # ---- define the scenarios
        all_slices, all_indices = self.select()

        if nprobabilistic:
            en.random_weights(self.weights)
        else:
            nprobabilistic = 1

        if self.procs is None:
            # ---- get the best scenario in serial
            consumptions, others, sl, changed_values, self.state2d = \
                self.best_scenario(all_slices, all_indices)
            nscenarios = others[
                0, self.state2d._fields.index('nscenarios') - 1]
        else:
            # ---- get the best scenario in parallel
            self.current_step += 1
            self.total_step += 1
            self.total_step_run += 1
            for conn in self.parent_conns:
                conn.send(['update', init_sl,
                           self.data[init_sl] if init_sl is not None else None,
                           self.state, self.weights])
                conn.recv()
            nprocs = self.nprocs
            splitted_slices = np.array_split(all_slices, nprocs)
            splitted_indices = np.array_split(all_indices, nprocs)
            for i, (conn, slices, indices) in enumerate(zip(
                    self.parent_conns, splitted_slices, splitted_indices)):
                if len(slices):
                    conn.send(['best', slices, indices])
                else:
                    nprocs = i
                    break
            #: The states of the model of the best scenario for each process
            all_consumptions = np.zeros((nprocs, nprobabilistic))
            all_others = np.zeros((nprocs, nprobabilistic,
                                   len(self.state) - 1))
            #: The slicers for the best scenario for each process
            proc_slices = [[0] * nprobabilistic] * nprocs
            #: The values of the changed cells of the best scenario for each
            #: process
            all_changed = [
                [[] for _ in range(nprobabilistic)] for _ in self.procs]
            # ---- receive results
            state2d = self.state2d
            for arr in state2d:
                arr[:] = np.nan
            for i, conn in zip(range(nprocs), self.parent_conns):
                all_consumptions[i], all_others[i], proc_slices[i], \
                    all_changed[i], st2d = conn.recv()
                mask = ~np.isnan(st2d.scenarios)
                for j, arr in enumerate(st2d):
                    state2d[j][mask] = arr[mask]
                # add the number of previous scenarios to the one of the
                # current process `i` which doesn't see the other processes
                if i:
                    state2d.scenarios[mask] += sum(map(len,
                                                       splitted_indices[:i]))
            # choose the best scenario from all processes
            argmins = all_consumptions.argmin(axis=0)
            consumptions = all_consumptions[argmins, np.arange(nprobabilistic)]
            others = all_others[argmins]
            sl = [proc_slices[j][i] for i, j in enumerate(argmins)]
            changed_values = [all_changed[j][i] for i, j in enumerate(argmins)]
            nscenarios = np.unique(argmins).size
        if self.nprobabilistic >= 2:
            self.state, sl, changed_values = self.distribute_probabilistic(
                sl, nscenarios)
        else:
            self.state = chain(consumptions, np.squeeze(others))
            sl = sl[0]
            changed_values = changed_values[0]
            self.state2d.nscenarios[:] = 0.0
            self.state2d.nscenarios[sl] = 1.0
        if len(changed_values):
            self.data[sl] = changed_values
            # ---- send the changes to the new processes
            if self.procs is not None:
                for conn in self.parent_conns:
                    conn.send(['update', sl, changed_values, self.state])
                    conn.recv()
        # ---- store the current state and data of the model
        self.write()

    def distribute_probabilistic(self, slices, nscenarios):
        """
        Redistribute the population increase to the best scenarios

        This method distributes the population changes to the cells that have
        been computed as the best scenarios. It takes the input of the
        :meth:`best_scenario` method

        Parameters
        ----------
        slices: list
            The slicers of the best scenarios"""
        change_per_cell = self.total_change / len(slices)
        scenario_fract = 1. / len(slices)
        left_over = 0
        arr = self.data.copy()
        changed_cells = np.zeros_like(arr, dtype=np.bool)
        nscenarios_2d = self.state2d.nscenarios
        nscenarios_2d[:] = 0
        for sl in slices:
            arr[sl] = self.value_update(arr[sl], sl, change_per_cell)
            changed_cells[sl] = True
            left_over += self._left_over
            nscenarios_2d[sl] += scenario_fract
        if left_over:  # redistribute the left_over to the other cells
            for sl in slices:
                self.value_update(arr[sl], sl, left_over)
                left_over = self._left_over
                if not left_over:
                    break
        en_info = en.energy_consumption(
            arr, self.x, self.y, dist0=self.dist,
            increase=arr[changed_cells] - self.data[changed_cells],
            slicer=changed_cells,
            indices=np.arange(len(arr))[changed_cells], weights=self.weights)
        return (
            # the state
            Output(en_info[0].mean(), *en_info[1:],
                   cons_det=en._calculate_en(en_info[1:]),
                   cons_std=en_info[0].std(),
                   left_over=self._left_over, nscenarios=nscenarios),
            # the slicer
            changed_cells,
            # the changed cells
            arr[changed_cells])

    # -------------------------------------------------------------------------
    # --------------------------- First step part -----------------------------
    # ---------------------------- Initialization -----------------------------
    # -------------------------------------------------------------------------

    @property
    def init_step_methods(self):
        """Mapping from *update_method* name to the corresponding init function

        This property defines the init_step methods. Those methods are called
        at the beginning of each step on the main processor (I/O-processor).
        Each init_step method must accept no arguments and return a tuple with

        - an :attr:`slice` object or boolean array containing the information
          where the data changed
        - the indices of the cells in :attr:`data` that changed"""
        return {'categorical': _nothing,
                'random': _nothing,
                'forced': self.subtract_random}

    def subtract_random(self):
        """Subtract the people moving during this timestep"""
        movement = self.movement
        if movement == 0.0:
            return None, None
        data = self.data
        changed = 0.0
        sl = np.zeros_like(data, dtype=np.bool)
        for i in np.random.permutation(np.arange(data.size, dtype=int)):
            cell = data[i]
            if not np.isnan(cell):
                cell_change = cell * np.random.rand()
                data[i] -= cell_change
                sl[i] = True
                changed += cell_change
                if changed >= movement:
                    break
        return sl, np.arange(len(data), dtype=np.int64)[sl]

    # -------------------------------------------------------------------------
    # --------------------------- Second step part ----------------------------
    # ------------------------------- Selection -------------------------------
    # -------------------------------------------------------------------------

    @property
    def selection_methods(self):
        """Mapping from *selection_method* name to the corresponding function

        This property defines the selection methods. Those methods are called
        at the beginning of each step on the main processor (I/O-processor).
        Each selection_step method must accept no arguments and return a tuple
        with

        - an :attr:`slice` object or boolean array containing the information
          where the data changed. Alternatively it can be a list of ``None``
          and those slicing objects will be computed from the second argument
        - a 2D list of dtype integer containing the indices of the cells that
          for changed for the corresponding scenario"""
        return {'consecutive': self.consecutive_selection,
                'random': self.random_selection}

    def consecutive_selection(self):
        total_cells = self.data.size
        ncells = self.ncells
        if self.data2modify is None:
            slices = [slice(i, i + ncells) for i in range(
                0, self.data.size, ncells)]
            indices = [np.arange(i, min(i + ncells, total_cells))
                       for i in range(0, total_cells, ncells)]
        else:
            arr = self.data2modify
            indices = [
                arr[i:i + ncells] for i in range(0, arr.size, ncells)]
            slices = [None] * len(indices)
        return slices, indices

    def random_selection(self):
        if self.data2modify is None:
            indices = np.random.permutation(self.data.size)
        else:
            indices = np.random.permutation(self.data2modify)
        ncells = self.ncells
        indices = [
            indices[i:i + ncells] for i in range(0, indices.size, ncells)]
        return [None] * len(indices), indices

    # -------------------------------------------------------------------------
    # ---------------------------- Third step part ----------------------------
    # -------------------------------- Update ---------------------------------
    # -------------------------------------------------------------------------

    @property
    def update_methods(self):
        """Mapping from *update_method* name to the corresponding function

        This property defines the update methods. Each update method must
        accept and return a 1D numpy array of dtype float64 containing a view
        of the :attr:`data` attribute. The data must be modified in place!"""
        return {'categorical': self.categorical_update,
                'random': self.randomized_update,
                'forced': self.value_update}

    def value_update(self, cell_values, slicer, remaining=None):
        """Change the cells by using the forcing

        This update method changes the given cells based upon the
        :attr:`movement` information and the :attr:`population_change`
        information from the :attr:`forcing` dataset"""
        if remaining is None:
            remaining = self.total_change
        max_pop = self.max_pop[slicer]
        n = cell_values.size
        popsum = cell_values.sum()
        final_pop = max(popsum + remaining, 0)
        if remaining < 0 or final_pop <= max_pop.sum() * n:
            if n == 1:
                cell_values += remaining
                if cell_values[0] < 0:
                    remaining = cell_values[0]
                    cell_values[:] = 0
                else:
                    remaining = 0
            else:
                for i, cell in enumerate(cell_values):
                    # the maximum the current population inside the cell and
                    # otherwise the corresponding fraction at the popsum
                    change = max(-cell_values[i], min(
                        0 if popsum == 0 else (
                            remaining * (popsum - cell) / (
                                popsum * (n - 1))),
                        max_pop[i] - cell))
                    remaining -= change
                    cell_values[i] += change
        self._left_over = remaining
        return cell_values

    def categorical_update(self, cell_values, slicer):
        """Change the values through an update to the next category

        This method increases the population by updating the cells to the next
        (possible) category"""
        mask = ~np.isnan(cell_values)
        categories = self.categories
        max_pop = self.max_pop[slicer]
        indices = categories.searchsorted(cell_values[mask]) + 1
        cell_values[mask] += (
            categories[indices.clip(0, len(categories) - 1)] -
            categories[(indices-1).clip(0, len(categories) - 1)])
        cell_values[mask] = np.c_[cell_values[mask], max_pop[mask]].min(axis=1)
        return cell_values

    def randomized_update(self, cell_values, slicer):
        """Change the values through an update to a number within the next
        category

        This method increases the population by updating the cells to a random
        value within the next (possible) category"""
        def get_random(i):
            return np.random.randint(categories[i], categories[i + 1])
        categories = self.categories
        max_pop = self.max_pop[slicer]
        mask = ~np.isnan(cell_values)
        indices = categories.searchsorted(cell_values[mask]) + 1
        cell_values[mask] = list(map(get_random, indices.clip(
            0, len(categories) - 2)))
        cell_values[mask] = np.c_[cell_values[mask], max_pop[mask]].min(axis=1)
        return cell_values

    # -------------------------------------------------------------------------
    # ------------------------- Parallel Processing ---------------------------
    # -------- Necessary parts for letting the model run in parallel ----------
    # -------------------------------------------------------------------------

    #: :class:`multiprocessing.Process`. The processes of this model
    procs = None

    @property
    def nprocs(self):
        """:class:`int`. The number of processes started for this model"""
        return len(self.procs or [])

    def start_processes(self, nprocs):
        """Start `nprocs` processes for the model"""
        import multiprocessing as mp
        logger.debug('Start %i processes', nprocs)
        parent_conns = []
        child_conns = []
        for i in range(nprocs):
            parent_conn, child_conn = mp.Pipe()
            parent_conns.append(parent_conn)
            child_conns.append(child_conn)
        procs = [
            mp.Process(target=self, args=(conn, )) for conn in child_conns]
        for proc in procs:
            proc.daemon = True
            proc.start()
        self.procs = procs
        self.parent_conns = parent_conns

    def stop_processes(self):
        """Stop the processes for the model"""
        for conn in self.parent_conns:
            conn.send(None)

    def __call__(self, conn):
        logger.debug('Enter loop with %s and %s', self.consumption,
                     self.dist)
        received = True
        try:
            while received:
                received = conn.recv()
                if received:
                    try:
                        conn.send(self._step_methods[received[0]](
                            *received[1:]))
                    except Exception as e:
                        conn.close()
                        raise e
                else:
                    logger.debug('Stopping processes with %s and %s',
                                 self.consumption, self.dist)
        except EOFError:  # the connection has been closed
            pass

    @property
    def _step_methods(self):
        """The methods that are called during one :meth:`step` to synchronize
        and manage the different processes"""
        return {'best': self.best_scenario,
                'update': self.sync_state}

    def sync_state(self, sl, cell_values, state=None, weights=None):
        """
        Synchronize the :attr:`data` attribute between the processes

        This method is called to synchronize the states of the model in the
        different processes

        Parameters
        ----------
        sl: :class:`slice` or boolean np.ndarray
            The slicer that can be used to create a view on the :attr:`data`
        cell_values: np.ndarray of dtype float
            The values of the cells described by `sl`
        state: Output
            The state of the model
        """
        if sl is not None:
            self.data[sl] = cell_values
        if state is not None:
            self.state = state
        if weights is not None:
            self.weights = weights
        return

    def __reduce__(self):
        return self.__class__, (
            self.data, self.x, self.y, self.selection_method,
            self.update_model, self.ncells, self.categories,
            self.state, self.forcing, self.nprobabilistic, self.max_pop), dict(
                total_step=self.total_step, data2modify=self.data2modify,
                weights=self.weights)

    # -------------------------------------------------------------------------
    # ------------------------------- I/O Part --------------------------------
    # ----- Necessary parts for Input/Output (only called on main process) ----
    # -------------------------------------------------------------------------

    #: :class:`xarray.Dataset`. The output dataset
    _dso = None

    #: Flag that is True if the data was written to a file during the last
    #: step
    output_written = False

    #: :class:`int`. The step (corresponding to :attr:`current_step`) when the
    #: next output is written
    _next_ostep = None

    def allocate_output(self, da, dsi, steps):
        """Create the dataset for the output

        Parameters
        ----------
        da: psyplot.data.IneractiveArray
            The input data for the model
        dsi: xarray.Dataset
            The dataset `data` belongs to. If None, the
            :attr:`psyplot.data.InteractiveArray.base` attribute is used
        steps: int
            The number of steps
        """
        def empty_2d():
            ret = np.empty_like(odata)
            return ret
        logger.debug('Allocating new output dataset...')
        dso = da.to_dataset()[list(da.coords)]
        forcing = self.forcing
        if forcing is not None:
            self._tname = tname = forcing.movement.dims[0]
            current_step = self.total_step + 1
            time = forcing.coords[tname][current_step:current_step + steps]
            steps = len(time)
        else:
            tname = None
        if da.ndim == 2:
            odata = np.zeros([steps] + list(da.shape[-2:]))
            if tname is None:
                self._tname = tname = 'time'
                time = xr.Variable(('time', ), range(1, steps + 1))
            dims = [tname] + list(da.dims)
        else:
            odata = np.zeros([steps] + list(da.shape[-2:]))
            dims = da.dims
            if tname is None:
                last_time = dso[dims[0]][-1]
                self._tname = tname = last_time.name
                time = xr.concat([last_time + i for i in range(1, steps + 1)],
                                 dim=last_time.name)
            dso.drop(tname)
        dso[tname] = time
        dso[da.name] = vout = xr.Variable(dims, odata, attrs=da.attrs)

        # create other output fields
        vars2d = []
        for key, meta in fields.items():
            if key in dsi.variables:
                meta = dsi.variables[key].attrs
            dso[key] = xr.Variable((tname, ), np.zeros(steps), meta)
        for key, meta in fields2D.items():
            if key in dsi.variables:
                meta = dsi.variables[key].attrs
            vars2d.append(xr.Variable(dims, empty_2d(), meta))
            dso[key + '_2d'] = vars2d[-1]
        # create weights output field
        dso['weights'] = v_weights = xr.Variable(
            (tname, 'en_variables', 'probabilistic'),
            np.zeros((len(time), len(self.weights), self.nprobabilistic or 1)),
            {'long_name': 'Used energy consumption weights'})
        dso['en_variables'] = xr.Variable(
            ('en_variables', ), np.array(self.weights._fields),
            {'long_name': 'Variables to calculate the energy consumption'})

        self._dso = dso
        self._data_out = vout.values
        self._state2d_out = self.state2d.__class__(*[v.values for v in vars2d])
        self._weights_out = v_weights.values
        self._vname = da.name

    def write(self):
        """Write the current state to the output dataset"""
        dso = self._dso
        if dso is not None:
            i = self.current_step
            mask = self.mask
            self._data_out[i, mask] = self.data
            self._data_out[i, ~mask] = np.nan
            for key, val in six.iteritems(self.state_dict):
                dso[key][i] = val
            for j, arr in enumerate(self.state2d):
                self._state2d_out[j][i, mask] = arr
            self._weights_out[i][:, :] = self.weights
            if self._next_ostep is not None and i == self._next_ostep - 1:
                self.write_output()

    def write_output(self, complete=True):
        """Write the data to the next netCDF file

        Parameters
        ----------
        complete: bool
            If True, write the complete dataset, otherwise only until the
            current step"""
        ofile = next(self._ofiles)
        logger.info('Saving to %s...', ofile)
        if not complete:
            dso = self._dso.isel(**{
                self._tname: slice(0, self.current_step + 1)})
        else:
            dso = self._dso
        psyd.to_netcdf(dso, ofile)
        self._next_ostep = next(self._osteps)
        self.allocate_output(dso[self._vname], dsi=dso, steps=self._next_ostep)
        self.current_step = -1
        self.output_written = ofile

    docstrings.keep_params('PopulationModel.initialize_model.parameters',
                           'data', 'dsi')

    @classmethod
    @docstrings.dedent
    def get_input_ds(cls, data, dsi, **kwargs):
        """
        Return the input dataset which can be concatenated with the output

        Parameters
        ----------
        %(PopulationModel.initialize_model.parameters.data|dsi)s

        Returns
        -------
        xr.Dataset
            The modified `dsi`
        """
        # to make sure, it doesn't complain about the categories, we set a
        # value for max_pop
        kwargs['max_pop'] = kwargs.get('max_pop') or 15000
        model = cls.from_da(
            data, ofiles=[None], osteps=[2], dsi=dsi, **kwargs)
        model.current_step += 1
        model.write()
        tname = model._tname
        dsi = model._dso.isel(**{tname: slice(0, 1)}).copy()
        # estimate the first time step from the difference inside the model
        dsi[tname].values -= (
            model._dso[tname][1] - model._dso[tname][0]).values
        return dsi

    # -------------------------------------------------------------------------
    # ------------------------------ Miscallaneous ----------------------------
    # -------------------------------------------------------------------------

    @classmethod
    def _bool_or_slice(cls, arr, slicer, indices):
        """Transform the given indices to a slice object or boolean array

        Convenience method such that we do not have to give share large boolean
        arrays with other processes

        Parameters
        ----------
        arr: np.ndarray
            The data whose shape to use
        slicer: :class:`slice` object, boolean numpy.ndarray or None
            If None, a boolean array will be created using `indices and `data`.
            Otherwise, the given `slicer` is returned
        indices: np.ndarray of dtype int
            The indices of the cells that changed in `arr`

        Returns
        -------
        :class:`slice` object, boolean numpy.ndarray
            The `slicer` that can be used to create a view on `arr` for the
            given `indices`"""
        if slicer is None:
            arr_bool = np.zeros(arr.shape, dtype=bool)
            for i in indices:
                arr_bool[i] = True
            return arr_bool
        return slicer
