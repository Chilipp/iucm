# -*- coding: utf-8 -*-
"""Energy consumption module of the iucm package

This module, together with the `dist_numerator` package contains the necessary
functions for computing the energy consumption of a city. The main API function
is the :func:`energy_consumption` function. Note that the :data:`OWN` value is
set to the value for Stuttgart, Germany. You should change it if you want to
model another city. The parameters of this module come from [LeNechet2012]_.

References
----------
.. [LeNechet2012] Le Néchet, Florent. "Urban spatial structure, daily
   mobility and energy consumption: a study of 34 european cities."
   Cybergeo: European Journal of Geography (2012)."""
from __future__ import division
from scipy import stats
import numpy as np
from numpy.linalg import lstsq
from iucm.dist import dist
import iucm.utils as utils
from collections import namedtuple

docstrings = utils.docstrings


if np.__version__ < '1.14':
    rcond = -1
else:
    rcond = None


#: number of cars per 100 people. The default is 0, i.e. the value is ignored.
#: Another possible value that has been previously used might be 37.7, the
#: value for Stuttgart
OWN = 0

#: Intercept of energy consumption
K = -346.5

# ---- The weights for calculating the energy consumption ----
#: weight for car owner ship on energy consumption
wOWN = 17.36

#: weight for average distance of two individuals on energy consumption
wDIST = 279.0

#: weight for rank-size rule slope on energy consumption
wRSS = -9343.0

#: weight for entropy on energy consumption
wENTROP = 21700.0

EnVariables = utils.append_doc(
    namedtuple('EnVariables', ['k', 'dist', 'entrop', 'rss', 'own']),
    """A tuple containing the values that setup the energy consumption

    Parameters
    ----------
    k: float
        Intercept of energy consumption
    dist: float
        Average distance between two individuals
    entrop: float
        Entropy
    rss: float
        Rank-Size Slope
    own: float
        weight for car owner ship on energy consumption

    See Also
    --------
    iucm.model.Output
    iucm.model.Output2D
    iucm.model.PopulationModel.state
    iucm.model.PopulationModel.allocate_output
    """)


#: The weights for the variables to calculate the energy_consumption from
#: the multiple regression after [LeNechet2012]_
weights_LeNechet = EnVariables(k=K, dist=wDIST, rss=wRSS, entrop=wENTROP,
                               own=wOWN)

#: The standard errors of the weights used in [LeNechet2012]_ (obtained through
#: private communication via R. Cremades)
std_err_LeNechet = EnVariables(k=10500., own=4.696, dist=74.88, rss=2776.,
                               entrop=9172.)


def random_weights(weights):
    """Draw random weights

    This functions draws random weights and fills the arrays in the given
    `weights` with them. Weights are drawn using normal distributions defined
    through the :attr:`weights_LeNechet` and :attr:`std_err_LeNechet`.

    Parameters
    ----------
    weights: EnVariables
        The arrays to fill

    Notes
    -----
    `weights` are modified inplace!"""
    for arr, loc, scale in zip(weights, weights_LeNechet, std_err_LeNechet):
        arr[:] = np.random.normal(loc, scale, arr.shape)


def rss(population):
    """
    Compute the Rank-Size slope coefficient

    The rank-size coefficient :math:`a > 0` is calculated through a linear fit
    after [LeNechet2012]_ with

    .. math::

        \\log\\frac{p_k}{p_1} = -a \\log{k}

    where :math:`p_k` is the :population of the math:`k`-th ranking cell.

    Parameters
    ----------
    population: 1D np.ndarray
        The population data (must not contain 0!)

    Returns
    -------
    float
        The rank-size coefficient
    """
    k = stats.rankdata(population, method='dense')[:, np.newaxis]
    k[:, :] = k.max() - k[:, :] + 1
    k[:, :] = np.log(k)
    pmax = population.max()
    y_fit = np.log(population / pmax)
    ret = lstsq(k, y_fit, rcond=rcond)[0][0]
    return np.abs(ret)


def entrop(population, size):
    """
    Compute the entropy of a city

    Compute the entropy of a city after [LeNechet2012]_ via

    .. math::

        ENTROP = \\frac{
            \\sum_{i=1}^{size}\\frac{p_i}{P_{sum}} \\log\\frac{p_i}{P_{sum}}}{
            \log size}

    Parameters
    ----------
    population: 1D np.ndarray
        The population data (must not contain 0!)
    size: int
        The original size of the `population` data (including 0)

    Returns
    -------
    float
        The entropy value
    """
    inv_pop = 1 / population.sum()
    s = inv_pop / np.log10(size)
    return -(population * np.log10(population * inv_pop)).sum() * s


def energy_consumption(population, x, y, dist0=-1, slicer=None, indices=None,
                       increase=None, weights=weights_LeNechet):
    """
    Compute the energy consumption of a city

    Compute the energy consumption according to [LeNechet2012]_ via

    .. math::

        E = -346 + 17.4 \\cdot OWN + 279\\cdot DIST
                 - 9340\\cdot RSS + 21700\\cdot ENTROP

    Parameters
    ----------
    population: 1D np.ndarray
        The 1D population data
    x: 1D np.ndarray
        The x coordinates information for each cell in `population` in
        kilometers
    y: 1D np.ndarray
        The y coordinates information for each cell in `population` in
        kilometers
    dist0: float, optional
        The previous average distance between two individuals (see
        :func:`iucm.dist.dist` function). Speeds up the computation
        significantly
    slicer: :class:`slice` or boolean array, optional
        The slicer that can be use to access the changed cells specified by
        `increase`
    indices: 1D np.ndarray of dtype int, optional
        The indices corresponding to the increase in `increase`
    increase: 1D np.ndarray, optional
        The changed population which will be added on `population`. Specifying
        this and `dist0` speeds up the computation significantly instead of
        using the `population` alone. Note that you must then also specify
        `slicer` and `indices`
    weights: EnVariables
        The multiple regression coefficients (weights) for the calculating the
        energy consumption. If not given, the (0-dimensional) weights after
        [LeNechet2012]_ (:attr:`weights_LeNechet`, see above equation) are
        used.


    Returns
    -------
    np.ndarray of dtype float
        The energy consumption. The shape of the array depends on the given
        `weights`
    float
        The average distance between two individuals (DIST)
    float
        The entropy ENTROP
    float
        The rank-size-slope RSS

    See Also
    --------
    OWN
    entrop
    rss
    dist_numerator
    """
    if increase is not None:
        increased = population.copy()
        increased[slicer] += increase
    else:
        increased = population
    increased = increased[increased > 0]
    if x.size != population.size:
        # 1d-coordinates
        x, y = [a.astype(np.float64).ravel() for a in np.meshgrid(x, y)]
    elif x.ndim == 2:
        x = x.ravel()
        y = y.ravel()
    # round the DIST variable because otherwise you may get numerical
    # instabilities after multiple steps
    DIST = np.round(dist(population, x, y, dist0=dist0, indices=indices,
                         increase=increase), 5)
    ENTROP = entrop(increased, population.size)
    try:
        RSS = rss(increased)
    except RuntimeError:
        return np.inf, DIST, ENTROP, 0
    return (
        # energy consumption
        _calculate_en([DIST, ENTROP, RSS], weights, False),
        # other returns
        DIST, ENTROP, RSS)


def _calculate_en(dist_entrop_rss,
                  weights=np.array(weights_LeNechet)[:, np.newaxis],
                  squeeze=True):
    """Calculate the energy consumption

    Parameters
    ----------
    dist_entrop_rss: 1D np.ndarray of dtype float and shape (3, )
        DIST, ENTROP and RSS
    weights: 2D np.ndarray of shape (5, N)
        The weights to use (by default, :attr:`weights_LeNechet`). They must
        correspond to the attributes in the :class:`EnVariables` class
    squeeze: bool
        If True, squeeze the result

    Returns
    -------
    np.ndarray of shape (N, )
        The energy consumption. If ``N == 1 and squeeze``, this dimension is
        squeezed
    """
    weights = np.asarray(weights)
    if weights.ndim == 1:
        return weights[0] + weights[-1] * OWN + \
            np.sum(np.asarray(dist_entrop_rss) * weights[1:-1])
    else:
        ret = weights[0] + weights[-1] * OWN + \
            np.sum(np.asarray(dist_entrop_rss)[:, np.newaxis] * weights[1:-1],
                   axis=0)
    if squeeze:
        return np.squeeze(ret)
    else:
        return ret
