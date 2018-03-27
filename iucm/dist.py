"""Module for calculating the average distance between two individuals"""
from iucm._dist import dist as _dist


def dist(population, x, y, dist0=-1, indices=None, increase=None):
    """Calculate the average distance between two individuals

    This function calculates

    .. math::

        d = \\frac{\\sum_{i,j=1}^N d_{ij} P_i P_j}{P_{tot} (P_{tot}-1)}

    after [LeNechet2012]_, where :math:`P_i` is the population of the i-th grid
    cell, :math:`P_{tot}` is the total population and :math:`d_{ij}` is the
    distance between the i-th and j-th grid cell

    Parameters
    ----------
    population: 1D np.ndarray of dtype float
        The population data for each grid cell. If ``dist0>=0`` then this
        array must hold the population of the previous step corresponding to
        `dist0`. Otherwise it should be the real population.
    x: 1D np.ndarray of dtype float
        The x-coordinates for each element in `population`
    y: 1D np.ndarray of dtype float
        The y-coordinates for each element in `population`
    dist0: float, optional
        The previous average distance between individuals. If this is given,
        `increase` and `indices` must not be None and `population` must
        represent the population of the previous step. Specifying this value,
        significantly speeds up the calculation.
    indices: 1D np.ndarray of dtype int
        The indices of the grid cells, where the population has been increased.
    increase: 1D np.ndarray of dtype float
        The increase in the grid cells corresponding to the given `indices`

    Returns
    -------
    float
        The average distance between two individuals.
    """
    return _dist(population, x, y, dist0, indices, increase)
