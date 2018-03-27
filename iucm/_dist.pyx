"""Module for calculating the average distance between individuals"""
import numpy as np
cimport numpy as np

def dist(np.ndarray[np.float64_t, ndim=1] population,
         np.ndarray[np.float64_t, ndim=1] x,
         np.ndarray[np.float64_t, ndim=1] y,
         double dist0=-1,
         np.ndarray[np.int64_t, ndim=1] indices=None,
         np.ndarray[np.float64_t, ndim=1] increase=None):
    """Calculate the average distance between two individuals

    This function calculates

    .. math::

        d = \\frac{\\sum_{i,j=1}^N d_{ij} P_i P_j}{P_{tot} (P_{tot}-1)}
    """
    cdef double ret = 0, pop_sum = 0, new = 0, increased_sum
    cdef long n, i, j
    cdef np.ndarray[np.float64_t, ndim=1] pop1d
    cdef np.ndarray dist = np.zeros(len(x), dtype=np.float64)
    cdef np.ndarray[np.complex128_t, ndim=1] xy
    xy = x + 1j*y
    n = population.size
    pop_sum = population.sum()
    if dist0 >= 0:
        increased_sum = pop_sum + increase.sum()
        ret = dist0 * (pop_sum * (pop_sum - 1)) / (increased_sum * (increased_sum - 1))
        for i in xrange(len(indices)):
            dist[:] = np.abs(xy[:] - xy[indices[i]])
            ret += increase[i] / (increased_sum * (increased_sum - 1)) * (
                2 * (population[:] * dist[:]).sum() +
                (dist[indices] * increase[:]).sum())
    else:
        for i in xrange(n):
            dist[:] = np.abs(xy[:] - xy[i])
            ret += (dist[:] * population[:] * population[i]).sum() / (pop_sum * (pop_sum - 1))
    return ret
