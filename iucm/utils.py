"""General utilities for the iucm package"""
import six
from itertools import chain
from docrep import DocstringProcessor

docstrings = DocstringProcessor()


docstrings.params['str_ranges.s_help'] = """
    A semicolon (``';'``) separated string. A single value in this string
    represents one number, ranges can also be used via a separation by
    comma (``','``). Hence, ``'2009;2012,2015'`` will be
    converted to ``[2009,2012, 2013, 2014]`` and ``2009;2012,2015,2`` to
    ``[2009, 2012, 2015]``"""


@docstrings.dedent
def str_ranges(s):
    """
    Convert a string of comma separated values to an iterable

    Parameters
    ----------
    s: str%(str_ranges.s_help)s

    Returns
    -------
    list
        The values in s converted to a list"""
    def get_numbers(s):
        nums = list(map(int, s.split(',')))
        if len(nums) == 1:
            return nums
        else:
            import numpy as np
            return np.arange(*nums)
    return list(chain(*map(get_numbers, s.split(';'))))


def append_doc(namedtuple_cls, doc):
    """Append a documentation to a namedtuple

    Parameters
    ----------
    namedtuple_cls: type
        The type that has been created with :func:`collections.namedtuple`
    doc: str
        The documentation docstring"""
    if six.PY3:
        namedtuple_cls.__doc__ += '\n' + doc
        return namedtuple_cls
    else:
        class DocNamedTuple(namedtuple_cls):
            __doc__ = namedtuple_cls.__doc__ + '\n' + doc
            __slots__ = ()
        DocNamedTuple.__name__ = namedtuple_cls.__name__
        return DocNamedTuple
