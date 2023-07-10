from enum import EnumMeta

import numpy as np
from strenum import StrEnum as StrEnumBase


class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        return item in tuple(cls)


class StrEnum(StrEnumBase, metaclass=MetaEnum):
    ...


def invert_lookup(lookup: np.ndarray):
    lookup = np.asarray(lookup)
    assert lookup.ndim == 1, "Lookup must be 1D"

    inv_lookup = np.zeros(lookup.max() + 1, dtype=lookup.dtype)
    inv_lookup[lookup] = np.arange(lookup.size)
    return inv_lookup
