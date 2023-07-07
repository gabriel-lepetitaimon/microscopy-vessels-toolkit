from typing import Iterable

import numpy as np


def label_1d(vector: np.ndarray):
    """
    Label connected components in a 1D vector.
    """
    upfront = np.asarray([-1, 1])
    upfronts = np.convolve(vector, upfront, mode="same") == 1
    return np.cumsum(np.concatenate([vector[:1], upfronts[:-1]])) * vector


def vessels_radius(segmentation, i: int | Iterable[int] = 10):
    """
    Count the number of vessels in a 2D segmentation.
    """

    if isinstance(i, int):
        i = [(segmentation.shape[0] // (i + 1)) * (_ + 1) for _ in range(i)]

    vessels = []
    for j in i:
        vessels_label = label_1d(segmentation[j, :]).astype(np.int64)
        vessels.append(np.bincount(vessels_label)[1:].tolist())

    return vessels
