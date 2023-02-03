from dataclasses import dataclass

import numpy as np


@dataclass
class NeighborsOutput:
    """Output of the nearest neighbors function.

    Attributes
    ----------
    distances : np.ndarray
        Array of distances to the nearest neighbors.
    indices : np.ndarray
        Array of indices of the nearest neighbors.
    """

    indices: np.ndarray
    distances: np.ndarray
