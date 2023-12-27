from dataclasses import dataclass
from functools import cached_property

import chex
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors
from umap.umap_ import fuzzy_simplicial_set


@dataclass
class NeighborsResults:
    """Nearest neighbors results data store.

    Attributes
    ----------
    distances : np.ndarray
        Array of distances to the nearest neighbors.
    indices : np.ndarray
        Array of indices of the nearest neighbors. Self should always
        be included here.
    """

    indices: np.ndarray
    distances: np.ndarray

    def __post_init__(self):
        chex.assert_equal_shape([self.indices, self.distances])
        # Guarantee that the first neighbor is self.
        # This is a simple and lightweight way to ensure this.
        nn_obj = NearestNeighbors(n_neighbors=self.n_neighbors, metric="precomputed").fit(self.knn_graph_distances)
        self.distances, self.indices = nn_obj.kneighbors(self.knn_graph_distances)

    @property
    def n_samples(self):
        return self.indices.shape[0]

    @property
    def n_neighbors(self):
        return self.indices.shape[1]

    @cached_property
    def knn_graph_distances(self) -> csr_matrix:
        """Return the sparse weighted adjacency matrix."""
        n_samples, n_neighbors = self.indices.shape
        # Efficient creation of row pointer
        rowptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
        # Create CSR matrix
        return csr_matrix((self.distances.ravel(), self.indices.ravel(), rowptr), shape=(n_samples, n_samples))

    @cached_property
    def knn_graph_connectivities(self) -> csr_matrix:
        """Compute connectivities using the UMAP approach.

        This function computes connectivities (similarities) from distances
        using the approach from the UMAP method, which is also used by scanpy.
        """
        conn_graph = coo_matrix(([], ([], [])), shape=(self.n_samples, 1))
        connectivities = fuzzy_simplicial_set(
            conn_graph,
            n_neighbors=self.n_neighbors,
            random_state=None,
            metric=None,
            knn_indices=self.indices,
            knn_dists=self.distances,
            set_op_mix_ratio=1.0,
            local_connectivity=1.0,
        )
        return connectivities[0]

    def subset_neighbors(self, n: int) -> "NeighborsResults":
        if n > self.n_neighbors:
            raise ValueError("n must be smaller than the number of neighbors")
        return self.__class__(self.distances[:, :n], self.indices[:, :n])
