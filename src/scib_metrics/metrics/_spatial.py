"""Spatial transcriptomics metrics for scib-metrics.

**Embedding-based** metrics (``spatial_mrre``, ``spatial_knn_overlap``,
``spatial_distance_correlation``, ``spatial_morans_i``) compare the latent
embedding produced by a spatial model against the physical coordinates.  They
vary across embeddings and directly quantify how well each model preserves
local and global spatial structure in its latent space.

All functions return a float in **[0, 1]** where **higher is always better**.

References
----------
Hu et al. (2024) Benchmarking clustering, alignment, and integration
    methods for spatial transcriptomics. PMC11312151.
Chen et al. (2025) A comprehensive benchmarking for spatially resolved
    transcriptomics clustering methods. PMC12747554.
MuST (2023) Multi-modal spatial transcriptomics benchmark.
"""

import warnings

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.neighbors import NearestNeighbors


def spatial_mrre(
    X_embedding: np.ndarray,
    spatial_coords: np.ndarray,
    k: int = 15,
    max_cells: int = 2000,
    seed: int = 42,
) -> float:
    """Mean Relative Rank Error (MRRE), normalised to [0, 1].

    For each spot, finds its ``k`` nearest spatial neighbours and compares
    their rank ordering in spatial space to their rank ordering in the latent
    embedding.  The mean absolute rank difference, normalised by ``k``,
    measures how much the local geometry is distorted.  The score is
    ``1 - MRRE/k`` so that perfect rank preservation yields 1.

    Parameters
    ----------
    X_embedding
        Array of shape ``(n_spots, n_dims)`` — latent representation.
    spatial_coords
        Array of shape ``(n_spots, 2)`` with spatial coordinates (x, y).
    k
        Neighbourhood size. Default ``15``.
    max_cells
        Subsample to this many cells before computation (O(n·k) cost).
        Default ``2000``.
    seed
        Random seed for subsampling. Default ``42``.

    Returns
    -------
    float
        Score in ``[0, 1]``. **Higher is better** (better rank preservation).

    References
    ----------
    MuST benchmark (2023). Lähnemann et al. (2020).
    """
    X_embedding = np.asarray(X_embedding, dtype=float)
    spatial_coords = np.asarray(spatial_coords, dtype=float)
    n = len(X_embedding)

    if n > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_cells, replace=False)
        X_embedding = X_embedding[idx]
        spatial_coords = spatial_coords[idx]
        n = max_cells

    k = min(k, n - 1)
    if k == 0:
        return 1.0
    kp1 = k + 1

    nn_s = NearestNeighbors(n_neighbors=kp1, algorithm="kd_tree").fit(spatial_coords)
    _, s_inds = nn_s.kneighbors(spatial_coords)
    s_inds = s_inds[:, 1:]  # (n, k)

    nn_l = NearestNeighbors(n_neighbors=kp1, algorithm="kd_tree").fit(X_embedding)
    _, l_inds = nn_l.kneighbors(X_embedding)
    l_inds = l_inds[:, 1:]  # (n, k)

    total_error = 0.0
    for i in range(n):
        rank_lookup = np.full(n, k, dtype=np.int32)
        rank_lookup[l_inds[i]] = np.arange(k, dtype=np.int32)
        lat_ranks = rank_lookup[s_inds[i]]
        total_error += float(np.sum(np.abs(np.arange(k) - lat_ranks)))

    mrre = total_error / (n * k)
    return float(max(0.0, 1.0 - mrre / k))


def spatial_knn_overlap(
    X_embedding: np.ndarray,
    spatial_coords: np.ndarray,
    k: int = 15,
    max_cells: int = 2000,
    seed: int = 42,
) -> float:
    """k-NN overlap score between spatial and latent neighbourhoods.

    For each spot, computes the fraction of its ``k`` spatial nearest
    neighbours that are also among its ``k`` latent nearest neighbours.
    The mean over all spots gives an intuitive measure of local geometry
    preservation.

    Parameters
    ----------
    X_embedding
        Array of shape ``(n_spots, n_dims)`` — latent representation.
    spatial_coords
        Array of shape ``(n_spots, 2)`` with spatial coordinates (x, y).
    k
        Neighbourhood size. Default ``15``.
    max_cells
        Subsample to this many cells before computation. Default ``2000``.
    seed
        Random seed for subsampling. Default ``42``.

    Returns
    -------
    float
        Score in ``[0, 1]``. **Higher is better** (more neighbours shared).

    References
    ----------
    MuST benchmark (2023).
    """
    X_embedding = np.asarray(X_embedding, dtype=float)
    spatial_coords = np.asarray(spatial_coords, dtype=float)
    n = len(X_embedding)

    if n > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_cells, replace=False)
        X_embedding = X_embedding[idx]
        spatial_coords = spatial_coords[idx]
        n = max_cells

    k = min(k, n - 1)
    if k == 0:
        return 1.0
    kp1 = k + 1

    nn_s = NearestNeighbors(n_neighbors=kp1, algorithm="kd_tree").fit(spatial_coords)
    _, s_inds = nn_s.kneighbors(spatial_coords)
    s_inds = s_inds[:, 1:]

    nn_l = NearestNeighbors(n_neighbors=kp1, algorithm="kd_tree").fit(X_embedding)
    _, l_inds = nn_l.kneighbors(X_embedding)
    l_inds = l_inds[:, 1:]

    overlaps = np.array([np.sum(np.isin(s_inds[i], l_inds[i])) / k for i in range(n)])
    return float(np.mean(overlaps))


def spatial_distance_correlation(
    X_embedding: np.ndarray,
    spatial_coords: np.ndarray,
    max_cells: int = 1000,
    seed: int = 42,
) -> float:
    """Spearman correlation of pairwise distances, rescaled to [0, 1].

    Computes pairwise Euclidean distances in spatial coordinate space and in
    the latent embedding, then measures their Spearman rank correlation.
    A high correlation indicates that the embedding preserves the global
    spatial distance structure.

    The Spearman correlation (in ``[-1, 1]``) is rescaled to ``[0, 1]`` via
    ``(r + 1) / 2``.

    Parameters
    ----------
    X_embedding
        Array of shape ``(n_spots, n_dims)`` — latent representation.
    spatial_coords
        Array of shape ``(n_spots, 2)`` with spatial coordinates (x, y).
    max_cells
        Subsample to this many cells before computation (O(n²) cost).
        Default ``1000``.
    seed
        Random seed for subsampling. Default ``42``.

    Returns
    -------
    float
        Score in ``[0, 1]``. **Higher is better** (better global preservation).
    """
    X_embedding = np.asarray(X_embedding, dtype=float)
    spatial_coords = np.asarray(spatial_coords, dtype=float)
    n = len(X_embedding)

    if n > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_cells, replace=False)
        X_embedding = X_embedding[idx]
        spatial_coords = spatial_coords[idx]

    sp_dists = pdist(spatial_coords, metric="euclidean")
    lat_dists = pdist(X_embedding, metric="euclidean")

    if len(sp_dists) < 2:
        return 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        corr, _ = spearmanr(sp_dists, lat_dists)
    corr = 0.0 if np.isnan(corr) else float(corr)
    return float((corr + 1.0) / 2.0)


def spatial_morans_i(
    X_embedding: np.ndarray,
    spatial_coords: np.ndarray,
    n_neighbors: int = 6,
) -> float:
    """Mean Moran's I of latent dimensions, rescaled to [0, 1].

    Computes Moran's I spatial autocorrelation statistic for each latent
    dimension using a row-standardised k-NN spatial weight matrix.  Positive
    Moran's I indicates that spatially adjacent spots have similar values in
    that latent dimension (smooth spatial variation).  The mean across all
    latent dimensions is rescaled from ``[-1, 1]`` to ``[0, 1]``.

    Parameters
    ----------
    X_embedding
        Array of shape ``(n_spots, n_dims)`` — latent representation.
    spatial_coords
        Array of shape ``(n_spots, 2)`` with spatial coordinates (x, y).
    n_neighbors
        Number of spatial neighbours for the weight matrix. Default ``6``.

    Returns
    -------
    float
        Score in ``[0, 1]``. **Higher is better** (more spatial autocorrelation
        in the latent space, indicating the model captures spatial patterns).
    """
    X_embedding = np.asarray(X_embedding, dtype=float)
    spatial_coords = np.asarray(spatial_coords, dtype=float)
    n = len(X_embedding)

    k = min(n_neighbors, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(spatial_coords)
    _, inds = nn.kneighbors(spatial_coords)
    inds = inds[:, 1:]  # (n, k), exclude self

    # Vectorised Moran's I for all latent dimensions simultaneously.
    # Using row-standardised weights: W_sum = n, so I = numerator / denominator.
    X_c = X_embedding - X_embedding.mean(axis=0)  # (n, d)
    spatial_lag = X_c[inds].mean(axis=1)  # (n, d)
    numerator = np.sum(X_c * spatial_lag, axis=0)  # (d,)
    denominator = np.sum(X_c**2, axis=0)  # (d,)
    I_per_dim = np.where(denominator > 0, numerator / denominator, 0.0)

    mean_I = float(np.mean(I_per_dim))
    return float((mean_I + 1.0) / 2.0)
