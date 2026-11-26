"""Spatial transcriptomics metrics for scib-metrics.

Metrics are organised into three conceptual axes:

**Coordinate preservation** (``spatial_mrre``, ``spatial_knn_overlap``,
``spatial_distance_correlation``, ``spatial_morans_i``) — asks whether the
latent embedding reproduces the physical XY geometry of spots.  Appropriate
for spatial graph autoencoders (STAGATE-style) where the latent is explicitly
trained to be a surrogate for tissue coordinates.

**Niche preservation** (``spatial_niche_knn_overlap``,
``spatial_neighbor_knn_overlap``) — asks whether cells that share a similar
local microenvironment or explicit spatial graph neighbourhood are also close
in latent space.  This is the primary objective of models like scVIVA and
other niche-aware methods.

**Domain boundary faithfulness** (``spatial_pas``, ``spatial_chaos``) — asks
whether clusters derived from the latent space are spatially coherent.
PAS (Proportion of Abnormal Spots) and CHAOS measure how well latent-derived
domains align with tissue boundaries, which is relevant for models aimed at
spatial domain identification.

All functions return a float in **[0, 1]** where **higher is always better**.
Scores are chance-corrected or clipped so that a random embedding scores
close to 0.  Metrics that look at spatial neighbours evaluate a subsample of
``max_cells`` cells, but search neighbours among all cells, so a "spatial
neighbour" is always an adjacent cell.

References
----------
Hu et al. (2024) Benchmarking clustering, alignment, and integration
    methods for spatial transcriptomics. PMC11312151.
Chen et al. (2025) A comprehensive benchmarking for spatially resolved
    transcriptomics clustering methods. PMC12747554.
MuST (2023) Multi-modal spatial transcriptomics benchmark.
"""

import warnings
from collections.abc import Callable

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def _query_indices(n: int, max_cells: int, seed: int) -> np.ndarray:
    """Indices of the cells at which a metric is evaluated.

    Only the *query* cells are subsampled; neighbour searches always run over
    all ``n`` cells, so "spatial neighbour" keeps its meaning (adjacent cells)
    regardless of ``max_cells``.
    """
    if n <= max_cells:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, max_cells, replace=False))


def _knn_of_queries(data: np.ndarray, query_idx: np.ndarray, k: int) -> np.ndarray:
    """``k`` nearest neighbours (among all rows of ``data``) of each query row, excluding itself."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(data)
    inds = nn.kneighbors(data[query_idx], return_distance=False)
    keep = inds != query_idx[:, None]
    # With duplicate points the query itself may fall outside the k+1 hits;
    # then drop the farthest hit instead.
    no_self = keep.all(axis=1)
    keep[no_self, -1] = False
    return inds[keep].reshape(len(query_idx), k)


def _chance_normalised_overlap(ref_inds: np.ndarray, lat_inds: np.ndarray, k: int, n: int) -> float:
    """Mean fraction of reference neighbours found in the latent kNN, rescaled so random -> 0."""
    raw = float(np.mean((ref_inds[:, :, None] == lat_inds[:, None, :]).any(-1).mean(axis=1)))
    chance = k / (n - 1)
    if chance >= 1.0:
        return raw
    return float(np.clip((raw - chance) / (1.0 - chance), 0.0, 1.0))


def _sample_weighted_score(
    fn: Callable[..., float],
    X_embedding: np.ndarray,
    spatial_coords: np.ndarray,
    sample_labels: np.ndarray | None,
    **kwargs,
) -> float | None:
    """Average a spatial metric within independent spatial samples."""
    if sample_labels is None:
        return None

    sample_labels = np.asarray(sample_labels)
    if len(sample_labels) != len(X_embedding):
        raise ValueError("sample_labels must have the same length as X_embedding and spatial_coords.")

    scores = []
    weights = []
    for sample in np.unique(sample_labels):
        mask = sample_labels == sample
        n = int(mask.sum())
        if n == 0:
            continue
        scores.append(fn(X_embedding[mask], spatial_coords[mask], sample_labels=None, **kwargs))
        weights.append(n)

    if not scores:
        return np.nan
    return float(np.average(scores, weights=weights))


def spatial_mrre(
    X_embedding: np.ndarray,
    spatial_coords: np.ndarray,
    k: int = 15,
    max_cells: int = 2000,
    seed: int = 42,
    *,
    sample_labels: np.ndarray | None = None,
) -> float:
    """Mean Relative Rank Error (MRRE), normalised to [0, 1].

    For each spot, finds its ``k`` nearest spatial neighbours and compares
    their rank ordering in spatial space to their rank ordering in the latent
    embedding (rank ``k`` if it is not a latent k-NN).  The mean absolute
    rank difference, normalised by ``k``, measures how much the local
    geometry is distorted.  ``1 - MRRE/k`` is then rescaled against its
    expected value for a random embedding, so random -> 0 and perfect rank
    preservation -> 1.

    Parameters
    ----------
    X_embedding
        Array of shape ``(n_spots, n_dims)`` — latent representation.
    spatial_coords
        Array of shape ``(n_spots, 2)`` with spatial coordinates (x, y).
    k
        Neighbourhood size. Default ``15``.
    max_cells
        Number of cells at which the score is evaluated. Neighbours are
        always searched among all cells. Default ``2000``.
    seed
        Random seed for subsampling. Default ``42``.
    sample_labels
        Optional labels for independent spatial samples/sections. When set,
        the score is computed within each sample and averaged by sample size.

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
    grouped = _sample_weighted_score(
        spatial_mrre, X_embedding, spatial_coords, sample_labels, k=k, max_cells=max_cells, seed=seed
    )
    if grouped is not None:
        return grouped
    n = len(X_embedding)
    k = min(k, n - 1)
    if k == 0:
        return 1.0
    q = _query_indices(n, max_cells, seed)
    s_inds = _knn_of_queries(spatial_coords, q, k)
    l_inds = _knn_of_queries(X_embedding, q, k)

    # Latent rank of every spatial neighbour; neighbours outside the latent
    # top-k get rank k.
    eq = s_inds[:, :, None] == l_inds[:, None, :]
    lat_ranks = np.where(eq.any(-1), eq.argmax(-1), k)
    mrre = float(np.mean(np.abs(np.arange(k)[None, :] - lat_ranks))) / k
    score = 1.0 - mrre

    # Expected score of a random embedding: a spatial neighbour lands in the
    # latent top-k with probability p = k / (n - 1), at a uniform rank.
    p = k / (n - 1)
    err_out = (k + 1) / 2  # mean |j - k| for j uniform on 0..k-1
    err_in = (k**2 - 1) / (3 * k)  # mean |j - u| for j, u uniform on 0..k-1
    baseline = 1.0 - ((1 - p) * err_out + p * err_in) / k
    if baseline >= 1.0:
        return float(np.clip(score, 0.0, 1.0))
    return float(np.clip((score - baseline) / (1.0 - baseline), 0.0, 1.0))


def spatial_knn_overlap(
    X_embedding: np.ndarray,
    spatial_coords: np.ndarray,
    k: int = 15,
    max_cells: int = 2000,
    seed: int = 42,
    *,
    sample_labels: np.ndarray | None = None,
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
        Number of cells at which the score is evaluated. Neighbours are
        always searched among all cells. Default ``2000``.
    seed
        Random seed for subsampling. Default ``42``.
    sample_labels
        Optional labels for independent spatial samples/sections. When set,
        the score is computed within each sample and averaged by sample size.

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
    grouped = _sample_weighted_score(
        spatial_knn_overlap, X_embedding, spatial_coords, sample_labels, k=k, max_cells=max_cells, seed=seed
    )
    if grouped is not None:
        return grouped
    n = len(X_embedding)
    k = min(k, n - 1)
    if k == 0:
        return 1.0
    q = _query_indices(n, max_cells, seed)
    s_inds = _knn_of_queries(spatial_coords, q, k)
    l_inds = _knn_of_queries(X_embedding, q, k)
    return _chance_normalised_overlap(s_inds, l_inds, k, n)


def spatial_distance_correlation(
    X_embedding: np.ndarray,
    spatial_coords: np.ndarray,
    max_cells: int = 1000,
    seed: int = 42,
    *,
    sample_labels: np.ndarray | None = None,
) -> float:
    """Spearman correlation of pairwise distances, clipped to [0, 1].

    Computes pairwise Euclidean distances in spatial coordinate space and in
    the latent embedding, then measures their Spearman rank correlation.
    A high correlation indicates that the embedding preserves the global
    spatial distance structure.

    The Spearman correlation is clipped to ``[0, 1]`` (negative values -> 0),
    so a random embedding scores ~0.

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
    sample_labels
        Optional labels for independent spatial samples/sections. When set,
        the score is computed within each sample and averaged by sample size.

    Returns
    -------
    float
        Score in ``[0, 1]``. **Higher is better** (better global preservation).
    """
    X_embedding = np.asarray(X_embedding, dtype=float)
    spatial_coords = np.asarray(spatial_coords, dtype=float)
    grouped = _sample_weighted_score(
        spatial_distance_correlation, X_embedding, spatial_coords, sample_labels, max_cells=max_cells, seed=seed
    )
    if grouped is not None:
        return grouped
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
    # Negative correlations are clipped to 0 so that a random embedding (r ~ 0)
    # scores ~0 rather than 0.5.
    return float(np.clip(corr, 0.0, 1.0))


def spatial_morans_i(
    X_embedding: np.ndarray,
    spatial_coords: np.ndarray,
    n_neighbors: int = 6,
    *,
    sample_labels: np.ndarray | None = None,
) -> float:
    """Mean Moran's I of latent dimensions, clipped to [0, 1].

    Computes Moran's I spatial autocorrelation statistic for each latent
    dimension using a row-standardised k-NN spatial weight matrix.  Positive
    Moran's I indicates that spatially adjacent spots have similar values in
    that latent dimension (smooth spatial variation).  The mean across all
    latent dimensions is clipped to ``[0, 1]`` (negative autocorrelation -> 0),
    so a random embedding scores ~0.

    Parameters
    ----------
    X_embedding
        Array of shape ``(n_spots, n_dims)`` — latent representation.
    spatial_coords
        Array of shape ``(n_spots, 2)`` with spatial coordinates (x, y).
    n_neighbors
        Number of spatial neighbours for the weight matrix. Default ``6``.
    sample_labels
        Optional labels for independent spatial samples/sections. When set,
        the score is computed within each sample and averaged by sample size.

    Returns
    -------
    float
        Score in ``[0, 1]``. **Higher is better** (more spatial autocorrelation
        in the latent space, indicating the model captures spatial patterns).
    """
    X_embedding = np.asarray(X_embedding, dtype=float)
    spatial_coords = np.asarray(spatial_coords, dtype=float)
    grouped = _sample_weighted_score(
        spatial_morans_i, X_embedding, spatial_coords, sample_labels, n_neighbors=n_neighbors
    )
    if grouped is not None:
        return grouped
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
    # Clip instead of rescaling with (I + 1) / 2 so a random embedding
    # (I ~ -1 / (n - 1)) scores ~0 rather than 0.5.
    return float(np.clip(mean_I, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Niche preservation
# ---------------------------------------------------------------------------


def spatial_niche_knn_overlap(
    X_embedding: np.ndarray,
    spatial_coords: np.ndarray,
    X_expression: np.ndarray | None = None,
    k: int = 15,
    k_spatial: int = 6,
    max_cells: int = 2000,
    seed: int = 42,
    *,
    sample_labels: np.ndarray | None = None,
) -> float:
    """k-NN overlap between latent space and spatial niche feature space.

    For each spot, the *niche feature* is the mean embedding (or expression)
    of its ``k_spatial`` spatial neighbours.  This vector encodes the local
    microenvironment rather than the spot's own position.  The score is the
    chance-normalised fraction of latent k-NN that are also niche-feature
    k-NN — analogous to :func:`spatial_knn_overlap` but using niche
    descriptors as the reference instead of raw XY coordinates.

    Unlike coordinate-preservation metrics, this rewards models that learn
    to represent *who shares the same microenvironment*, which is the
    primary objective of niche-aware methods such as scVIVA.

    Parameters
    ----------
    X_embedding
        Array of shape ``(n_spots, n_dims)`` — latent representation.
    spatial_coords
        Array of shape ``(n_spots, 2)`` with spatial coordinates (x, y).
    X_expression
        Array of shape ``(n_spots, n_features)`` used to build niche
        descriptors.  Typically the pre-integrated PCA embedding so that
        niche features remain low-dimensional.  If ``None``, falls back to
        averaging spatial coordinates of neighbours, which degrades the
        metric to a coarser spatial-proximity measure.
    k
        Neighbourhood size for latent and niche kNN. Default ``15``.
    k_spatial
        Number of spatial neighbours used to aggregate the niche descriptor.
        Default ``6``.
    max_cells
        Number of cells at which the score is evaluated. Neighbours are
        always searched among all cells. Default ``2000``.
    seed
        Random seed for subsampling. Default ``42``.
    sample_labels
        Optional labels for independent spatial samples/sections. When set,
        niche descriptors and kNN overlaps are computed within each sample and
        averaged by sample size.

    Returns
    -------
    float
        Score in ``[0, 1]``. **Higher is better** (latent neighbours match
        niche-feature neighbours more often than chance).

    References
    ----------
    Inspired by scVIVA (Boyeau et al.) and COVET/ENVI niche representations.
    """
    X_embedding = np.asarray(X_embedding, dtype=float)
    spatial_coords = np.asarray(spatial_coords, dtype=float)
    if sample_labels is not None:
        sample_labels = np.asarray(sample_labels)
        if len(sample_labels) != len(X_embedding):
            raise ValueError("sample_labels must have the same length as X_embedding and spatial_coords.")
        scores = []
        weights = []
        X_expression_arr = None if X_expression is None else np.asarray(X_expression, dtype=float)
        for sample in np.unique(sample_labels):
            mask = sample_labels == sample
            n_sample = int(mask.sum())
            if n_sample == 0:
                continue
            scores.append(
                spatial_niche_knn_overlap(
                    X_embedding[mask],
                    spatial_coords[mask],
                    None if X_expression_arr is None else X_expression_arr[mask],
                    sample_labels=None,
                    k=k,
                    k_spatial=k_spatial,
                    max_cells=max_cells,
                    seed=seed,
                )
            )
            weights.append(n_sample)
        return float(np.average(scores, weights=weights)) if scores else np.nan
    n = len(X_embedding)
    k = min(k, n - 1)
    k_sp = min(k_spatial, n - 1)
    if k == 0 or k_sp == 0:
        return 1.0

    # ── niche descriptors from the true spatial neighbours (all cells) ──────
    all_idx = np.arange(n)
    sp_inds = _knn_of_queries(spatial_coords, all_idx, k_sp)
    if X_expression is not None:
        niche_features = np.asarray(X_expression, dtype=float)[sp_inds].mean(axis=1)
    else:
        niche_features = spatial_coords[sp_inds].mean(axis=1)

    # ── kNN in latent and niche spaces, evaluated at a subsample of cells ───
    q = _query_indices(n, max_cells, seed)
    lat_inds = _knn_of_queries(X_embedding, q, k)
    niche_inds = _knn_of_queries(niche_features, q, k)
    return _chance_normalised_overlap(niche_inds, lat_inds, k, n)


def spatial_neighbor_knn_overlap(
    X_embedding: np.ndarray,
    neighbor_indices: np.ndarray,
    k: int = 15,
    max_cells: int = 2000,
    seed: int = 42,
    *,
    sample_labels: np.ndarray | None = None,
) -> float:
    """k-NN overlap between latent space and a supplied spatial graph.

    This measures whether neighbours from an external graph, such as scVIVA's
    ``index_neighbor``/niche graph, are also close in the learned latent space.
    It is useful when the model is trained against a graph that may not be
    identical to raw Euclidean coordinate kNN.

    Parameters
    ----------
    X_embedding
        Array of shape ``(n_spots, n_dims)`` — latent representation.
    neighbor_indices
        Integer array of shape ``(n_spots, n_neighbors)`` containing reference
        graph neighbours for each spot. Self indices and negative padding are
        ignored.
    k
        Number of reference and latent neighbours to compare. Default ``15``.
    max_cells
        Number of cells at which the score is evaluated. Neighbours are
        always searched among all cells. Default ``2000``.
    seed
        Random seed for subsampling. Default ``42``.
    sample_labels
        Optional labels for independent spatial samples/sections. When set,
        latent kNN and graph-neighbour overlap are computed within each sample
        and averaged by sample size.

    Returns
    -------
    float
        Score in ``[0, 1]``. **Higher is better** (latent neighbours preserve
        explicit graph connectivity more often than chance).
    """
    X_embedding = np.asarray(X_embedding, dtype=float)
    neighbor_indices = np.asarray(neighbor_indices)

    if len(neighbor_indices) != len(X_embedding):
        raise ValueError("neighbor_indices must have the same number of rows as X_embedding.")

    if sample_labels is not None:
        sample_labels = np.asarray(sample_labels)
        if len(sample_labels) != len(X_embedding):
            raise ValueError("sample_labels must have the same length as X_embedding.")
        scores = []
        weights = []
        global_to_local = np.full(len(X_embedding), -1, dtype=int)
        for sample in np.unique(sample_labels):
            mask = sample_labels == sample
            global_idx = np.flatnonzero(mask)
            n_sample = len(global_idx)
            if n_sample == 0:
                continue
            global_to_local[global_idx] = np.arange(n_sample)
            sample_neighbors = neighbor_indices[mask]
            local_neighbors = np.full(sample_neighbors.shape, -1, dtype=int)
            valid = (sample_neighbors >= 0) & (sample_neighbors < len(X_embedding))
            local_neighbors[valid] = global_to_local[sample_neighbors[valid]]
            scores.append(
                spatial_neighbor_knn_overlap(
                    X_embedding[mask],
                    local_neighbors,
                    k=k,
                    max_cells=max_cells,
                    seed=seed,
                    sample_labels=None,
                )
            )
            weights.append(n_sample)
            global_to_local[global_idx] = -1
        return float(np.average(scores, weights=weights)) if scores else np.nan

    n = len(X_embedding)
    k = min(k, n - 1)
    if k == 0:
        return 1.0
    q = _query_indices(n, max_cells, seed)
    lat_inds = _knn_of_queries(X_embedding, q, k)

    overlaps = []
    for row_pos, i in enumerate(q):
        row = np.asarray(neighbor_indices[i], dtype=int)
        row = row[(row >= 0) & (row < n) & (row != i)]
        if row.size == 0:
            continue
        ref = row[:k]
        overlaps.append(np.sum(np.isin(ref, lat_inds[row_pos])) / len(ref))

    if not overlaps:
        return np.nan
    raw = float(np.mean(overlaps))
    chance = k / (n - 1)
    if chance >= 1.0:
        return raw
    return float(np.clip((raw - chance) / (1.0 - chance), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Domain boundary faithfulness
# ---------------------------------------------------------------------------


def _cluster_embedding(X_embedding: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    """K-means cluster X_embedding; returns integer label array."""
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    return km.fit_predict(X_embedding).astype(np.int32)


def spatial_pas(
    X_embedding: np.ndarray,
    spatial_coords: np.ndarray,
    n_clusters: int = 10,
    k_spatial: int = 6,
    max_cells: int = 5000,
    seed: int = 42,
    *,
    sample_labels: np.ndarray | None = None,
) -> float:
    """Proportion of Abnormal Spots (PAS), inverted to [0, 1] higher=better.

    Clusters the latent embedding with k-means, then measures how often
    spatial neighbours belong to the same cluster.  For each spot the
    *abnormality* is the fraction of its ``k_spatial`` spatial neighbours
    assigned to a different cluster; PAS is the mean over all spots.  The
    score is ``1 - PAS``, rescaled against the agreement expected by chance
    (``sum_c p_c^2``) so that perfect spatial coherence of clusters → 1.  Random clusters → 0.

    A high score indicates that latent-derived domains are spatially
    contiguous and align with tissue boundaries.

    Parameters
    ----------
    X_embedding
        Array of shape ``(n_spots, n_dims)`` — latent representation.
    spatial_coords
        Array of shape ``(n_spots, 2)`` with spatial coordinates (x, y).
    n_clusters
        Number of k-means clusters to derive from the latent embedding.
        Default ``10``.
    k_spatial
        Number of spatial neighbours used to assess cluster consistency.
        Default ``6``.
    max_cells
        Number of cells at which the score is evaluated (and on which k-means
        is fitted). Spatial neighbours are searched among all cells.
        Default ``5000``.
    seed
        Random seed. Default ``42``.
    sample_labels
        Optional labels for independent spatial samples/sections. When set,
        latent clusters and spatial-neighbour consistency are computed within
        each sample and averaged by sample size.

    Returns
    -------
    float
        Score in ``[0, 1]``. **Higher is better** (more spatially coherent
        clusters).

    References
    ----------
    Hu et al. (2024) Benchmarking clustering methods for spatial
        transcriptomics. PMC11312151.
    """
    X_embedding = np.asarray(X_embedding, dtype=float)
    spatial_coords = np.asarray(spatial_coords, dtype=float)
    grouped = _sample_weighted_score(
        spatial_pas,
        X_embedding,
        spatial_coords,
        sample_labels,
        n_clusters=n_clusters,
        k_spatial=k_spatial,
        max_cells=max_cells,
        seed=seed,
    )
    if grouped is not None:
        return grouped
    n = len(X_embedding)
    k = min(k_spatial, n - 1)
    if k == 0:
        return 1.0
    q = _query_indices(n, max_cells, seed)

    # k-means is fitted on the subsample and applied to all cells, so that
    # every true spatial neighbour has a cluster label.
    n_clusters = min(n_clusters, len(q) - 1)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(X_embedding[q])
    cluster_labels = km.predict(X_embedding).astype(np.int32)

    inds = _knn_of_queries(spatial_coords, q, k)
    agree = float(np.mean(cluster_labels[inds] == cluster_labels[q][:, None]))
    # Chance agreement: probability that two random cells share a cluster.
    frac = np.bincount(cluster_labels) / n
    chance = float(np.sum(frac**2))
    if chance >= 1.0:
        return agree
    return float(np.clip((agree - chance) / (1.0 - chance), 0.0, 1.0))


def spatial_chaos(
    X_embedding: np.ndarray,
    spatial_coords: np.ndarray,
    n_clusters: int = 10,
    max_cells: int = 2000,
    seed: int = 42,
    *,
    sample_labels: np.ndarray | None = None,
) -> float:
    """CHAOS score (spatial cluster compactness), inverted to [0, 1] higher=better.

    Clusters the latent embedding with k-means, then measures how spatially
    compact the clusters are.  For each cluster the mean pairwise spatial
    distance is computed and normalised by the global mean pairwise spatial
    distance.  CHAOS is the mean normalised intra-cluster distance; the score
    is ``1 - CHAOS``.

    A high score means that latent-derived clusters occupy small, contiguous
    patches of tissue rather than scattered, fragmented regions.

    Parameters
    ----------
    X_embedding
        Array of shape ``(n_spots, n_dims)`` — latent representation.
    spatial_coords
        Array of shape ``(n_spots, 2)`` with spatial coordinates (x, y).
    n_clusters
        Number of k-means clusters to derive from the latent embedding.
        Default ``10``.
    max_cells
        Subsample to this many cells before computation (O(n²) cost for
        pairwise distances). Default ``2000``.
    seed
        Random seed. Default ``42``.
    sample_labels
        Optional labels for independent spatial samples/sections. When set,
        latent clusters and spatial compactness are computed within each
        sample and averaged by sample size.

    Returns
    -------
    float
        Score in ``[0, 1]``. **Higher is better** (more spatially compact
        clusters).

    References
    ----------
    Hu et al. (2024) Benchmarking clustering methods for spatial
        transcriptomics. PMC11312151.
    Chen et al. (2025) A comprehensive benchmarking for spatially resolved
        transcriptomics clustering methods. PMC12747554.
    """
    X_embedding = np.asarray(X_embedding, dtype=float)
    spatial_coords = np.asarray(spatial_coords, dtype=float)
    grouped = _sample_weighted_score(
        spatial_chaos,
        X_embedding,
        spatial_coords,
        sample_labels,
        n_clusters=n_clusters,
        max_cells=max_cells,
        seed=seed,
    )
    if grouped is not None:
        return grouped
    n = len(X_embedding)

    if n > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_cells, replace=False)
        X_embedding = X_embedding[idx]
        spatial_coords = spatial_coords[idx]
        n = max_cells

    n_clusters = min(n_clusters, n - 1)
    cluster_labels = _cluster_embedding(X_embedding, n_clusters, seed)

    global_dists = pdist(spatial_coords)
    global_mean = float(np.mean(global_dists)) if len(global_dists) > 0 else 0.0
    if global_mean == 0.0:
        return 1.0

    per_cluster = []
    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.sum() < 2:
            per_cluster.append(0.0)
            continue
        d = pdist(spatial_coords[mask])
        per_cluster.append(float(np.mean(d)))

    chaos = float(np.mean(per_cluster)) / global_mean
    return float(np.clip(1.0 - chaos, 0.0, 1.0))
