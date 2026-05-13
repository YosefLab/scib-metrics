"""Spatial transcriptomics metrics for scib-metrics.

Metrics are organised into three conceptual axes:

**Coordinate preservation** (``spatial_mrre``, ``spatial_knn_overlap``,
``spatial_distance_correlation``, ``spatial_morans_i``) — asks whether the
latent embedding reproduces the physical XY geometry of spots.  Appropriate
for spatial graph autoencoders (STAGATE-style) where the latent is explicitly
trained to be a surrogate for tissue coordinates.

**Niche preservation** (``spatial_niche_knn_overlap``) — asks whether cells
that share a similar local microenvironment (similar average expression of
spatial neighbours) are also close in latent space.  This is the primary
objective of models like scVIVA and other niche-aware methods.

**Domain boundary faithfulness** (``spatial_pas``, ``spatial_chaos``) — asks
whether clusters derived from the latent space are spatially coherent.
PAS (Proportion of Abnormal Spots) and CHAOS measure how well latent-derived
domains align with tissue boundaries, which is relevant for models aimed at
spatial domain identification.

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
from sklearn.cluster import KMeans
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
    raw = float(np.mean(overlaps))

    # Normalise against the random-chance baseline: for n spots and k neighbours
    # a random embedding shares k/(n-1) neighbours on average.  Rescale so that
    # random → 0 and perfect overlap → 1, then clip to [0, 1].
    chance = k / (n - 1)
    if chance >= 1.0:
        return raw
    return float(np.clip((raw - chance) / (1.0 - chance), 0.0, 1.0))


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
        Subsample to this many cells before computation. Default ``2000``.
    seed
        Random seed for subsampling. Default ``42``.

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
    n = len(X_embedding)

    if n > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_cells, replace=False)
        X_embedding = X_embedding[idx]
        spatial_coords = spatial_coords[idx]
        if X_expression is not None:
            X_expression = np.asarray(X_expression, dtype=float)[idx]
        n = max_cells

    # ── build niche descriptors ──────────────────────────────────────────────
    k_sp = min(k_spatial, n - 1)
    nn_sp = NearestNeighbors(n_neighbors=k_sp + 1, algorithm="kd_tree").fit(spatial_coords)
    _, sp_inds = nn_sp.kneighbors(spatial_coords)
    sp_inds = sp_inds[:, 1:]  # exclude self

    if X_expression is not None:
        X_expr = np.asarray(X_expression, dtype=float)
        niche_features = X_expr[sp_inds].mean(axis=1)  # (n, n_features)
    else:
        niche_features = spatial_coords[sp_inds].mean(axis=1)  # (n, 2)

    # ── kNN in latent and niche spaces ───────────────────────────────────────
    k = min(k, n - 1)
    if k == 0:
        return 1.0
    kp1 = k + 1

    nn_lat = NearestNeighbors(n_neighbors=kp1, algorithm="auto").fit(X_embedding)
    _, lat_inds = nn_lat.kneighbors(X_embedding)
    lat_inds = lat_inds[:, 1:]

    nn_niche = NearestNeighbors(n_neighbors=kp1, algorithm="auto").fit(niche_features)
    _, niche_inds = nn_niche.kneighbors(niche_features)
    niche_inds = niche_inds[:, 1:]

    # ── chance-normalised overlap ────────────────────────────────────────────
    overlaps = np.array([np.sum(np.isin(niche_inds[i], lat_inds[i])) / k for i in range(n)])
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
) -> float:
    """Proportion of Abnormal Spots (PAS), inverted to [0, 1] higher=better.

    Clusters the latent embedding with k-means, then measures how often
    spatial neighbours belong to the same cluster.  For each spot the
    *abnormality* is the fraction of its ``k_spatial`` spatial neighbours
    assigned to a different cluster; PAS is the mean over all spots.  The
    score is ``1 - PAS`` so that perfect spatial coherence of clusters → 1.

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
        Subsample to this many cells before computation. Default ``5000``.
    seed
        Random seed. Default ``42``.

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
    n = len(X_embedding)

    if n > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_cells, replace=False)
        X_embedding = X_embedding[idx]
        spatial_coords = spatial_coords[idx]
        n = max_cells

    n_clusters = min(n_clusters, n - 1)
    cluster_labels = _cluster_embedding(X_embedding, n_clusters, seed)

    k = min(k_spatial, n - 1)
    if k == 0:
        return 1.0

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(spatial_coords)
    _, inds = nn.kneighbors(spatial_coords)
    inds = inds[:, 1:]  # exclude self

    pas_per_spot = np.array([np.mean(cluster_labels[inds[i]] != cluster_labels[i]) for i in range(n)])
    pas = float(np.mean(pas_per_spot))
    return float(1.0 - pas)


def spatial_chaos(
    X_embedding: np.ndarray,
    spatial_coords: np.ndarray,
    n_clusters: int = 10,
    max_cells: int = 2000,
    seed: int = 42,
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
