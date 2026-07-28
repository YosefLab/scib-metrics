import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, jensenshannon

from scib_metrics.nearest_neighbors import NeighborsResults


def _js(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    p = np.clip(p, a_min=epsilon, a_max=None)
    q = np.clip(q, a_min=epsilon, a_max=None)
    p = p / p.sum()
    q = q / q.sum()
    return jensenshannon(p, q, base=2.0)


def _build_distribution(
    neighbor_indices: np.ndarray,
    batches: np.ndarray,
    labels: np.ndarray,
    dist_type: str = "count",
) -> np.ndarray:
    """Build local or global batch distribution per cell.

    Parameters
    ----------
    neighbor_indices
        Array of shape (n_cells, k) with kNN indices.
    batches
        Array of shape (n_cells,) with integer-encoded batch labels.
    labels
        Array of shape (n_cells,) with integer-encoded cell type labels.
    dist_type
        One of 'count' (local) or 'global'.

    Returns
    -------
    dist
        Array of shape (n_cells, n_batches) with the distribution for each cell.
    """
    n_cells = len(batches)
    unique_batches = np.unique(batches)
    n_batches = len(unique_batches)
    batch_to_idx = {b: i for i, b in enumerate(unique_batches)}
    k = neighbor_indices.shape[1]

    dist = np.zeros((n_cells, n_batches), dtype=float)

    celltype_counts = pd.Series(labels).value_counts().to_dict()

    for cell_i in range(n_cells):
        cell_type = labels[cell_i]
        k_adjusted = min(k, celltype_counts[cell_type])
        neigh_idx = neighbor_indices[cell_i, :k_adjusted]
        neigh_labels = labels[neigh_idx]
        neigh_batches = batches[neigh_idx]

        for b in unique_batches:
            b_col = batch_to_idx[b]
            if dist_type == "global":
                dist[cell_i, b_col] = np.sum((batches == b) & (labels == cell_type))
            else:
                same_type_batch = (neigh_labels == cell_type) & (neigh_batches == b)
                dist[cell_i, b_col] = same_type_batch.sum()

    return dist


def _compute_js_scores(
    neighbor_indices: np.ndarray,
    batches: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute per-cell JS distance between local and global batch distributions.

    Returns
    -------
    js_distances
        Array of shape (n_cells,) with JS distance per cell.
    """
    local_dist = _build_distribution(neighbor_indices, batches, labels, dist_type="count")
    global_dist = _build_distribution(neighbor_indices, batches, labels, dist_type="global")

    js_distances = np.array([_js(local_dist[i], global_dist[i]) for i in range(len(batches))])
    return js_distances


def _compute_intra_inter_ratio(
    X_emb: np.ndarray,
    batches: np.ndarray,
    labels: np.ndarray,
    agg: str = "median",
) -> np.ndarray:
    """Compute per-cell intra/inter batch distance ratio (same cell type).

    Parameters
    ----------
    X_emb
        Embedding array of shape (n_cells, n_dims).
    batches
        Integer-encoded batch labels.
    labels
        Integer-encoded cell type labels.
    agg
        Aggregation function: 'median' or 'mean'.

    Returns
    -------
    ratio
        Array of shape (n_cells,) with intra/inter ratio. NaN where undefined.
    """
    agg_fn = np.nanmedian if agg == "median" else np.nanmean
    n_cells = len(batches)
    intra = np.zeros(n_cells)
    inter = np.zeros(n_cells)

    unique_pairs = np.unique(np.stack([labels, batches], axis=1), axis=0)

    for ct, b in unique_pairs:
        group_mask = (labels == ct) & (batches == b)
        group_idx = np.where(group_mask)[0]
        X_group = X_emb[group_idx]

        # Intra: same (cell_type, batch), exclude self
        if len(group_idx) > 1:
            D_intra = cdist(X_group, X_group, metric="euclidean")
            np.fill_diagonal(D_intra, np.nan)
            intra[group_idx] = agg_fn(D_intra, axis=1)

        # Inter: same cell_type, different batch
        other_mask = (labels == ct) & (batches != b)
        other_idx = np.where(other_mask)[0]
        if len(other_idx) > 0:
            X_other = X_emb[other_idx]
            inter[group_idx] = np.median(cdist(X_group, X_other, metric="euclidean"), axis=1)
        else:
            inter[group_idx] = intra[group_idx]

    # Avoid division by zero
    ratio = np.where(
        (intra == 0) | (inter == 0),
        np.nan,
        intra / inter,
    )
    return ratio


def sbee(
    X: NeighborsResults,
    X_emb: np.ndarray,
    batches: np.ndarray,
    labels: np.ndarray,
    sensitivity: float = 0.15,
) -> float:
    """Compute sBEE (single-cell Batch Effect Evaluator) score :cite:p:`myradov2026systematic`.

    sBEE is a per-cell batch integration metric that produces scores in [0, 1],
    where higher values indicate better batch mixing. It combines two components
    via their harmonic mean.

    The **distance component** checks whether a cell is geometrically closer to
    same-type cells from other batches than to same-type cells from its own batch.
    When the ratio of intra-batch to inter-batch distance is 1 or above, the
    component is set to 1 (no penalty). When the ratio is below 1, a penalty is
    applied that grows with the degree of separation. Cells whose cell type appears
    in only one batch are assigned a perfect score, as batch correction is not
    applicable there.

    The **neighborhood composition component** checks whether the local batch
    composition around a cell matches the global batch distribution for that cell
    type. It compares batch proportions among same-type cells in the k-nearest
    neighborhood against global proportions using Jensen-Shannon distance. Smaller
    divergence gives a higher score.

    The two components are combined via harmonic mean. A low score on either
    component pulls the overall score down. Cell-type scores are computed by
    macro-averaging across batches so that each batch contributes equally
    regardless of its size.

    Parameters
    ----------
    X
        A :class:`~scib_metrics.nearest_neighbors.NeighborsResults` object
        (kNN graph of the integrated embedding).
    X_emb
        Integrated embedding array of shape (n_cells, n_dims). Used to compute
        intra/inter batch distances.
    batches
        Array of shape (n_cells,) with batch labels (any dtype; will be encoded).
    labels
        Array of shape (n_cells,) with cell type labels (any dtype; will be encoded).
    sensitivity
        Controls the sharpness of the distance component penalty. Default: 0.15.

    Returns
    -------
    float
        sBEE score in [0, 1]. Higher is better.
    """
    if len(batches) != len(X.indices):
        raise ValueError("Length of batches does not match number of cells.")
    if len(labels) != len(X.indices):
        raise ValueError("Length of labels does not match number of cells.")
    if X_emb.shape[0] != len(X.indices):
        raise ValueError("X_emb row count does not match number of cells.")

    # Encode labels and batches as integers
    batches_enc = np.asarray(pd.Categorical(batches).codes)
    labels_enc = np.asarray(pd.Categorical(labels).codes)

    neighbor_indices = X.indices  # shape (n_cells, k)

    # 1. JS scores
    js_distances = _compute_js_scores(neighbor_indices, batches_enc, labels_enc)
    js_score = 1.0 - js_distances

    # 2. Intra/inter ratio scores
    ratio = _compute_intra_inter_ratio(X_emb, batches_enc, labels_enc)
    effective_ratio = np.clip(ratio, a_min=None, a_max=1.0)
    ratio_score = np.exp(-np.abs(1.0 - effective_ratio) / sensitivity)

    # 3. Per-cell harmonic mean (sBEE)
    denom = js_score + ratio_score
    per_cell_sbee = np.where(denom > 0, 2 * js_score * ratio_score / denom, 0.0)

    # 4. Macro-average: per (cell_type, batch) → per cell_type → overall
    scores_df = pd.DataFrame(
        {
            "cell_type": labels,
            "batch": batches,
            "sBEE": per_cell_sbee,
        }
    )

    final_score = scores_df.groupby(["cell_type", "batch"])["sBEE"].mean().groupby(level="cell_type").mean().mean()

    return float(final_score)
