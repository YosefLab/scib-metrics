"""Evaluation metrics for perturbation-response prediction.

All metrics consume expression *deltas* (predicted and true, relative to control), matching
the output convention of `scib_metrics.perturbation._baselines`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from scib_metrics.perturbation._utils import import_pertpy

if TYPE_CHECKING:
    from collections.abc import Sequence


def delta_correlation(
    predicted_deltas: np.ndarray,
    true_deltas: np.ndarray,
    gene_indices: Sequence[np.ndarray] | None = None,
    method: Literal["pearson", "spearman"] = "pearson",
) -> dict[str, np.ndarray | float]:
    """Per-perturbation correlation between predicted and true expression deltas.

    Parameters
    ----------
    predicted_deltas
        Array of shape `(n_perturbations, n_genes)`.
    true_deltas
        Array of shape `(n_perturbations, n_genes)`, aligned row-for-row with `predicted_deltas`.
    gene_indices
        Optional per-perturbation gene index arrays (length `n_perturbations`) restricting the
        correlation to a caller-supplied gene subset (e.g. top-DE genes) for that perturbation.
    method
        `"pearson"` or `"spearman"`.

    Returns
    -------
    Dict with `"per_perturbation"` (array of shape `(n_perturbations,)`) and `"mean"` (float).
    """
    predicted_deltas = np.asarray(predicted_deltas)
    true_deltas = np.asarray(true_deltas)
    if predicted_deltas.shape != true_deltas.shape:
        raise ValueError("`predicted_deltas` and `true_deltas` must have the same shape.")
    pt = import_pertpy()
    metric_name = "pearson_distance" if method == "pearson" else "spearman_distance"
    distance = pt.tl.Distance(metric=metric_name)
    scores = np.empty(predicted_deltas.shape[0])
    for i in range(predicted_deltas.shape[0]):
        pred_i, true_i = predicted_deltas[i], true_deltas[i]
        if gene_indices is not None:
            idx = np.asarray(gene_indices[i])
            pred_i, true_i = pred_i[idx], true_i[idx]
        scores[i] = 1 - distance(pred_i[None, :], true_i[None, :])
    return {"per_perturbation": scores, "mean": float(scores.mean())}


def de_rank_recovery(
    predicted_deltas: np.ndarray,
    true_de_gene_indices: Sequence[np.ndarray],
    k: int,
) -> dict[str, np.ndarray | float]:
    """Recall@k of top predicted-delta-magnitude genes against a true DE gene set.

    Parameters
    ----------
    predicted_deltas
        Array of shape `(n_perturbations, n_genes)`.
    true_de_gene_indices
        Per-perturbation array of the true top-DE gene indices, length `n_perturbations`.
    k
        Number of top genes (by predicted delta magnitude) considered "predicted as DE".

    Returns
    -------
    Dict with `"per_perturbation"` (recall@k per perturbation) and `"mean"` (float).
    """
    predicted_deltas = np.asarray(predicted_deltas)
    if predicted_deltas.shape[0] != len(true_de_gene_indices):
        raise ValueError("`predicted_deltas` and `true_de_gene_indices` must have the same length.")
    scores = np.empty(predicted_deltas.shape[0])
    for i, true_idx in enumerate(true_de_gene_indices):
        true_idx = np.asarray(true_idx)
        predicted_top_k = np.argsort(-np.abs(predicted_deltas[i]))[:k]
        n_recovered = np.intersect1d(predicted_top_k, true_idx).shape[0]
        scores[i] = n_recovered / true_idx.shape[0]
    return {"per_perturbation": scores, "mean": float(scores.mean())}
