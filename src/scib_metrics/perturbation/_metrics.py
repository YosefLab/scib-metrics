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
