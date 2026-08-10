"""Evaluation metrics for perturbation-response prediction.

All metrics consume expression *deltas* (predicted and true, relative to control), matching
the output convention of `scib_metrics.perturbation._baselines`.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

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


def systema_decomposition(predicted_deltas: np.ndarray, true_deltas: np.ndarray) -> dict[str, float]:
    """Score shared (systematic) vs. perturbation-specific components of the effect separately.

    Requires at least 2 held-out perturbations to estimate the shared component; a predictor
    that only reproduces the systematic shift will score well on `"shared"` but poorly on
    `"specific"`.

    Parameters
    ----------
    predicted_deltas
        Array of shape `(n_perturbations, n_genes)`, `n_perturbations >= 2`.
    true_deltas
        Array of shape `(n_perturbations, n_genes)`, aligned row-for-row with `predicted_deltas`.

    Returns
    -------
    Dict with `"shared"` and `"specific"` correlation scores.
    """
    predicted_deltas = np.asarray(predicted_deltas)
    true_deltas = np.asarray(true_deltas)
    if predicted_deltas.shape != true_deltas.shape:
        raise ValueError("`predicted_deltas` and `true_deltas` must have the same shape.")
    if predicted_deltas.shape[0] < 2:
        raise ValueError("`systema_decomposition` requires at least 2 held-out perturbations.")
    pt = import_pertpy()
    distance = pt.tl.Distance(metric="pearson_distance")

    true_shared = true_deltas.mean(axis=0)
    predicted_shared = predicted_deltas.mean(axis=0)
    shared_score = 1 - distance(predicted_shared[None, :], true_shared[None, :])
    if np.isnan(shared_score):
        shared_score = 0.0

    true_specific = (true_deltas - true_shared).ravel()
    predicted_specific = (predicted_deltas - predicted_shared).ravel()
    specific_score = 1 - distance(predicted_specific[None, :], true_specific[None, :])
    if np.isnan(specific_score):
        specific_score = 0.0

    return {"shared": float(shared_score), "specific": float(specific_score)}


def combination_additivity(
    adata,
    target_col: str = "perturbation",
    reference_key: str = "control",
    combinations: Sequence[str] | None = None,
    sep: str = "+",
) -> pd.DataFrame:
    """Score how well an additive model predicts combination perturbations.

    Thin wrapper over `pertpy.tools.PerturbationSpace.evaluate_combinations`. `adata` must be
    perturbation-level (one observation per perturbation, e.g. from
    `pertpy.tools.PseudobulkSpace.compute`).

    Parameters
    ----------
    adata
        Perturbation-level AnnData (one observation per perturbation).
    target_col
        `.obs` column identifying each perturbation.
    reference_key
        Control perturbation subtracted to obtain effects.
    combinations
        Combination names to evaluate. If `None`, every `obs_name` containing `sep` whose
        components are all present as singles is used.
    sep
        Separator between components in combination names.

    Returns
    -------
    DataFrame indexed by combination with `"distance"`, `"predicted_magnitude"` and
    `"measured_magnitude"` columns, or an empty DataFrame with those columns if no evaluable
    combination is present.
    """
    pt = import_pertpy()
    names = adata.obs_names.astype(str)
    if combinations is None:
        combinations = [n for n in names if sep in n and all(part in names for part in n.split(sep))]
    if not combinations:
        warnings.warn(
            f"No combination-named perturbations found in `adata.obs_names` (using separator {sep!r}); "
            "returning an empty result.",
            stacklevel=2,
        )
        return pd.DataFrame(columns=["distance", "predicted_magnitude", "measured_magnitude"])
    return pt.tl.PseudobulkSpace().evaluate_combinations(
        adata, combinations=combinations, target_col=target_col, reference_key=reference_key, sep=sep
    )
