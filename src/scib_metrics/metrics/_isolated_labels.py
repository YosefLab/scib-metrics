import logging

import numpy as np
import pandas as pd

from scib_metrics.utils import silhouette_samples

logger = logging.getLogger(__name__)


def isolated_labels(
    X: np.ndarray,
    labels: np.ndarray,
    batch: np.ndarray,
    rescale: bool = True,
    iso_threshold: int | None = None,
) -> float:
    """Isolated label score :cite:p:`luecken2022benchmarking`.

    Score how well labels of isolated labels are distiguished in the dataset by
    average-width silhouette score (ASW) on isolated label vs all other labels.

    The default of the original scib package is to use a cluster-based F1 scoring
    procedure, but here we use the ASW for speed and simplicity.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).
    labels
        Array of shape (n_cells,) representing label values
    batch
        Array of shape (n_cells,) representing batch values
    rescale
        Scale asw into the range [0, 1].
    iso_threshold
        Max number of batches per label for label to be considered as
        isolated, if integer. If `None`, considers minimum number of
        batches that labels are present in

    Returns
    -------
    isolated_label_score
    """
    scores = {}
    isolated_labels = _get_isolated_labels(labels, batch, iso_threshold)

    silhouette_all = silhouette_samples(X, labels)
    if rescale:
        silhouette_all = (silhouette_all + 1) / 2

    for label in isolated_labels:
        score = np.mean(silhouette_all[labels == label])
        scores[label] = score
    scores = pd.Series(scores)

    return scores.mean()


def _get_isolated_labels(labels: np.ndarray, batch: np.ndarray, iso_threshold: float):
    """Get labels that are isolated depending on the number of batches."""
    tmp = pd.DataFrame()
    label_key = "label"
    batch_key = "batch"
    tmp[label_key] = labels
    tmp[batch_key] = batch
    tmp = tmp.drop_duplicates()
    batch_per_lab = tmp.groupby(label_key).agg({batch_key: "count"})

    # threshold for determining when label is considered isolated
    if iso_threshold is None:
        iso_threshold = batch_per_lab.min().tolist()[0]

    logging.info(f"isolated labels: no more than {iso_threshold} batches per label")

    labels = batch_per_lab[batch_per_lab[batch_key] <= iso_threshold].index.tolist()
    if len(labels) == 0:
        logging.info(f"no isolated labels with less than {iso_threshold} batches")

    return np.array(labels)
