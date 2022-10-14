import numpy as np
import pandas as pd

from scib_metrics.utils import silhouette_samples


def silhouette_label(X: np.ndarray, labels: np.ndarray, rescale: bool = True) -> float:
    """Average silhouette width (ASW) :cite:p:`luecken2022benchmarking`.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).
    labels
        Array of shape (n_cells,) representing label values
    rescale
        Scale asw into the range [0, 1].

    Returns
    -------
    silhouette score
    """
    asw = np.mean(silhouette_samples(X, labels))
    if rescale:
        asw = (asw + 1) / 2
    return np.mean(asw)


def silhouette_batch(X: np.ndarray, labels: np.ndarray, batch: np.ndarray, rescale: bool = True) -> float:
    """Average silhouette width (ASW) with respect to batch ids within each label :cite:p:`luecken2022benchmarking`.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).
    labels
        Array of shape (n_cells,) representing label values
    batch
        Array of shape (n_cells,) representing batch values
    rescale
        Scale asw into the range [0, 1]. If True, higher values are better.

    Returns
    -------
    silhouette score
    """
    sil_dfs = []
    unique_labels = np.unique(labels)
    for group in unique_labels:
        labels_mask = labels == group
        X_subset = X[labels_mask]
        batch_subset = batch[labels_mask]
        n_batches = len(np.unique(batch_subset))

        if (n_batches == 1) or (n_batches == X_subset.shape[0]):
            continue

        sil_per_group = silhouette_samples(X_subset, batch_subset)

        # take only absolute value
        sil_per_group = np.abs(sil_per_group)

        if rescale:
            # scale s.t. highest number is optimal
            sil_per_group = 1 - sil_per_group

        sil_dfs.append(
            pd.DataFrame(
                {
                    "group": [group] * len(sil_per_group),
                    "silhouette_score": sil_per_group,
                }
            )
        )

    sil_df = pd.concat(sil_dfs).reset_index(drop=True)
    sil_means = sil_df.groupby("group").mean()
    asw = sil_means["silhouette_score"].mean()

    return asw
