"""Naive baseline predictors for perturbation-response evaluation.

All predictors operate in delta-space: `.predict()` returns the predicted expression
*delta relative to control*, not an absolute expression profile. This matches
`pertpy.tools.PerturbationSpace.compute_control_diff`'s convention and the metric
functions in `scib_metrics.perturbation._metrics`, which all consume deltas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from scib_metrics.perturbation._utils import import_pertpy

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from anndata import AnnData

    from scib_metrics._types import NdArray


class BasePerturbationPredictor(ABC):
    """Base class for perturbation-response baseline predictors."""

    @abstractmethod
    def fit(
        self,
        adata_train: AnnData,
        target_col: str = "perturbation",
        reference_key: str = "control",
        perturbation_encodings: Mapping[str, NdArray] | None = None,
    ) -> Self:
        """Fit the baseline on training data.

        Parameters
        ----------
        adata_train
            Cell-level AnnData. `adata_train.obs[target_col]` holds the perturbation label
            of each cell; `reference_key` marks control cells.
        target_col
            `.obs` column name holding the perturbation label.
        reference_key
            Perturbation label marking control cells.
        perturbation_encodings
            Optional feature vector per perturbation name. Required by `LinearBaseline`,
            ignored by `MeanBaseline` and `AdditiveBaseline`.

        Returns
        -------
        `self`.
        """

    @abstractmethod
    def predict(self, perturbations: Sequence[str]) -> NdArray:
        """Predict expression deltas (relative to control) for the requested perturbations.

        Parameters
        ----------
        perturbations
            Names of the perturbations to predict for.

        Returns
        -------
        Array of shape `(len(perturbations), n_genes)`.
        """


def _pseudobulk_control_diff(pt, adata: AnnData, target_col: str, reference_key: str) -> tuple[AnnData, Any]:
    """Pseudobulk `adata` by `target_col` (mean mode) and subtract the control mean in place.

    Note: `pertpy.tools.PerturbationSpace` is not exported at the `pt.tl` top level (it exists
    only as the internal base class of `PseudobulkSpace`/`CentroidSpace`/etc.) — always call
    `compute_control_diff`/`add`/`subtract` on a `PseudobulkSpace` (or other concrete space)
    instance, never `pt.tl.PerturbationSpace()` directly (that raises `AttributeError`).

    Returns
    -------
    Tuple of `(diffed pseudobulk AnnData, the PseudobulkSpace instance used)` — callers that
    also need `.add()` (e.g. `AdditiveBaseline`) reuse the same instance rather than creating
    a second one.
    """
    ps = pt.tl.PseudobulkSpace()
    pseudobulk = ps.compute(adata, target_col=target_col, mode="mean")
    ps.compute_control_diff(pseudobulk, target_col=target_col, reference_key=reference_key, copy=False)
    return pseudobulk, ps


class MeanBaseline(BasePerturbationPredictor):
    """Predicts the mean training-perturbation delta for every requested perturbation.

    Deliberately naive: this is the "did your model beat just guessing the average
    perturbed profile" floor from the perturbation-evaluation literature. It is blind to
    which perturbation is being requested.
    """

    def fit(
        self,
        adata_train: AnnData,
        target_col: str = "perturbation",
        reference_key: str = "control",
        perturbation_encodings: Mapping[str, NdArray] | None = None,
    ) -> Self:
        pt = import_pertpy()
        pseudobulk, _ = _pseudobulk_control_diff(pt, adata_train, target_col, reference_key)
        is_control = pseudobulk.obs[target_col].astype(str).to_numpy() == reference_key
        deltas = np.asarray(pseudobulk.X)[~is_control]
        if deltas.shape[0] == 0:
            raise ValueError(f"No non-control perturbations found in `adata_train.obs[{target_col!r}]`.")
        self.mean_delta_ = deltas.mean(axis=0)
        return self

    def predict(self, perturbations: Sequence[str]) -> NdArray:
        return np.tile(self.mean_delta_, (len(perturbations), 1))
