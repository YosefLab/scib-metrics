"""Orchestrator for the perturbation-prediction evaluation surface."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import anndata
import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.preprocessing import MinMaxScaler

from scib_metrics.perturbation._baselines import AdditiveBaseline, LinearBaseline, MeanBaseline
from scib_metrics.perturbation._metrics import (
    combination_additivity,
    de_rank_recovery,
    delta_correlation,
    systema_decomposition,
)
from scib_metrics.perturbation._utils import import_pertpy

if TYPE_CHECKING:
    from scib_metrics._types import NdArray

Kwargs = dict[str, Any]
MetricType = bool | Kwargs

_BASELINE_CLASSES = {"mean": MeanBaseline, "additive": AdditiveBaseline, "linear": LinearBaseline}


@dataclass(frozen=True)
class PerturbationBaselines:
    """Specification of which baseline predictors to run in the pipeline.

    Baselines can be included using a boolean flag. Custom keyword args (passed to the
    predictor's constructor) can be used by passing a dictionary here.
    """

    mean: MetricType = True
    additive: MetricType = True
    linear: MetricType = True


@dataclass(frozen=True)
class PerturbationMetrics:
    """Specification of which perturbation-evaluation metrics to run in the pipeline."""

    delta_correlation: MetricType = True
    de_rank_recovery: MetricType = False
    systema_decomposition: MetricType = True
    combination_additivity: MetricType = True
    ground_truth_significance: bool = False


class PerturbationBenchmarker:
    """Benchmarking pipeline for perturbation-response prediction.

    Runs mandatory naive baselines (mean / additive / linear) and, optionally, the user's own
    model predictions, through a shared set of evaluation metrics, so every run carries an
    honest floor.

    Parameters
    ----------
    adata_train
        Cell-level AnnData. `adata_train.obs[target_col]` holds the perturbation label of each
        cell; `reference_key` marks control cells.
    adata_test
        Held-out cell-level AnnData (not pre-averaged), `adata_test.obs[target_col]`
        identifying which held-out perturbation each row belongs to.
    predictions
        The user's own model predictions to compare against the baselines, as
        `{model_name: deltas}` where `deltas` has shape `(n_held_out_perturbations, n_genes)`
        aligned to the sorted, deduplicated, non-control values of
        `adata_test.obs[target_col]`.
    target_col
        `.obs` column name holding the perturbation label, in both `adata_train` and `adata_test`.
    reference_key
        Perturbation label marking control cells, in both `adata_train` and `adata_test`.
    perturbation_encodings
        Optional feature vector per perturbation name, covering both trained and held-out
        names. Required if `baselines.linear` is enabled.
    true_de_gene_indices
        Optional true top-DE gene indices per held-out perturbation. Required if
        `metrics.de_rank_recovery` is enabled.
    baselines
        Specification of which baseline predictors to run.
    metrics
        Specification of which metrics to run.
    """

    def __init__(
        self,
        adata_train: AnnData,
        adata_test: AnnData,
        predictions: dict[str, NdArray] | None = None,
        target_col: str = "perturbation",
        reference_key: str = "control",
        perturbation_encodings: dict[str, NdArray] | None = None,
        true_de_gene_indices: dict[str, NdArray] | None = None,
        baselines: PerturbationBaselines = PerturbationBaselines(),
        metrics: PerturbationMetrics = PerturbationMetrics(),
    ) -> None:
        self.adata_train = adata_train
        self.adata_test = adata_test
        self.predictions = predictions or {}
        self.target_col = target_col
        self.reference_key = reference_key
        self.perturbation_encodings = perturbation_encodings
        self.true_de_gene_indices = true_de_gene_indices
        self.baselines = baselines
        self.metrics = metrics
        self._results: pd.DataFrame | None = None
        self._combination_additivity: pd.DataFrame | None = None
        self._ground_truth_significance: pd.DataFrame | None = None

    def benchmark(self) -> None:
        """Fit and run every enabled baseline and the user's predictions through every enabled metric."""
        held_out = [
            p for p in sorted(self.adata_test.obs[self.target_col].astype(str).unique()) if p != self.reference_key
        ]
        if not held_out:
            raise ValueError(f"No held-out perturbations found in `adata_test.obs[{self.target_col!r}]`.")

        baseline_names, predicted_deltas = [], {}
        for field in fields(self.baselines):
            flag = getattr(self.baselines, field.name)
            if not flag:
                continue
            kwargs = flag if isinstance(flag, dict) else {}
            predictor = _BASELINE_CLASSES[field.name](**kwargs)
            predictor.fit(
                self.adata_train,
                target_col=self.target_col,
                reference_key=self.reference_key,
                perturbation_encodings=self.perturbation_encodings,
            )
            predicted_deltas[field.name] = predictor.predict(held_out)
            baseline_names.append(field.name)

        for name, preds in self.predictions.items():
            predicted_deltas[name] = np.asarray(preds)

        true_deltas = self._compute_true_deltas(held_out)
        gene_indices = [self.true_de_gene_indices[p] for p in held_out] if self.true_de_gene_indices else None

        rows: dict[str, dict[str, float | bool]] = {}
        for name, preds in predicted_deltas.items():
            row: dict[str, float | bool] = {"is_baseline": name in baseline_names}
            if self.metrics.delta_correlation:
                kwargs = self.metrics.delta_correlation if isinstance(self.metrics.delta_correlation, dict) else {}
                row["delta_correlation"] = delta_correlation(preds, true_deltas, gene_indices=gene_indices, **kwargs)[
                    "mean"
                ]
            if self.metrics.de_rank_recovery:
                if not self.true_de_gene_indices:
                    raise ValueError("`metrics.de_rank_recovery` requires `true_de_gene_indices`.")
                kwargs = self.metrics.de_rank_recovery if isinstance(self.metrics.de_rank_recovery, dict) else {}
                k = kwargs.get("k", 50)
                row["de_rank_recovery"] = de_rank_recovery(preds, gene_indices, k=k)["mean"]
            if self.metrics.systema_decomposition and len(held_out) >= 2:
                decomposition = systema_decomposition(preds, true_deltas)
                row["systema_shared"] = decomposition["shared"]
                row["systema_specific"] = decomposition["specific"]
            rows[name] = row
        self._results = pd.DataFrame.from_dict(rows, orient="index")

        if self.metrics.combination_additivity:
            pt = import_pertpy()
            pseudobulk = pt.tl.PseudobulkSpace().compute(self.adata_train, target_col=self.target_col, mode="mean")
            self._combination_additivity = combination_additivity(
                pseudobulk, target_col=self.target_col, reference_key=self.reference_key
            )

        if self.metrics.ground_truth_significance:
            self._ground_truth_significance = self._compute_ground_truth_significance()

    def _compute_true_deltas(self, held_out: list[str]) -> NdArray:
        pt = import_pertpy()
        test_labels = self.adata_test.obs[self.target_col].astype(str)
        if self.reference_key in test_labels.unique():
            source = self.adata_test
        else:
            # `adata_test` is documented (see the class docstring) as holding only the
            # held-out perturbation(s), with no requirement that it carry its own control
            # cells — and the reference synthetic train/test split (test = only the "A+B"
            # cells) confirms this is the expected shape. But `compute_control_diff` needs a
            # `reference_key` group to diff against; pseudobulking `adata_test` alone in that
            # case raises `ValueError: Reference key control not found in perturbation`
            # (confirmed while running this task's tests). Borrow control cells from
            # `adata_train`, which always has them (every baseline's `.fit()` requires it),
            # and diff the held-out perturbations against that instead.
            train_labels = self.adata_train.obs[self.target_col].astype(str)
            is_control = train_labels.to_numpy() == self.reference_key
            if not is_control.any():
                raise ValueError(
                    f"No {self.reference_key!r} cells found in `adata_test` or `adata_train`; "
                    "cannot compute true deltas."
                )
            source = anndata.concat([self.adata_train[is_control], self.adata_test])
        ps = pt.tl.PseudobulkSpace()
        pseudobulk = ps.compute(source, target_col=self.target_col, mode="mean")
        ps.compute_control_diff(pseudobulk, target_col=self.target_col, reference_key=self.reference_key, copy=False)
        obs_names = pseudobulk.obs_names.astype(str)
        return np.stack([np.asarray(pseudobulk.X)[obs_names.get_loc(p)] for p in held_out])

    def _compute_ground_truth_significance(self) -> pd.DataFrame:
        test_labels = self.adata_test.obs[self.target_col].astype(str)
        if self.reference_key not in test_labels.unique():
            warnings.warn(
                f"No {self.reference_key!r} cells found in `adata_test`; skipping the ground-truth "
                "significance diagnostic.",
                stacklevel=2,
            )
            return pd.DataFrame(columns=["distance", "pvalue", "pvalue_adj", "significant"])
        pt = import_pertpy()
        # `Distance`/`DistanceTest`'s AnnData-based methods read from `.obsm["X_pca"]` by default
        # when neither `layer_key` nor `obsm_key` is given (confirmed against the installed pertpy:
        # omitting both raises `KeyError: 'X_pca'` on an AnnData with only `.X` populated). Route
        # through a layer instead so this works on raw/normalized expression directly.
        adata_for_test = self.adata_test.copy()
        adata_for_test.layers["_ground_truth_significance_expression"] = adata_for_test.X
        distance_test = pt.tl.DistanceTest(metric="edistance", layer_key="_ground_truth_significance_expression")
        return distance_test(adata_for_test, groupby=self.target_col, contrast=self.reference_key)

    def get_results(self, min_max_scale: bool = True) -> pd.DataFrame:
        """Return the benchmarking results.

        Parameters
        ----------
        min_max_scale
            Whether to min-max scale the score columns (excludes `is_baseline`).

        Returns
        -------
        DataFrame indexed by predictor name, with an `is_baseline` flag column and one column
        per enabled metric.
        """
        if self._results is None:
            raise RuntimeError("Call `.benchmark()` before `.get_results()`.")
        results = self._results.copy()
        if min_max_scale:
            score_cols = [c for c in results.columns if c != "is_baseline"]
            results[score_cols] = MinMaxScaler().fit_transform(results[score_cols])
        return results
