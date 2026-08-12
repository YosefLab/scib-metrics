import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pertpy")

from scib_metrics.perturbation._core import PerturbationBaselines, PerturbationBenchmarker, PerturbationMetrics
from tests.perturbation._synthetic import make_synthetic_perturbation_adata


def _make_train_test_split():
    adata = make_synthetic_perturbation_adata()
    train = adata[adata.obs["perturbation"].isin(["control", "A", "B"])].copy()
    test = adata[adata.obs["perturbation"] == "A+B"].copy()
    return train, test


def _perturbation_encodings():
    # `LinearBaseline` (Task 5) hard-requires a `perturbation_encodings` entry for every
    # trained and held-out name whenever it's enabled (`PerturbationBaselines.linear=True`
    # by default) -- see its docstring ("Required if `baselines.linear` is enabled."). A
    # multi-hot encoding over the base perturbations ("A", "B") that composes additively for
    # combinations ("A+B" = "A" | "B") is the natural minimal encoding for this synthetic
    # fixture, and lets `LinearBaseline` generalize to the held-out combination.
    return {"A": np.array([1.0, 0.0]), "B": np.array([0.0, 1.0]), "A+B": np.array([1.0, 1.0])}


def test_benchmarker_runs_default_baselines_and_metrics():
    train, test = _make_train_test_split()
    benchmarker = PerturbationBenchmarker(train, test, perturbation_encodings=_perturbation_encodings())
    benchmarker.benchmark()
    results = benchmarker.get_results(min_max_scale=False)
    assert isinstance(results, pd.DataFrame)
    assert set(results.index) == {"mean", "additive", "linear"}
    assert results.loc["mean", "is_baseline"]
    assert "delta_correlation" in results.columns
    assert "systema_shared" not in results.columns  # only 1 held-out perturbation: A+B


def test_benchmarker_includes_user_predictions_alongside_baselines():
    train, test = _make_train_test_split()

    user_predictions = {"my_model": np.zeros((1, 20))}
    benchmarker = PerturbationBenchmarker(
        train, test, predictions=user_predictions, perturbation_encodings=_perturbation_encodings()
    )
    benchmarker.benchmark()
    results = benchmarker.get_results(min_max_scale=False)
    assert "my_model" in results.index
    assert not results.loc["my_model", "is_baseline"]


def test_benchmarker_can_disable_a_baseline():
    train, test = _make_train_test_split()
    benchmarker = PerturbationBenchmarker(train, test, baselines=PerturbationBaselines(linear=False))
    benchmarker.benchmark()
    results = benchmarker.get_results(min_max_scale=False)
    assert "linear" not in results.index


def test_benchmarker_raises_before_get_results_without_benchmark():
    train, test = _make_train_test_split()
    benchmarker = PerturbationBenchmarker(train, test)
    with pytest.raises(RuntimeError):
        benchmarker.get_results()


def test_benchmarker_de_rank_recovery_requires_true_de_gene_indices():
    train, test = _make_train_test_split()
    benchmarker = PerturbationBenchmarker(
        train,
        test,
        metrics=PerturbationMetrics(de_rank_recovery=True),
        perturbation_encodings=_perturbation_encodings(),
    )
    with pytest.raises(ValueError, match="true_de_gene_indices"):
        benchmarker.benchmark()


def test_benchmarker_raises_on_prediction_baseline_collision():
    train, test = _make_train_test_split()
    user_predictions = {"mean": np.zeros((1, 20))}
    benchmarker = PerturbationBenchmarker(
        train,
        test,
        predictions=user_predictions,
        perturbation_encodings=_perturbation_encodings(),
    )
    with pytest.raises(ValueError, match="collide with enabled baseline names"):
        benchmarker.benchmark()


def test_get_ground_truth_significance_requires_flag_enabled():
    train, test = _make_train_test_split()
    benchmarker = PerturbationBenchmarker(train, test, baselines=PerturbationBaselines(linear=False))
    benchmarker.benchmark()
    with pytest.raises(RuntimeError):
        benchmarker.get_ground_truth_significance()


def test_get_ground_truth_significance_warns_without_control_cells_in_test():
    train, test = _make_train_test_split()  # test split has no control cells
    benchmarker = PerturbationBenchmarker(
        train,
        test,
        baselines=PerturbationBaselines(linear=False),
        metrics=PerturbationMetrics(ground_truth_significance=True),
    )
    with pytest.warns(UserWarning, match="skipping"):
        benchmarker.benchmark()
    result = benchmarker.get_ground_truth_significance()
    assert result.empty


def test_get_ground_truth_significance_runs_with_control_cells_in_test():
    adata = make_synthetic_perturbation_adata()
    train = adata[adata.obs["perturbation"].isin(["control", "A", "B"])].copy()
    test = adata[adata.obs["perturbation"].isin(["control", "A+B"])].copy()
    benchmarker = PerturbationBenchmarker(
        train,
        test,
        baselines=PerturbationBaselines(linear=False),
        metrics=PerturbationMetrics(ground_truth_significance=True),
    )
    benchmarker.benchmark()
    result = benchmarker.get_ground_truth_significance()
    assert "pvalue" in result.columns


def test_get_combination_additivity_raises_before_benchmark():
    train, test = _make_train_test_split()
    benchmarker = PerturbationBenchmarker(train, test, perturbation_encodings=_perturbation_encodings())
    with pytest.raises(RuntimeError):
        benchmarker.get_combination_additivity()


def test_get_combination_additivity_finds_combination_present_only_in_test():
    # train only has the trained singles ("A", "B"); the combination "A+B" is held out in
    # `test`. `combination_additivity` must pseudobulk the *union* of train and test to ever
    # see a combination-named perturbation at all.
    train, test = _make_train_test_split()
    benchmarker = PerturbationBenchmarker(train, test, perturbation_encodings=_perturbation_encodings())
    benchmarker.benchmark()
    result = benchmarker.get_combination_additivity()
    assert not result.empty
    assert "A+B" in result.index


def test_delta_correlation_unaffected_by_true_de_gene_indices_when_de_rank_recovery_disabled():
    train, test = _make_train_test_split()
    gene_indices = {"A+B": np.arange(5)}

    benchmarker_without = PerturbationBenchmarker(
        train,
        test,
        perturbation_encodings=_perturbation_encodings(),
        metrics=PerturbationMetrics(de_rank_recovery=False),
    )
    benchmarker_without.benchmark()
    results_without = benchmarker_without.get_results(min_max_scale=False)

    benchmarker_with = PerturbationBenchmarker(
        train,
        test,
        perturbation_encodings=_perturbation_encodings(),
        true_de_gene_indices=gene_indices,
        metrics=PerturbationMetrics(de_rank_recovery=False),
    )
    benchmarker_with.benchmark()
    results_with = benchmarker_with.get_results(min_max_scale=False)

    pd.testing.assert_series_equal(
        results_without["delta_correlation"], results_with["delta_correlation"], check_names=True
    )


def test_systema_decomposition_warns_when_fewer_than_two_held_out_perturbations():
    # The default fixture's test split holds out only "A+B" -- a single perturbation --
    # while `metrics.systema_decomposition` defaults to `True`.
    train, test = _make_train_test_split()
    benchmarker = PerturbationBenchmarker(train, test, perturbation_encodings=_perturbation_encodings())
    with pytest.warns(UserWarning, match="systema_decomposition"):
        benchmarker.benchmark()
    results = benchmarker.get_results(min_max_scale=False)
    assert "systema_shared" not in results.columns
    assert "systema_specific" not in results.columns


def test_plot_results_table_returns_a_table():
    from plottable import Table

    train, test = _make_train_test_split()
    benchmarker = PerturbationBenchmarker(train, test, baselines=PerturbationBaselines(linear=False))
    benchmarker.benchmark()
    table = benchmarker.plot_results_table(show=False)
    assert isinstance(table, Table)


def test_perturbation_benchmarker():
    # Mirrors `tests/test_benchmarker.py::test_benchmarker`: run the full default pipeline
    # (all baselines, all default-on metrics) end-to-end on synthetic data and plot it, the
    # same way that test exercises `Benchmarker` with `BatchCorrection()`/`BioConservation()`.
    #
    # Unlike `_make_train_test_split()` (which holds out only "A+B" and is used by the other,
    # narrower tests in this file), this split holds out *two* perturbations -- "B" and "A+B"
    # -- so `systema_decomposition` (needs >=2 held-out perturbations) actually runs instead
    # of warning and skipping, and "A+B" is a real combination in the train+test union so
    # `combination_additivity` finds a non-empty result instead of warning and returning empty.
    adata = make_synthetic_perturbation_adata()
    train = adata[adata.obs["perturbation"].isin(["control", "A"])].copy()
    test = adata[adata.obs["perturbation"].isin(["B", "A+B"])].copy()
    encodings = {"A": np.array([1.0, 0.0]), "B": np.array([0.0, 1.0]), "A+B": np.array([1.0, 1.0])}

    benchmarker = PerturbationBenchmarker(train, test, perturbation_encodings=encodings)
    benchmarker.benchmark()

    results = benchmarker.get_results()
    assert isinstance(results, pd.DataFrame)
    assert set(results.index) == {"mean", "additive", "linear"}
    assert "delta_correlation" in results.columns
    assert "systema_shared" in results.columns
    assert "systema_specific" in results.columns

    combinations = benchmarker.get_combination_additivity()
    assert "A+B" in combinations.index

    # `show=True` opens a real, blocking GUI window (confirmed identical to
    # `Benchmarker.plot_results_table`'s behavior) -- unreliable to depend on inside a test
    # runner. `save_dir` sidesteps that: it writes a real, inspectable SVG regardless of
    # backend/runner quirks. Open /tmp/perturbation_results.svg after running this test to
    # visually verify the plot (colors, baseline-row labeling, etc.).
    benchmarker.plot_results_table(show=False, save_dir="/tmp")
