def test_public_api_is_importable():
    from scib_metrics.perturbation import (
        AdditiveBaseline,
        LinearBaseline,
        MeanBaseline,
        PerturbationBaselines,
        PerturbationBenchmarker,
        PerturbationMetrics,
        combination_additivity,
        de_rank_recovery,
        delta_correlation,
        systema_decomposition,
    )

    assert all(
        callable(obj)
        for obj in (
            AdditiveBaseline,
            LinearBaseline,
            MeanBaseline,
            PerturbationBaselines,
            PerturbationBenchmarker,
            PerturbationMetrics,
            combination_additivity,
            de_rank_recovery,
            delta_correlation,
            systema_decomposition,
        )
    )
