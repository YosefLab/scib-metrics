import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

import scib_metrics
from scib_metrics.nearest_neighbors import NeighborsResults


def _make_neighbors(X_emb: np.ndarray, n_neighbors: int = 30) -> NeighborsResults:
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_emb)
    dists, inds = nbrs.kneighbors(X_emb)
    return NeighborsResults(indices=inds[:, 1:], distances=dists[:, 1:])


def test_sbee_returns_float_in_range():
    rng = np.random.default_rng(42)
    X_emb = rng.normal(size=(100, 10))
    batches = rng.integers(0, 3, size=100).astype(str)
    labels = rng.integers(0, 4, size=100).astype(str)
    knn = _make_neighbors(X_emb)
    score = scib_metrics.sbee(knn, X_emb, batches, labels)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_sbee_well_mixed_higher_than_separated():
    rng = np.random.default_rng(0)
    n = 200

    # Well-mixed: both batches drawn from the same distribution
    X_mixed = rng.normal(loc=0, scale=1.0, size=(n, 10))
    batches_mixed = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    labels_mixed = np.array(["T1"] * n)

    # Separated: batch A and B in completely different regions
    X_sep = np.vstack(
        [
            rng.normal(loc=0, scale=0.1, size=(n // 2, 10)),
            rng.normal(loc=10, scale=0.1, size=(n // 2, 10)),
        ]
    )
    batches_sep = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    labels_sep = np.array(["T1"] * n)

    score_mixed = scib_metrics.sbee(_make_neighbors(X_mixed), X_mixed, batches_mixed, labels_mixed)
    score_sep = scib_metrics.sbee(_make_neighbors(X_sep), X_sep, batches_sep, labels_sep)

    assert score_mixed > score_sep


def test_sbee_sensitivity_produces_valid_scores():
    rng = np.random.default_rng(42)
    X_emb = rng.normal(size=(100, 10))
    batches = rng.integers(0, 3, size=100).astype(str)
    labels = rng.integers(0, 4, size=100).astype(str)
    knn = _make_neighbors(X_emb)

    for sensitivity in [0.01, 0.15, 1.0]:
        score = scib_metrics.sbee(knn, X_emb, batches, labels, sensitivity=sensitivity)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_sbee_input_validation_batches():
    rng = np.random.default_rng(42)
    X_emb = rng.normal(size=(100, 10))
    knn = _make_neighbors(X_emb)
    batches = rng.integers(0, 2, size=100).astype(str)
    labels = rng.integers(0, 3, size=100).astype(str)

    with pytest.raises(ValueError, match="batches"):
        scib_metrics.sbee(knn, X_emb, batches[:50], labels)


def test_sbee_input_validation_labels():
    rng = np.random.default_rng(42)
    X_emb = rng.normal(size=(100, 10))
    knn = _make_neighbors(X_emb)
    batches = rng.integers(0, 2, size=100).astype(str)
    labels = rng.integers(0, 3, size=100).astype(str)

    with pytest.raises(ValueError, match="labels"):
        scib_metrics.sbee(knn, X_emb, batches, labels[:50])


def test_sbee_cell_type_in_single_batch():
    """Cover the fallback where a cell type appears in only one batch (inter = intra)."""
    rng = np.random.default_rng(7)
    # "T2" cells exist only in batch "A" — no inter-batch neighbours possible
    X_emb = rng.normal(size=(60, 10))
    batches = np.array(["A"] * 40 + ["B"] * 20)
    labels = np.array(["T1"] * 20 + ["T2"] * 20 + ["T1"] * 20)
    knn = _make_neighbors(X_emb, n_neighbors=10)
    score = scib_metrics.sbee(knn, X_emb, batches, labels)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_sbee_input_validation_x_emb():
    rng = np.random.default_rng(42)
    X_emb = rng.normal(size=(100, 10))
    knn = _make_neighbors(X_emb)
    batches = rng.integers(0, 2, size=100).astype(str)
    labels = rng.integers(0, 3, size=100).astype(str)

    with pytest.raises(ValueError, match="X_emb"):
        scib_metrics.sbee(knn, X_emb[:50], batches, labels)
