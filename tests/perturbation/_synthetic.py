"""Synthetic perturbation AnnData for testing scib_metrics.perturbation."""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def make_synthetic_perturbation_adata(
    n_genes: int = 20,
    n_cells_per_group: int = 30,
    seed: int = 0,
    target_col: str = "perturbation",
    reference_key: str = "control",
) -> AnnData:
    """Build a synthetic cell-level AnnData with a control group and perturbations A, B, A+B.

    Ground truth: perturbation `A` shifts genes `[0:5]` by `+3`, `B` shifts genes `[5:10]`
    by `+2`, and `A+B` is exactly the sum of the two shifts (perfectly additive), so an
    additive predictor should recover it exactly while a predictor blind to combination
    structure should not.

    Parameters
    ----------
    n_genes
        Number of genes.
    n_cells_per_group
        Number of cells per perturbation group.
    seed
        Random seed for the noise added on top of each group's shift.
    target_col
        `.obs` column name holding the perturbation label.
    reference_key
        Perturbation label marking control cells.

    Returns
    -------
    AnnData of shape `(4 * n_cells_per_group, n_genes)` with groups
    `[reference_key, "A", "B", "A+B"]`.
    """
    rng = np.random.default_rng(seed)
    shift_a = np.zeros(n_genes)
    shift_a[0:5] = 3.0
    shift_b = np.zeros(n_genes)
    shift_b[5:10] = 2.0
    shifts = {
        reference_key: np.zeros(n_genes),
        "A": shift_a,
        "B": shift_b,
        "A+B": shift_a + shift_b,
    }
    rows, labels = [], []
    for name, shift in shifts.items():
        base = rng.normal(loc=10.0, scale=1.0, size=(n_cells_per_group, n_genes))
        rows.append(base + shift)
        labels.extend([name] * n_cells_per_group)
    adata = AnnData(X=np.concatenate(rows, axis=0).astype(np.float32))
    adata.obs[target_col] = labels
    adata.obs[target_col] = adata.obs[target_col].astype("category")
    return adata
