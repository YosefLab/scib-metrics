# API

## Benchmarking pipeline

Import as:

```
from scib_metrics.benchmark import Benchmarker
```

```{eval-rst}
.. module:: scib_metrics.benchmark
.. currentmodule:: scib_metrics

.. autosummary::
    :toctree: generated

    benchmark.Benchmarker
    benchmark.BioConservation
    benchmark.BatchCorrection
```

## Metrics

Import as:

```
import scib_metrics
scib_metrics.ilisi_knn(...)
```

```{eval-rst}
.. module:: scib_metrics
.. currentmodule:: scib_metrics

.. autosummary::
    :toctree: generated

    isolated_labels
    nmi_ari_cluster_labels_kmeans
    nmi_ari_cluster_labels_leiden
    pcr_comparison
    silhouette_label
    silhouette_batch
    bras
    ilisi_knn
    clisi_knn
    kbet
    kbet_per_label
    graph_connectivity
```

## Utils

```{eval-rst}
.. module:: scib_metrics.utils
.. currentmodule:: scib_metrics

.. autosummary::
    :toctree: generated

    utils.cdist
    utils.pdist_squareform
    utils.silhouette_samples
    utils.KMeans
    utils.pca
    utils.principal_component_regression
    utils.one_hot
    utils.compute_simpson_index
    utils.convert_knn_graph_to_idx
    utils.check_square
    utils.diffusion_nn
```

### Nearest neighbors

```{eval-rst}
.. module:: scib_metrics.nearest_neighbors
.. currentmodule:: scib_metrics

.. autosummary::
    :toctree: generated

    nearest_neighbors.pynndescent
    nearest_neighbors.jax_approx_min_k
    nearest_neighbors.NeighborsResults
```

## Settings

An instance of the {class}`~scib_metrics._settings.ScibConfig` is available as `scib_metrics.settings` and allows configuring scib_metrics.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   _settings.ScibConfig
```
