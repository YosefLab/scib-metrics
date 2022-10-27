# API

## Metrics

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
    ilisi_knn
    clisi_knn
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
    utils.KMeansJax
    utils.pca
    utils.principal_component_regression
    utils.one_hot
    utils.compute_simpson_index
```

## Settings

An instance of the {class}`~scib_metrics._settings.ScibConfig` is available as `scib_metrics.settings` and allows configuring scib_metrics.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   _settings.ScibConfig
```
