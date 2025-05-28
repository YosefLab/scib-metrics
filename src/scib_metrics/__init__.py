import logging
from importlib.metadata import version

from . import nearest_neighbors, utils
from .metrics import (
    graph_connectivity,
    isolated_labels,
    kbet,
    kbet_per_label,
    clisi_knn,
    ilisi_knn,
    lisi_knn,
    nmi_ari_cluster_labels_kmeans,
    nmi_ari_cluster_labels_leiden,
    pcr_comparison,
    silhouette_batch,
    silhouette_label,
    bras,
)
from ._settings import settings

__all__ = [
    "utils",
    "nearest_neighbors",
    "isolated_labels",
    "pcr_comparison",
    "silhouette_label",
    "silhouette_batch",
    "bras",
    "ilisi_knn",
    "clisi_knn",
    "lisi_knn",
    "nmi_ari_cluster_labels_kmeans",
    "nmi_ari_cluster_labels_leiden",
    "kbet",
    "kbet_per_label",
    "graph_connectivity",
    "settings",
]

__version__ = version("scib-metrics")

settings.verbosity = logging.INFO
# Jax sets the root logger, this prevents double output.
logger = logging.getLogger("scib_metrics")
logger.propagate = False
