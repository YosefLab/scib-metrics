import logging
from importlib.metadata import version

from . import utils
from ._graph_connectivity import graph_connectivity
from ._isolated_labels import isolated_labels
from ._kbet import kbet, kbet_per_label
from ._lisi import clisi_knn, ilisi_knn, lisi_knn
from ._nmi_ari import nmi_ari_cluster_labels_kmeans, nmi_ari_cluster_labels_leiden
from ._pcr_comparison import pcr_comparison
from ._settings import settings
from ._silhouette import silhouette_batch, silhouette_label

__all__ = [
    "utils",
    "isolated_labels",
    "pcr_comparison",
    "silhouette_label",
    "silhouette_batch",
    "ilisi_knn",
    "clisi_knn",
    "lisi_knn",
    "nmi_ari_cluster_labels_kmeans",
    "nmi_ari_cluster_labels_leiden",
    "kbet",
    "kbet_per_label",
    "graph_connectivity",
]

__version__ = version("scib-metrics")

settings.verbosity = logging.INFO
# Jax sets the root logger, this prevents double output.
logger = logging.getLogger("scib_metrics")
logger.propagate = False
