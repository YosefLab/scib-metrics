import logging
from importlib.metadata import version

from . import utils
from ._ari_nmi import nmi_ari_cluster_labels_kmeans, nmi_ari_cluster_labels_leiden
from ._isolated_labels import isolated_labels
from ._lisi import clisi_knn, ilisi_knn, lisi_knn
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
    "nmi_ari_cluster_labels_kmeans",
    "nmi_ari_cluster_labels_leiden",
]

__version__ = version("scib-metrics")

settings.verbosity = logging.INFO
# Jax sets the root logger, this prevents double output.
logger = logging.getLogger("scib_metrics")
logger.propagate = False
