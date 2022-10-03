from importlib.metadata import version

from . import utils
from ._ari_nmi import nmi_ari_cluster_labels
from ._silhouette import silhouette_batch, silhouette_label

__all__ = [
    "utils",
    "silhouette_label",
    "silhouette_batch",
    "nmi_ari_cluster_labels",
]

__version__ = version("scib-metrics")
