import logging
from importlib.metadata import version

from . import utils
from ._ari_nmi import nmi_ari_cluster_labels
from ._isolated_labels import isolated_labels
from ._settings import settings
from ._silhouette import silhouette_batch, silhouette_label

__all__ = [
    "utils",
    "isolated_labels",
    "silhouette_label",
    "silhouette_batch",
    "nmi_ari_cluster_labels",
]

__version__ = version("scib-metrics")

settings.verbosity = logging.INFO
# Jax sets the root logger, this prevents double output.
logger = logging.getLogger("scib_metrics")
logger.propagate = False
