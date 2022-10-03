from importlib.metadata import version

from . import utils
from ._silhouette import silhouette_batch, silhouette_label

__all__ = ["utils", "silhouette_label", "silhouette_batch"]

__version__ = version("scib-metrics")
