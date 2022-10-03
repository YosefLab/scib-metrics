from importlib.metadata import version

from .silhouette import silhouette_samples

__all__ = ["silhouette_samples"]

__version__ = version("scib-metrics")
