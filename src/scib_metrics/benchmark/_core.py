from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, List, Optional, Union

import pandas as pd
import scanpy as sc
from anndata import AnnData

import scib_metrics

_LABELS = "labels"
_BATCH = "batch"
_X_PRE = "X_pre"


@dataclass
class BioConvervation:
    """Specification of bio conservation metrics to run in the pipeline.

    Metrics can be included using a boolean flag. Custom keyword args can be
    used by passing a partial callable of that metric here.
    """

    isolated_labels: Union[bool, Callable] = True
    nmi_ari_cluster_labels_leiden: Union[bool, Callable] = True
    nmi_ari_cluster_labels_kmeans: Union[bool, Callable] = False
    silhouette_label: Union[bool, Callable] = True
    clisi_knn: Union[bool, Callable] = True


@dataclass
class BatchCorrection:
    """Specification of which batch correction metrics to run in the pipeline.

    Metrics can be included using a boolean flag. Custom keyword args can be
    used by passing a partial callable of that metric here.
    """

    silhouette_batch: Union[bool, Callable] = True
    pcr_comparison: Union[bool, Callable] = True
    ilisi_knn: Union[bool, Callable] = True
    kbet_per_label: Union[bool, Callable] = True


class MetricAnnDataAPI(Enum):
    """Specification of the AnnData API for a metric."""

    isolated_labels: lambda ad, fn: fn(ad.X, ad.obs[_LABELS], ad.obs[_BATCH])
    nmi_ari_cluster_labels_leiden: lambda ad, fn: fn(ad.obsp["15_connectivities"], ad.obs[_LABELS])
    nmi_ari_cluster_labels_kmeans: lambda ad, fn: fn(ad.X, ad.obs[_LABELS])
    silhouette_label: lambda ad, fn: fn(ad.X, ad.obs[_LABELS])
    clisi_knn: lambda ad, fn: fn(ad.obsp["90_distances"], ad.obs[_LABELS])
    silhouette_batch: lambda ad, fn: fn(ad.X, ad.obs[_LABELS], ad.obs[_BATCH])
    pcr_comparison: lambda ad, fn: fn(ad.obsm[_X_PRE], ad.X, ad.obs[_BATCH], categorical=True)
    ilisi_knn: lambda ad, fn: fn(ad.obsp["90_distances"], ad.obs[_BATCH])
    kbet_per_label: lambda ad, fn: fn(ad.obsp["50_connectivities"], ad.obs[_BATCH], ad.obs[_LABELS])


class Benchmarker:
    """Benchmarking pipeline for the single-cell integration task.

    Parameters
    ----------
    adata
        AnnData object containing the raw count data and integrated embeddings as obsm keys.
    batch_key
        Key in `adata.obs` that contains the batch information.
    label_key
        Key in `adata.obs` that contains the cell type labels.
    embedding_obsm_keys
        List of obsm keys that contain the embeddings to be benchmarked.
    bio_conservation_metrics
        Specification of which bio conservation metrics to run in the pipeline.
    batch_correction_metrics
        Specification of which batch correction metrics to run in the pipeline.
    pre_integrated_embedding_obsm_key
        Obsm key containing a non-integrated embedding of the data. If `None`, the embedding will be computed
        in the prepare step. See the notes below for more information.

    Notes
    -----
    `adata.X` should contain a form of the data that is not integrated, but is normalized. The `prepare` method will
    use `adata.X` for PCA via :func:`~scanpy.tl.pca`, which also only uses features masked via `adata.var['highly_variable']`.
    """

    def __init__(
        self,
        adata: AnnData,
        batch_key: str,
        label_key: str,
        embedding_obsm_keys: List[str],
        bio_conservation_metrics: Optional[BioConvervation] = None,
        batch_correction_metrics: Optional[BatchCorrection] = None,
        pre_integrated_embedding_obsm_key: Optional[str] = None,
    ):
        self._adata = adata
        self._embedding_obsm_keys = embedding_obsm_keys
        self._pre_integrated_embedding_obsm_key = pre_integrated_embedding_obsm_key
        self._bio_conservation_metrics = bio_conservation_metrics if bio_conservation_metrics else BioConvervation()
        self._batch_correction_metrics = batch_correction_metrics if batch_correction_metrics else BatchCorrection()
        self._results = pd.DataFrame(columns=self._embedding_obsm_keys)
        self._emb_adatas = {}
        self._neighbor_values = (15, 50, 90)
        self._prepared = False

        for emb_key in self._embedding_obsm_keys:
            self._emb_adatas[emb_key] = AnnData(self._adata.obsm[emb_key], obs=self._adata.obs)
            self._emb_adatas[emb_key].obs[_BATCH] = adata.obs[batch_key]
            self._emb_adatas[emb_key].obs[_LABELS] = adata.obs[label_key]

    def prepare(self) -> None:
        """Prepare the data for benchmarking."""
        # Compute neighbors
        # TODO: only compute largest n neighbors and subset
        # This will need to rerun the distances -> connectivies for each neighbor subset
        for ad in self._emb_adatas.values():
            for n in self._neighbor_values:
                sc.pp.neighbors(ad, use_rep="X", n_neighbors=n, key_added=f"{n}")

        # Compute PCA
        sc.tl.pca(self._adata)

        self._prepared = True

    def benchmark(self) -> None:
        """Run the pipeline."""
        for emb_key, ad in self._emb_adatas.items():
            for metric_collection in (self._bio_conservation_metrics, self._batch_correction_metrics):
                for metric_name, use_metric in asdict(metric_collection).items():
                    if use_metric:
                        if isinstance(metric_name, str):
                            metric_fn = getattr(scib_metrics, metric_name)
                        else:
                            # Callable in this case
                            metric_fn = use_metric
                        metric_value = MetricAnnDataAPI[metric_name].value(ad, metric_fn)
                        self._results.loc[metric_name, emb_key] = metric_value
