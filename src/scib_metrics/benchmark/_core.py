from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import pandas as pd
from anndata import AnnData


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


class Benchmarker:
    """Benchmarking pipeline for the single-cell integration task.

    Parameters
    ----------
    adata
        AnnData object containing the raw count data and integrated embeddings as obsm keys.
    embedding_obsm_keys
        List of obsm keys that contain the embeddings to be benchmarked.
    pre_integrated_embedding_obsm_key
        Obsm key containing a non-integrated embedding of the data.
    bio_conservation_metrics
        Specification of which bio conservation metrics to run in the pipeline.
    batch_correction_metrics
        Specification of which batch correction metrics to run in the pipeline.
    """

    def __init__(
        self,
        adata: AnnData,
        embedding_obsm_keys: List[str],
        pre_integrated_embedding_obsm_key: str,
        bio_conservation_metrics: Optional[BioConvervation] = None,
        batch_correction_metrics: Optional[BatchCorrection] = None,
    ):
        self._adata = adata
        self._embedding_obsm_keys = embedding_obsm_keys
        self._pre_integrated_embedding_obsm_key = pre_integrated_embedding_obsm_key
        self._bio_conservation_metrics = bio_conservation_metrics if bio_conservation_metrics else BioConvervation()
        self._batch_correction_metrics = batch_correction_metrics if batch_correction_metrics else BatchCorrection()
        self._results = pd.DataFrame(columns=self._embedding_obsm_keys)

    def run(self) -> None:
        """Run the pipeline."""
        for _ in self._embedding_obsm_keys:
            continue
            # getattr from dataclasses or use the callable directly

            # adata.obsm[emb_key_] = adata.obsm[emb_key]
            # X = adata.obsm[emb_key_]
            # sc.tl.pca(adata)
            # X_pre = adata.obsm["X_pca"]
            # labels = np.array(adata.obs[label_key].astype("category").cat.codes).ravel()
            # batch = np.array(adata.obs[batch_key].astype("category").cat.codes).ravel()
            # sc.pp.neighbors(adata, use_rep=emb_key_)
            # graph_conn = adata.obsp["connectivities"]
            # df = pd.DataFrame(index=[model_name])
            # df["nmi_kmeans"], df["ari_kmeans"] = nmi_ari_cluster_labels_kmeans(X, labels)
            # df["nmi_leiden"], df["ari_leiden"] = nmi_ari_cluster_labels_leiden(graph_conn, labels, n_jobs=8)
            # df["sil_batch"] = silhouette_batch(X, labels, batch)
            # df["sil_labels"] = silhouette_label(X, labels)
            # df["isolated_labels"] = isolated_labels(X, labels, batch)
            # sc.pp.neighbors(adata, use_rep=emb_key_, n_neighbors=90)
            # graph_dist = adata.obsp["distances"]
            # df["ilisi"] = ilisi_knn(graph_dist, batch)
            # df["clisi"] = clisi_knn(graph_dist, labels)
            # df["pcr"] = pcr_comparison(X_pre, X, batch, categorical=True)
            # return df
