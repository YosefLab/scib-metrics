import os
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import scib_metrics
from scib_metrics.nearest_neighbors import NeighborsResults, pynndescent

Kwargs = dict[str, Any]
MetricType = bool | Kwargs

_LABELS = "labels"
_BATCH = "batch"
_X_PRE = "X_pre"
_METRIC_TYPE = "Metric Type"
_AGGREGATE_SCORE = "Aggregate score"

# Mapping of metric fn names to clean DataFrame column names
metric_name_cleaner = {
    "silhouette_label": "Silhouette label",
    "silhouette_batch": "Silhouette batch",
    "isolated_labels": "Isolated labels",
    "nmi_ari_cluster_labels_leiden_nmi": "Leiden NMI",
    "nmi_ari_cluster_labels_leiden_ari": "Leiden ARI",
    "nmi_ari_cluster_labels_kmeans_nmi": "KMeans NMI",
    "nmi_ari_cluster_labels_kmeans_ari": "KMeans ARI",
    "clisi_knn": "cLISI",
    "ilisi_knn": "iLISI",
    "kbet_per_label": "KBET",
    "graph_connectivity": "Graph connectivity",
    "pcr_comparison": "PCR comparison",
}


@dataclass(frozen=True)
class BioConservation:
    """Specification of bio conservation metrics to run in the pipeline.

    Metrics can be included using a boolean flag. Custom keyword args can be
    used by passing a dictionary here. Keyword args should not set data-related
    parameters, such as `X` or `labels`.
    """

    isolated_labels: MetricType = True
    nmi_ari_cluster_labels_leiden: MetricType = False
    nmi_ari_cluster_labels_kmeans: MetricType = True
    silhouette_label: MetricType = True
    clisi_knn: MetricType = True


@dataclass(frozen=True)
class BatchCorrection:
    """Specification of which batch correction metrics to run in the pipeline.

    Metrics can be included using a boolean flag. Custom keyword args can be
    used by passing a dictionary here. Keyword args should not set data-related
    parameters, such as `X` or `labels`.
    """

    silhouette_batch: MetricType = True
    ilisi_knn: MetricType = True
    kbet_per_label: MetricType = True
    graph_connectivity: MetricType = True
    pcr_comparison: MetricType = True


class MetricAnnDataAPI(Enum):
    """Specification of the AnnData API for a metric."""

    isolated_labels = lambda ad, fn: fn(ad.X, ad.obs[_LABELS], ad.obs[_BATCH])
    nmi_ari_cluster_labels_leiden = lambda ad, fn: fn(ad.uns["15_neighbor_res"], ad.obs[_LABELS])
    nmi_ari_cluster_labels_kmeans = lambda ad, fn: fn(ad.X, ad.obs[_LABELS])
    silhouette_label = lambda ad, fn: fn(ad.X, ad.obs[_LABELS])
    clisi_knn = lambda ad, fn: fn(ad.uns["90_neighbor_res"], ad.obs[_LABELS])
    graph_connectivity = lambda ad, fn: fn(ad.uns["15_neighbor_res"], ad.obs[_LABELS])
    silhouette_batch = lambda ad, fn: fn(ad.X, ad.obs[_LABELS], ad.obs[_BATCH])
    pcr_comparison = lambda ad, fn: fn(ad.obsm[_X_PRE], ad.X, ad.obs[_BATCH], categorical=True)
    ilisi_knn = lambda ad, fn: fn(ad.uns["90_neighbor_res"], ad.obs[_BATCH])
    kbet_per_label = lambda ad, fn: fn(ad.uns["50_neighbor_res"], ad.obs[_BATCH], ad.obs[_LABELS])


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
    n_jobs
        Number of jobs to use for parallelization of neighbor search.
    progress_bar
        Whether to show a progress bar for :meth:`~scib_metrics.benchmark.Benchmarker.prepare` and
        :meth:`~scib_metrics.benchmark.Benchmarker.benchmark`.

    Notes
    -----
    `adata.X` should contain a form of the data that is not integrated, but is normalized. The `prepare` method will
    use `adata.X` for PCA via :func:`~scanpy.tl.pca`, which also only uses features masked via `adata.var['highly_variable']`.

    See further usage examples in the following tutorial:

    1. :doc:`/notebooks/lung_example`
    """

    def __init__(
        self,
        adata: AnnData,
        batch_key: str,
        label_key: str,
        embedding_obsm_keys: list[str],
        bio_conservation_metrics: BioConservation | None = BioConservation(),
        batch_correction_metrics: BatchCorrection | None = BatchCorrection(),
        pre_integrated_embedding_obsm_key: str | None = None,
        n_jobs: int = 1,
        progress_bar: bool = True,
    ):
        self._adata = adata
        self._embedding_obsm_keys = embedding_obsm_keys
        self._pre_integrated_embedding_obsm_key = pre_integrated_embedding_obsm_key
        self._bio_conservation_metrics = bio_conservation_metrics
        self._batch_correction_metrics = batch_correction_metrics
        self._results = pd.DataFrame(columns=list(self._embedding_obsm_keys) + [_METRIC_TYPE])
        self._emb_adatas = {}
        self._neighbor_values = (15, 50, 90)
        self._prepared = False
        self._benchmarked = False
        self._batch_key = batch_key
        self._label_key = label_key
        self._n_jobs = n_jobs
        self._progress_bar = progress_bar

        if self._bio_conservation_metrics is None and self._batch_correction_metrics is None:
            raise ValueError("Either batch or bio metrics must be defined.")

        self._metric_collection_dict = {}
        if self._bio_conservation_metrics is not None:
            self._metric_collection_dict.update({"Bio conservation": self._bio_conservation_metrics})
        if self._batch_correction_metrics is not None:
            self._metric_collection_dict.update({"Batch correction": self._batch_correction_metrics})

    def prepare(self, neighbor_computer: Callable[[np.ndarray, int], NeighborsResults] | None = None) -> None:
        """Prepare the data for benchmarking.

        Parameters
        ----------
        neighbor_computer
            Function that computes the neighbors of the data. If `None`, the neighbors will be computed
            with :func:`~scib_metrics.utils.nearest_neighbors.pynndescent`. The function should take as input
            the data and the number of neighbors to compute and return a :class:`~scib_metrics.utils.nearest_neighbors.NeighborsResults`
            object.
        """
        # Compute PCA
        if self._pre_integrated_embedding_obsm_key is None:
            # This is how scib does it
            # https://github.com/theislab/scib/blob/896f689e5fe8c57502cb012af06bed1a9b2b61d2/scib/metrics/pcr.py#L197
            sc.tl.pca(self._adata, use_highly_variable=False)
            self._pre_integrated_embedding_obsm_key = "X_pca"

        for emb_key in self._embedding_obsm_keys:
            self._emb_adatas[emb_key] = AnnData(self._adata.obsm[emb_key], obs=self._adata.obs)
            self._emb_adatas[emb_key].obs[_BATCH] = np.asarray(self._adata.obs[self._batch_key].values)
            self._emb_adatas[emb_key].obs[_LABELS] = np.asarray(self._adata.obs[self._label_key].values)
            self._emb_adatas[emb_key].obsm[_X_PRE] = self._adata.obsm[self._pre_integrated_embedding_obsm_key]

        # Compute neighbors
        progress = self._emb_adatas.values()
        if self._progress_bar:
            progress = tqdm(progress, desc="Computing neighbors")

        for ad in progress:
            if neighbor_computer is not None:
                neigh_result = neighbor_computer(ad.X, max(self._neighbor_values))
            else:
                neigh_result = pynndescent(
                    ad.X, n_neighbors=max(self._neighbor_values), random_state=0, n_jobs=self._n_jobs
                )
            for n in self._neighbor_values:
                ad.uns[f"{n}_neighbor_res"] = neigh_result.subset_neighbors(n=n)

        self._prepared = True

    def benchmark(self) -> None:
        """Run the pipeline."""
        if self._benchmarked:
            warnings.warn(
                "The benchmark has already been run. Running it again will overwrite the previous results.",
                UserWarning,
            )

        if not self._prepared:
            self.prepare()

        num_metrics = sum(
            [sum([v is not False for v in asdict(met_col)]) for met_col in self._metric_collection_dict.values()]
        )

        progress_embs = self._emb_adatas.items()
        if self._progress_bar:
            progress_embs = tqdm(self._emb_adatas.items(), desc="Embeddings", position=0, colour="green")

        for emb_key, ad in progress_embs:
            pbar = None
            if self._progress_bar:
                pbar = tqdm(total=num_metrics, desc="Metrics", position=1, leave=False, colour="blue")
            for metric_type, metric_collection in self._metric_collection_dict.items():
                for metric_name, use_metric_or_kwargs in asdict(metric_collection).items():
                    if use_metric_or_kwargs:
                        pbar.set_postfix_str(f"{metric_type}: {metric_name}") if pbar is not None else None
                        metric_fn = getattr(scib_metrics, metric_name)
                        if isinstance(use_metric_or_kwargs, dict):
                            # Kwargs in this case
                            metric_fn = partial(metric_fn, **use_metric_or_kwargs)
                        metric_value = getattr(MetricAnnDataAPI, metric_name)(ad, metric_fn)
                        # nmi/ari metrics return a dict
                        if isinstance(metric_value, dict):
                            for k, v in metric_value.items():
                                self._results.loc[f"{metric_name}_{k}", emb_key] = v
                                self._results.loc[f"{metric_name}_{k}", _METRIC_TYPE] = metric_type
                        else:
                            self._results.loc[metric_name, emb_key] = metric_value
                            self._results.loc[metric_name, _METRIC_TYPE] = metric_type
                        pbar.update(1) if pbar is not None else None

        self._benchmarked = True

    def get_results(self, min_max_scale: bool = True, clean_names: bool = True) -> pd.DataFrame:
        """Return the benchmarking results.

        Parameters
        ----------
        min_max_scale
            Whether to min max scale the results.
        clean_names
            Whether to clean the metric names.

        Returns
        -------
        The benchmarking results.
        """
        df = self._results.transpose()
        df.index.name = "Embedding"
        df = df.loc[df.index != _METRIC_TYPE]
        if min_max_scale:
            # Use sklearn to min max scale
            df = pd.DataFrame(
                MinMaxScaler().fit_transform(df),
                columns=df.columns,
                index=df.index,
            )
        if clean_names:
            df = df.rename(columns=metric_name_cleaner)
        df = df.transpose()
        df[_METRIC_TYPE] = self._results[_METRIC_TYPE].values

        # Compute scores
        per_class_score = df.groupby(_METRIC_TYPE).mean().transpose()
        # This is the default scIB weighting from the manuscript
        if self._batch_correction_metrics is not None and self._bio_conservation_metrics is not None:
            per_class_score["Total"] = (
                0.4 * per_class_score["Batch correction"] + 0.6 * per_class_score["Bio conservation"]
            )
        df = pd.concat([df.transpose(), per_class_score], axis=1)
        df.loc[_METRIC_TYPE, per_class_score.columns] = _AGGREGATE_SCORE
        return df

    def plot_results_table(self, min_max_scale: bool = True, show: bool = True, save_dir: str | None = None) -> Table:
        """Plot the benchmarking results.

        Parameters
        ----------
        min_max_scale
            Whether to min max scale the results.
        show
            Whether to show the plot.
        save_dir
            The directory to save the plot to. If `None`, the plot is not saved.
        """
        num_embeds = len(self._embedding_obsm_keys)
        cmap_fn = lambda col_data: normed_cmap(col_data, cmap=mpl.cm.PRGn, num_stds=2.5)
        df = self.get_results(min_max_scale=min_max_scale)
        # Do not want to plot what kind of metric it is
        plot_df = df.drop(_METRIC_TYPE, axis=0)
        # Sort by total score
        if self._batch_correction_metrics is not None and self._bio_conservation_metrics is not None:
            sort_col = "Total"
        elif self._batch_correction_metrics is not None:
            sort_col = "Batch correction"
        else:
            sort_col = "Bio conservation"
        plot_df = plot_df.sort_values(by=sort_col, ascending=False).astype(np.float64)
        plot_df["Method"] = plot_df.index

        # Split columns by metric type, using df as it doesn't have the new method col
        score_cols = df.columns[df.loc[_METRIC_TYPE] == _AGGREGATE_SCORE]
        other_cols = df.columns[df.loc[_METRIC_TYPE] != _AGGREGATE_SCORE]
        column_definitions = [
            ColumnDefinition("Method", width=1.5, textprops={"ha": "left", "weight": "bold"}),
        ]
        # Circles for the metric values
        column_definitions += [
            ColumnDefinition(
                col,
                title=col.replace(" ", "\n", 1),
                width=1,
                textprops={
                    "ha": "center",
                    "bbox": {"boxstyle": "circle", "pad": 0.25},
                },
                cmap=cmap_fn(plot_df[col]),
                group=df.loc[_METRIC_TYPE, col],
                formatter="{:.2f}",
            )
            for i, col in enumerate(other_cols)
        ]
        # Bars for the aggregate scores
        column_definitions += [
            ColumnDefinition(
                col,
                width=1,
                title=col.replace(" ", "\n", 1),
                plot_fn=bar,
                plot_kw={
                    "cmap": mpl.cm.YlGnBu,
                    "plot_bg_bar": False,
                    "annotate": True,
                    "height": 0.9,
                    "formatter": "{:.2f}",
                },
                group=df.loc[_METRIC_TYPE, col],
                border="left" if i == 0 else None,
            )
            for i, col in enumerate(score_cols)
        ]
        # Allow to manipulate text post-hoc (in illustrator)
        with mpl.rc_context({"svg.fonttype": "none"}):
            fig, ax = plt.subplots(figsize=(len(df.columns) * 1.25, 3 + 0.3 * num_embeds))
            tab = Table(
                plot_df,
                cell_kw={
                    "linewidth": 0,
                    "edgecolor": "k",
                },
                column_definitions=column_definitions,
                ax=ax,
                row_dividers=True,
                footer_divider=True,
                textprops={"fontsize": 10, "ha": "center"},
                row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
                col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
                column_border_kw={"linewidth": 1, "linestyle": "-"},
                index_col="Method",
            ).autoset_fontcolors(colnames=plot_df.columns)
        if show:
            plt.show()
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "scib_results.svg"), facecolor=ax.get_facecolor(), dpi=300)

        return tab
