import gc
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

# Sentinel used to detect when spatial_conservation_metrics was not explicitly
# passed so that it can be auto-derived from spatial_key.
_SPATIAL_UNSET: object = object()

_LABELS = "labels"
_BATCH = "batch"
_X_PRE = "X_pre"
_SPATIAL = "spatial"
_X_EXPR = "X_expr_pre"  # pre-integrated PCA stored for niche metrics
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
    "bras": "BRAS",
    "graph_connectivity": "Graph connectivity",
    "pcr_comparison": "PCR comparison",
    # Coordinate preservation
    "spatial_mrre": "MRRE",
    "spatial_knn_overlap": "kNN overlap",
    "spatial_distance_correlation": "Distance corr.",
    "spatial_morans_i": "Moran's I",
    # Niche preservation
    "spatial_niche_knn_overlap": "Niche kNN",
    # Domain boundary
    "spatial_pas": "PAS",
    "spatial_chaos": "CHAOS",
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

    bras: MetricType = True
    ilisi_knn: MetricType = True
    kbet_per_label: MetricType = True
    graph_connectivity: MetricType = True
    pcr_comparison: MetricType = True


@dataclass(frozen=True)
class CoordinatePreservation:
    """Coordinate-preservation metrics: does the latent reproduce XY geometry?

    These metrics compare each model's latent representation directly against
    the physical spot coordinates.  They are most meaningful for spatial graph
    autoencoders (STAGATE-style) whose latent is explicitly trained as a
    surrogate for tissue coordinates.

    For models like scVIVA, resolVI, gimVI, or DestVI — where the latent
    captures expression state, niche structure, denoising, or deconvolution —
    these metrics measure a property the model was not trained to optimise.
    Use :class:`NichePreservation` and :class:`DomainBoundary` instead to
    assess those models on their intended objectives.

    All scores are in ``[0, 1]`` with higher better.

    Metrics (all higher = better):

    * ``spatial_mrre`` — 1 minus normalised mean relative rank error of
      spatial neighbours in the latent space.
    * ``spatial_knn_overlap`` — chance-normalised overlap of spatial vs.
      latent k-NN sets per spot.
    * ``spatial_distance_correlation`` — Spearman correlation of pairwise
      spatial vs. latent distances, rescaled to ``[0, 1]``.
    * ``spatial_morans_i`` — mean Moran's I of latent dimensions using the
      spatial weight graph, rescaled to ``[0, 1]``.
    """

    spatial_mrre: MetricType = True
    spatial_knn_overlap: MetricType = True
    spatial_distance_correlation: MetricType = True
    spatial_morans_i: MetricType = True


# Backward-compatible alias — ``SpatialConservation`` maps to the coordinate-
# preservation axis so that existing code continues to work unchanged.
SpatialConservation = CoordinatePreservation


@dataclass(frozen=True)
class NichePreservation:
    """Niche-preservation metrics: does the latent capture microenvironment?

    Asks whether cells that share a similar local microenvironment (similar
    average expression of spatial neighbours) are also close in latent space.
    This is a direct measure of what niche-aware models such as scVIVA are
    trained to do, and provides a complementary axis to raw coordinate
    preservation.

    Niche features are computed as the mean pre-integrated embedding (PCA)
    of spatial neighbours, keeping them low-dimensional and independent of
    the model being evaluated.

    Metrics (all higher = better):

    * ``spatial_niche_knn_overlap`` — chance-normalised overlap of niche-
      feature k-NN vs. latent k-NN per spot.
    """

    spatial_niche_knn_overlap: MetricType = True


@dataclass(frozen=True)
class DomainBoundary:
    """Domain-boundary metrics: do latent clusters align with tissue domains?

    Clusters the latent embedding with k-means and measures how spatially
    coherent the resulting domains are.  High scores indicate that
    latent-derived clusters occupy compact, contiguous patches of tissue
    rather than scattered fragments — directly relevant for models aimed at
    spatial domain identification.

    Metrics (all higher = better):

    * ``spatial_pas`` — 1 minus Proportion of Abnormal Spots; fraction of
      spatial neighbours in the same cluster (higher = more coherent).
    * ``spatial_chaos`` — 1 minus normalised mean intra-cluster spatial
      distance (higher = more spatially compact clusters).
    """

    spatial_pas: MetricType = True
    spatial_chaos: MetricType = True


class MetricAnnDataAPI(Enum):
    """Specification of the AnnData API for a metric."""

    isolated_labels = lambda ad, fn: fn(ad.X, ad.obs[_LABELS], ad.obs[_BATCH])
    nmi_ari_cluster_labels_leiden = lambda ad, fn: fn(ad.uns["15_neighbor_res"], ad.obs[_LABELS])
    nmi_ari_cluster_labels_kmeans = lambda ad, fn: fn(ad.X, ad.obs[_LABELS])
    silhouette_label = lambda ad, fn: fn(ad.X, ad.obs[_LABELS])
    clisi_knn = lambda ad, fn: fn(ad.uns["90_neighbor_res"], ad.obs[_LABELS])
    graph_connectivity = lambda ad, fn: fn(ad.uns["15_neighbor_res"], ad.obs[_LABELS])
    bras = lambda ad, fn: fn(ad.X, ad.obs[_LABELS], ad.obs[_BATCH])
    pcr_comparison = lambda ad, fn: fn(ad.obsm[_X_PRE], ad.X, ad.obs[_BATCH], categorical=True)
    ilisi_knn = lambda ad, fn: fn(ad.uns["90_neighbor_res"], ad.obs[_BATCH])
    kbet_per_label = lambda ad, fn: fn(ad.uns["50_neighbor_res"], ad.obs[_BATCH], ad.obs[_LABELS])
    # Coordinate preservation — latent embedding vs physical XY coordinates
    spatial_mrre = lambda ad, fn: fn(ad.X, ad.obsm[_SPATIAL])
    spatial_knn_overlap = lambda ad, fn: fn(ad.X, ad.obsm[_SPATIAL])
    spatial_distance_correlation = lambda ad, fn: fn(ad.X, ad.obsm[_SPATIAL])
    spatial_morans_i = lambda ad, fn: fn(ad.X, ad.obsm[_SPATIAL])
    # Niche preservation — latent kNN vs niche-feature kNN (pre-integrated PCA)
    spatial_niche_knn_overlap = lambda ad, fn: fn(ad.X, ad.obsm[_SPATIAL], ad.obsm.get(_X_EXPR))
    # Domain boundary — derived from k-means clustering of latent embedding
    spatial_pas = lambda ad, fn: fn(ad.X, ad.obsm[_SPATIAL])
    spatial_chaos = lambda ad, fn: fn(ad.X, ad.obsm[_SPATIAL])


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
    spatial_conservation_metrics
        Specification of which spatial conservation metrics to run in the pipeline.
        Requires ``spatial_key`` to be set. MRRE, kNN overlap, distance correlation
        and Moran's I compare each embedding against ``adata.obsm[spatial_key]``
        to quantify spatial structure preservation.
    spatial_key
        Key in ``adata.obsm`` that contains the 2-D spatial coordinates (x, y).
        Typically ``"spatial"`` (Squidpy / Scanpy convention).
        When set, spatial conservation metrics are **automatically enabled**
        using the default :class:`SpatialConservation` configuration unless
        ``spatial_conservation_metrics`` is explicitly passed as ``None``
        to opt out.  Spatial metrics are never computed when this is ``None``.
    pre_integrated_embedding_obsm_key
        Obsm key containing a non-integrated embedding of the data. If `None`, the embedding will be computed
        in the prepare step. See the notes below for more information.
    n_jobs
        Number of jobs to use for parallelization of neighbor search.
    progress_bar
        Whether to show a progress bar for :meth:`~scib_metrics.benchmark.Benchmarker.prepare` and
        :meth:`~scib_metrics.benchmark.Benchmarker.benchmark`.
    solver
        SVD solver to use during PCA. can help stability issues. Choose from: "arpack", "randomized" or "auto"

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
        solver: str = "arpack",
        spatial_conservation_metrics: CoordinatePreservation | None = _SPATIAL_UNSET,  # type: ignore[assignment]
        spatial_key: str | None = None,
        spatial_conservation_weight: float = 0.0,
        niche_preservation: NichePreservation | None = None,
        domain_boundary: DomainBoundary | None = None,
    ):
        self._adata = adata
        self._embedding_obsm_keys = embedding_obsm_keys
        self._pre_integrated_embedding_obsm_key = pre_integrated_embedding_obsm_key
        self._bio_conservation_metrics = bio_conservation_metrics
        self._batch_correction_metrics = batch_correction_metrics
        # Auto-enable coordinate preservation when spatial_key is provided and the
        # caller has not explicitly set spatial_conservation_metrics.
        if spatial_conservation_metrics is _SPATIAL_UNSET:
            spatial_conservation_metrics = CoordinatePreservation() if spatial_key is not None else None
        self._spatial_conservation_metrics = spatial_conservation_metrics
        self._niche_preservation = niche_preservation
        self._domain_boundary = domain_boundary
        self._spatial_key = spatial_key
        self._spatial_conservation_weight = spatial_conservation_weight
        self._results = pd.DataFrame(columns=list(self._embedding_obsm_keys) + [_METRIC_TYPE])
        self._emb_adatas = {}
        self._neighbor_values = (15, 50, 90)
        self._prepared = False
        self._benchmarked = False
        self._batch_key = batch_key
        self._label_key = label_key
        self._n_jobs = n_jobs
        self._progress_bar = progress_bar
        self._compute_neighbors = True
        self._solver = solver

        _any_spatial = (
            self._spatial_conservation_metrics is not None
            or self._niche_preservation is not None
            or self._domain_boundary is not None
        )
        if (
            self._bio_conservation_metrics is None
            and self._batch_correction_metrics is None
            and not _any_spatial
        ):
            raise ValueError("At least one of batch, bio, or spatial metrics must be defined.")

        if self._spatial_conservation_metrics is not None and self._spatial_key is None:
            raise ValueError(
                "spatial_key must be provided when spatial_conservation_metrics is set. "
                "Typically this is 'spatial' (adata.obsm['spatial'])."
            )
        if self._niche_preservation is not None and self._spatial_key is None:
            raise ValueError(
                "spatial_key must be provided when niche_preservation is set. "
                "Typically this is 'spatial' (adata.obsm['spatial'])."
            )
        if self._domain_boundary is not None and self._spatial_key is None:
            raise ValueError(
                "spatial_key must be provided when domain_boundary is set. "
                "Typically this is 'spatial' (adata.obsm['spatial'])."
            )

        if self._spatial_key is not None and self._spatial_key not in self._adata.obsm:
            raise ValueError(
                f"spatial_key '{self._spatial_key}' not found in adata.obsm. "
                f"Available keys: {list(self._adata.obsm.keys())}"
            )

        self._metric_collection_dict = {}
        if self._bio_conservation_metrics is not None:
            self._metric_collection_dict.update({"Bio conservation": self._bio_conservation_metrics})
        if self._batch_correction_metrics is not None:
            self._metric_collection_dict.update({"Batch correction": self._batch_correction_metrics})
        if self._spatial_conservation_metrics is not None:
            self._metric_collection_dict.update({"Coordinate preservation": self._spatial_conservation_metrics})
        if self._niche_preservation is not None:
            self._metric_collection_dict.update({"Niche preservation": self._niche_preservation})
        if self._domain_boundary is not None:
            self._metric_collection_dict.update({"Domain boundary": self._domain_boundary})

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
        gc.collect()

        # Compute PCA
        if self._pre_integrated_embedding_obsm_key is None:
            # This is how scib does it
            # https://github.com/theislab/scib/blob/896f689e5fe8c57502cb012af06bed1a9b2b61d2/scib/metrics/pcr.py#L197
            sc.tl.pca(self._adata, svd_solver=self._solver, use_highly_variable=False)
            self._pre_integrated_embedding_obsm_key = "X_pca"

        for emb_key in self._embedding_obsm_keys:
            self._emb_adatas[emb_key] = AnnData(self._adata.obsm[emb_key], obs=self._adata.obs)
            self._emb_adatas[emb_key].obs[_BATCH] = np.asarray(self._adata.obs[self._batch_key].values)
            self._emb_adatas[emb_key].obs[_LABELS] = np.asarray(self._adata.obs[self._label_key].values)
            self._emb_adatas[emb_key].obsm[_X_PRE] = self._adata.obsm[self._pre_integrated_embedding_obsm_key]
            if self._spatial_key is not None:
                self._emb_adatas[emb_key].obsm[_SPATIAL] = np.asarray(self._adata.obsm[self._spatial_key])
                # Store pre-integrated embedding as niche feature proxy for
                # spatial_niche_knn_overlap; set after PCA so the key exists.
                self._emb_adatas[emb_key].obsm[_X_EXPR] = np.asarray(
                    self._adata.obsm[self._pre_integrated_embedding_obsm_key]
                )

        # Compute neighbors
        if self._compute_neighbors:
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
        else:
            warnings.warn(
                "Computing Neighbors Skipped",
                UserWarning,
            )

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
                    gc.collect()
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

    def get_results(self, min_max_scale: bool = False, clean_names: bool = True) -> pd.DataFrame:
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

        # Compute per-category aggregate scores
        per_class_score = df.groupby(_METRIC_TYPE).mean().transpose()

        # Build Total score.  Weights follow the original scIB manuscript
        # (0.4 batch + 0.6 bio).  Spatial axes are averaged across all enabled
        # spatial groups and added with spatial_conservation_weight (default 0.0
        # so they do not affect Total unless explicitly enabled).
        if self._batch_correction_metrics is not None and self._bio_conservation_metrics is not None:
            total = 0.4 * per_class_score["Batch correction"] + 0.6 * per_class_score["Bio conservation"]
            if self._spatial_conservation_weight > 0.0:
                _spatial_groups = [
                    g
                    for g in ("Coordinate preservation", "Niche preservation", "Domain boundary")
                    if g in per_class_score.columns
                ]
                if _spatial_groups:
                    spatial_mean = sum(per_class_score[g] for g in _spatial_groups) / len(_spatial_groups)
                    total = total + self._spatial_conservation_weight * spatial_mean
            per_class_score["Total"] = total

        df = pd.concat([df.transpose(), per_class_score], axis=1)
        df.loc[_METRIC_TYPE, per_class_score.columns] = _AGGREGATE_SCORE
        return df

    def plot_results_table(self, min_max_scale: bool = False, show: bool = True, save_dir: str | None = None) -> Table:
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

        def _fmt(x: float) -> str:
            """Format to 2 d.p., mapping -0.00 → 0.00."""
            v = round(float(x), 2)
            return "0.00" if v == 0.0 else f"{v:.2f}"

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
        elif self._bio_conservation_metrics is not None:
            sort_col = "Bio conservation"
        else:
            # Only spatial conservation — no sensible ranking across embeddings
            sort_col = plot_df.columns[0]
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
                width=1.5,
                textprops={
                    "ha": "center",
                    "bbox": {"boxstyle": "circle", "pad": 0.25},
                },
                cmap=cmap_fn(plot_df[col]),
                group=df.loc[_METRIC_TYPE, col],
                formatter=_fmt,
            )
            for i, col in enumerate(other_cols)
        ]
        # Bars for the aggregate scores
        column_definitions += [
            ColumnDefinition(
                col,
                width=1.5,
                title=col.replace(" ", "\n", 1),
                plot_fn=bar,
                plot_kw={
                    "cmap": mpl.cm.YlGnBu,
                    "plot_bg_bar": False,
                    "annotate": True,
                    "height": 0.9,
                    "formatter": _fmt,
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
