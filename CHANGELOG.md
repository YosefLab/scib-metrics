# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## 0.6.0 (unreleased)

## 0.5.5 (2025-06-03)

### Added

- Add batch removal adapted silhouette (BRAS) metric ({func}`scib_metrics.metrics.bras`) {pr}`197`, which addresses limitations of silhouette for scoring batch effect removal.
- Add cosine distance implementation required for BRAS.

### Changed

- Changed {func}`scib_metrics.utils.cdist` to support cosine distance.
- Changed silhouette-related functions to be compatible with adaptions required for BRAS.

## 0.5.4 (2025-04-23)

### Fixed

- Apply default values for benchmarker metrics {pr}`203`.

## 0.5.3 (2025-02-17)

### Removed

- Reverted a change that was needed for scib-autotune in scvi-tools {pr}`189`.

## 0.5.2 (2025-02-13)

### Added

- Add `progress_bar` argument to {class}`scib_metrics.benchmark.Benchmarker` {pr}`152`.
- Add ability of {class}`scib_metrics.benchmark.Benchmarker` plotting code to handle missing sets of metrics {pr}`181`.
- Add random score in case of aggregate metrics not selected to be used in scib autotune in scvi-tools, {pr}`188`.

### Changed

- Changed Leiden clustering now has a seed argument for reproducibility {pr}`173`.
- Changed passing `None` to `bio_conservation_metrics` or `batch_correction_metrics` in {class}`scib_metrics.benchmark.Benchmarker` now implies to skip this set of metrics {pr}`181`.

### Fixed

- Fix neighbors connectivities in test to use new scanpy fn {pr}`170`.
- Fix Kmeans test {pr}`172`.
- Fix deprecation and future warnings {pr}`171`.
- Fix lisi return type and docstring {pr}`182`.

## 0.5.1 (2024-02-23)

### Changed

- Replace removed {class}`jax.random.KeyArray` with {class}`jax.Array` {pr}`135`.

## 0.5.0 (2024-01-04)

### Changed

- Refactor all relevant metrics to use `NeighborsResults` as input instead of sparse
  distance/connectivity matrices {pr}`129`.

## 0.4.1 (2023-10-08)

### Fixed

- Fix KMeans. All previous versions had a bug with KMeans and ARI/NMI metrics are not reliable
  with this clustering {pr}`115`.

## 0.4.0 (2023-09-19)

### Added

- Update isolated labels to use newest scib methodology {pr}`108`.

### Fixed

- Fix jax one-hot error {pr}`107`.

### Removed

- Drop Python 3.8 {pr}`107`.

## 0.3.3 (2023-03-29)

### Fixed

- Large scale tutorial now properly uses gpu index {pr}`92`

## 0.3.2 (2023-03-13)

### Changed

- Switch to Ruff for linting/formatting {pr}`87`
- Update cookiecutter template {pr}`88`

## 0.3.1 (2023-02-16)

### Changed

- Expose chunk size for silhouette {pr}`82`

## 0.3.0 (2023-02-16)

### Changed

- Rename `KmeansJax` to `Kmeans` and fix ++ initialization, use Kmeans as default in benchmarker instead of Leiden {pr}`81`.
- Warn about joblib, add progress bar postfix str {pr}`80`

## 0.2.0 (2023-02-02)

### Added

- Allow custom nearest neighbors methods in Benchmarker {pr}`78`.

## 0.1.1 (2023-01-04)

### Added

- Add new tutorial and fix scalability of lisi {pr}`71`.

## 0.1.0 (2023-01-03)

### Added

- Add benchmarking pipeline with plotting {pr}`52` {pr}`69`.

### Fixed

- Fix diffusion distance computation, affecting kbet {pr}`70`.

## 0.0.9 (2022-12-16)

### Added

- Add kbet {pr}`60`.
- Add graph connectivty metric {pr}`61`.

## 0.0.8 (2022-11-18)

### Changed

- Switch to random kmeans initialization due to kmeans++ complexity issues {pr}`54`.

### Fixed

- Begin fixes to make kmeans++ initialization faster {pr}`49`.

## 0.0.7 (2022-10-31)

### Changed

- Move PCR to utils module in favor of PCR comparison {pr}`46`.

### Fixed

- Fix memory issue in `KMeansJax` by using `_kmeans_full_run` with `map` instead of `vmap` {pr}`45`.

## 0.0.6 (2022-10-25)

### Changed

- Reimplement silhouette in a memory constant way, pdist using lax scan {pr}`42`.

## 0.0.5 (2022-10-24)

### Added

- Standardize language of docstring {pr}`30`.
- Use K-means++ initialization {pr}`23`.
- Add pc regression and pc comparsion {pr}`16` {pr}`38`.
- Lax'd silhouette {pr}`33`.
- Cookicutter template sync {pr}`35`.

## 0.0.4 (2022-10-10)

### Added

- NMI/ARI metric with Leiden clustering resolution optimization {pr}`24`.
- iLISI/cLISI metrics {pr}`20`.

## 0.0.1 - 0.0.3

See the [GitHub releases][https://github.com/yoseflab/scib-metrics/releases] for details.
