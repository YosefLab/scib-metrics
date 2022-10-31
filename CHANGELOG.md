# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## 0.0.7 (2022-10-31)

-   Fix memory issue in `KMeansJax` by using `_kmeans_full_run` with `map` instead of `vmap` ([#45][])
-   Move PCR to utils module in favor of PCR comparison ([#46][])

[#45]: https://github.com/YosefLab/scib-metrics/pull/45
[#46]: https://github.com/YosefLab/scib-metrics/pull/46

## 0.0.6 (2022-10-25)

-   Reimplement silhouette in a memory constant way, pdist using lax scan ([#42][])

[#42]: https://github.com/YosefLab/scib-metrics/pull/42

## 0.0.5 (2022-10-24)

### Added

-   Standardize language of docstring ([#30][])
-   Use K-means++ initialization ([#23][])
-   Add pc regression and pc comparsion ([#16][] and [#38][])
-   Lax'd silhouette ([#33][])
-   Cookicutter template sync ([#35][])

[#33]: https://github.com/YosefLab/scib-metrics/pull/33
[#38]: https://github.com/YosefLab/scib-metrics/pull/38
[#35]: https://github.com/YosefLab/scib-metrics/pull/35
[#16]: https://github.com/YosefLab/scib-metrics/pull/16
[#23]: https://github.com/YosefLab/scib-metrics/pull/23
[#30]: https://github.com/YosefLab/scib-metrics/pull/30

## 0.0.4 (2022-10-10)

### Added

-   NMI/ARI metric with Leiden clustering resolution optimization ([#24][])
-   iLISI/cLISI metrics ([#20][])

[#20]: https://github.com/YosefLab/scib-metrics/pull/20
[#24]: https://github.com/YosefLab/scib-metrics/pull/24

## 0.0.1 - 0.0.3

See the [GitHub releases][https://github.com/yoseflab/scib-metrics/releases] for details.
