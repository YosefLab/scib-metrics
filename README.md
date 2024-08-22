# scib-metrics

[![Stars][badge-stars]][link-stars]
[![PyPI][badge-pypi]][link-pypi]
[![PyPIDownloads][badge-downloads]][link-downloads]
[![Docs][badge-docs]][link-docs]
[![Build][badge-build]][link-build]
[![Coverage][badge-cov]][link-cov]
[![Discourse][badge-discourse]][link-discourse]
[![Chat][badge-zulip]][link-zulip]

[badge-stars]: https://img.shields.io/github/stars/YosefLab/scib-metrics?logo=GitHub&color=yellow
[link-stars]: https://github.com/YosefLab/scib-metrics/stargazers
[badge-pypi]: https://img.shields.io/pypi/v/scib-metrics.svg
[link-pypi]: https://pypi.org/project/scib-metrics
[badge-downloads]: https://static.pepy.tech/badge/scib-metrics
[link-downloads]: https://pepy.tech/project/scib-metrics
[badge-docs]: https://readthedocs.org/projects/scib-metrics/badge/?version=latest
[link-docs]: https://scib-metrics.readthedocs.io/en/latest/?badge=latest
[badge-build]: https://github.com/YosefLab/scib-metrics/actions/workflows/build.yaml/badge.svg
[link-build]: https://github.com/YosefLab/scib-metrics/actions/workflows/build.yaml/
[badge-cov]: https://codecov.io/gh/YosefLab/scib-metrics/branch/main/graph/badge.svg
[link-cov]: https://codecov.io/gh/YosefLab/scib-metrics
[badge-discourse]: https://img.shields.io/discourse/posts?color=yellow&logo=discourse&server=https%3A%2F%2Fdiscourse.scverse.org
[link-discourse]: https://discourse.scverse.org/
[badge-zulip]: https://img.shields.io/badge/zulip-join_chat-brightgreen.svg
[link-zulip]: https://scverse.zulipchat.com/

Accelerated and Python-only metrics for benchmarking single-cell integration outputs.

This package contains implementations of metrics for evaluating the performance of single-cell omics data integration methods. The implementations of these metrics use [JAX](https://jax.readthedocs.io/en/latest/) when possible for jit-compilation and hardware acceleration. All implementations are in Python.

Currently we are porting metrics used in the scIB [manuscript](https://www.nature.com/articles/s41592-021-01336-8) (and [code](https://github.com/theislab/scib)). Deviations from the original implementations are documented. However, metric values from this repository should not be compared to the scIB repository.

## Getting started

Please refer to the [documentation][link-docs].

## Installation

You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

There are several options to install scib-metrics:

1. Install the latest release on PyPI:

```bash
pip install scib-metrics
```

2. Install the latest development version:

```bash
pip install git+https://github.com/yoseflab/scib-metrics.git@main
```

To leverage hardware acceleration (e.g., GPU) please install the apprpriate version of [JAX](https://github.com/google/jax#installation) separately. Often this can be easier by using conda-distributed versions of JAX.

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse Discourse][link-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

References for individual metrics can be found in the corresponding documentation. This package is heavily inspired by the single-cell integration benchmarking work:

```
@article{luecken2022benchmarking,
  title={Benchmarking atlas-level data integration in single-cell genomics},
  author={Luecken, Malte D and B{\"u}ttner, Maren and Chaichoompu, Kridsadakorn and Danese, Anna and Interlandi, Marta and M{\"u}ller, Michaela F and Strobl, Daniel C and Zappia, Luke and Dugas, Martin and Colom{\'e}-Tatch{\'e}, Maria and others},
  journal={Nature methods},
  volume={19},
  number={1},
  pages={41--50},
  year={2022},
  publisher={Nature Publishing Group}
}
```

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/YosefLab/scib-metrics/issues
[changelog]: https://scib-metrics.readthedocs.io/en/latest/changelog.html
[link-api]: https://scib-metrics.readthedocs.io/en/latest/api.html
