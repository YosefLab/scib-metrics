# scib-metrics

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/scverse/scvi-tools/test.yml?branch=main
[link-tests]: https://github.com/yoseflab/scib-metrics/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/scib-metrics

Accelerated and Python-only metrics for benchmarking single-cell integration outputs.

This package contains implementations of metrics for evaluating the performance of single-cell omics data integration methods. The implementations of these metrics use [jax](https://jax.readthedocs.io/en/latest/) when possible for jit-compilation and hardware acceleration. All implementations are in Python.

Currently we are porting metrics used in the scIB [manuscript](https://www.nature.com/articles/s41592-021-01336-8) (and [code](https://github.com/theislab/scib)). Deviations from the original implementations are documented. However, metric values from this repository should not be compared to the scIB repository.

## Getting started

Please refer to the [documentation][link-docs].

## Installation

You need to have Python 3.8 or newer installed on your system. If you don't have
Python installed, we recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

There are several alternative options to install scib-metrics:

<!--
1) Install the latest release of `scib-metrics` from `PyPI <https://pypi.org/project/scib-metrics/>`_:

```bash
pip install scib-metrics
```
-->

1. Install the latest release on PyPI:

```bash
pip install scib-metrics
```

2. Install the latest development version:

```bash
pip install git+https://github.com/yoseflab/scib-metrics.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
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
[issue-tracker]: https://github.com/yoseflab/scib-metrics/issues
[changelog]: https://scib-metrics.readthedocs.io/latest/changelog.html
[link-docs]: https://scib-metrics.readthedocs.io
[link-api]: https://scib-metrics.readthedocs.io/latest/api.html
