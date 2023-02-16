import warnings
from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
from jax.tree_util import tree_map

from .._types import NdArray


def _anderson_ksamp_midrank(
    samples: Sequence[jnp.ndarray], Z: jnp.ndarray, Zstar: jnp.ndarray, k: int, n: jnp.ndarray, N: int
):
    """Compute A2akN equation 7 of Scholz and Stephens.

    Parameters
    ----------
    samples
        Array of sample arrays.
    Z
        Sorted array of all observations.
    Zstar
        Sorted array of unique observations.
    k
        Number of samples.
    n
        Number of observations in each sample.
    N
        Total number of observations.

    Returns
    -------
    A2aKN
        The A2aKN statistics of Scholz and Stephens 1987.
    """
    A2akN = 0.0
    Z_ssorted_left = Z.searchsorted(Zstar, "left")
    if N == Zstar.size:
        lj = 1.0
    else:
        lj = Z.searchsorted(Zstar, "right") - Z_ssorted_left
    Bj = Z_ssorted_left + lj / 2.0
    for i in jnp.arange(0, k):
        s = jnp.sort(samples[i])
        s_ssorted_right = s.searchsorted(Zstar, side="right")
        Mij = s_ssorted_right.astype(float)
        fij = s_ssorted_right - s.searchsorted(Zstar, "left")
        Mij -= fij / 2.0
        inner = lj / float(N) * (N * Mij - Bj * n[i]) ** 2 / (Bj * (N - Bj) - N * lj / 4.0)
        A2akN += inner.sum() / n[i]
    A2akN *= (N - 1.0) / N
    return A2akN


def _anderson_ksamp_right(
    samples: Sequence[jnp.ndarray], Z: jnp.ndarray, Zstar: jnp.ndarray, k: int, n: jnp.ndarray, N: int
):
    """Compute A2akN equation 6 of Scholz & Stephens.

    Parameters
    ----------
    samples
        Array of sample arrays.
    Z
        Sorted array of all observations.
    Zstar
        Sorted array of unique observations.
    k
        Number of samples.
    n
        Number of observations in each sample.
    N
        Total number of observations.

    Returns
    -------
    A2KN
        The A2KN statistics of Scholz and Stephens 1987.
    """
    A2kN = 0.0
    lj = Z.searchsorted(Zstar[:-1], "right") - Z.searchsorted(Zstar[:-1], "left")
    Bj = lj.cumsum()
    for i in jnp.arange(0, k):
        s = jnp.sort(samples[i])
        Mij = s.searchsorted(Zstar[:-1], side="right")
        inner = lj / float(N) * (N * Mij - Bj * n[i]) ** 2 / (Bj * (N - Bj))
        A2kN += inner.sum() / n[i]
    return A2kN


@dataclass
class Anderson_ksampResult:
    """Result of `anderson_ksamp`.

    Attributes
    ----------
    statistic : float
        Normalized k-sample Anderson-Darling test statistic.
    critical_values : array
        The critical values for significance levels 25%, 10%, 5%, 2.5%, 1%,
        0.5%, 0.1%.
    significance_level : float
        The approximate p-value of the test. The value is floored / capped
        at 0.1% / 25%.
    """

    statistic: float
    critical_values: jnp.ndarray
    significance_level: float


def anderson_ksamp(samples: Sequence[NdArray], midrank: bool = True) -> Anderson_ksampResult:
    """Jax implementation of :func:`scipy.stats.anderson_ksamp`.

    The k-sample Anderson-Darling test is a modification of the
    one-sample Anderson-Darling test. It tests the null hypothesis
    that k-samples are drawn from the same population without having
    to specify the distribution function of that population. The
    critical values depend on the number of samples.

    Parameters
    ----------
    samples
        Array of sample data in arrays.
    midrank
        Type of Anderson-Darling test which is computed. Default
        (True) is the midrank test applicable to continuous and
        discrete populations. If False, the right side empirical
        distribution is used.

    Returns
    -------
    result

    Raises
    ------
    ValueError
        If less than 2 samples are provided, a sample is empty, or no
        distinct observations are in the samples.

    Notes
    -----
    [1]_ defines three versions of the k-sample Anderson-Darling test:
    one for continuous distributions and two for discrete
    distributions, in which ties between samples may occur. The
    default of this routine is to compute the version based on the
    midrank empirical distribution function. This test is applicable
    to continuous and discrete data. If midrank is set to False, the
    right side empirical distribution is used for a test for discrete
    data. According to [1]_, the two discrete test statistics differ
    only slightly if a few collisions due to round-off errors occur in
    the test not adjusted for ties between samples.
    The critical values corresponding to the significance levels from 0.01
    to 0.25 are taken from [1]_. p-values are floored / capped
    at 0.1% / 25%. Since the range of critical values might be extended in
    future releases, it is recommended not to test ``p == 0.25``, but rather
    ``p >= 0.25`` (analogously for the lower bound).

    References
    ----------
    .. [1] Scholz, F. W and Stephens, M. A. (1987), K-Sample
           Anderson-Darling Tests, Journal of the American Statistical
           Association, Vol. 82, pp. 918-924.
    """
    k = len(samples)
    if k < 2:
        raise ValueError("anderson_ksamp needs at least two samples")

    samples = tree_map(jnp.asarray, samples)
    Z = jnp.sort(jnp.hstack(samples))
    N = jnp.array(Z.size)
    Zstar = jnp.unique(Z)
    if Zstar.size < 2:
        raise ValueError("anderson_ksamp needs more than one distinct " "observation")

    n = jnp.array([sample.size for sample in samples])
    if jnp.any(n == 0):
        raise ValueError("anderson_ksamp encountered sample without " "observations")

    if midrank:
        A2kN = _anderson_ksamp_midrank(samples, Z, Zstar, k, n, N)
    else:
        A2kN = _anderson_ksamp_right(samples, Z, Zstar, k, n, N)

    H = (1.0 / n).sum()
    hs_cs = (1.0 / jnp.arange(N - 1, 1, -1)).cumsum()
    h = hs_cs[-1] + 1
    g = (hs_cs / jnp.arange(2, N)).sum()

    a = (4 * g - 6) * (k - 1) + (10 - 6 * g) * H
    b = (2 * g - 4) * k**2 + 8 * h * k + (2 * g - 14 * h - 4) * H - 8 * h + 4 * g - 6
    c = (6 * h + 2 * g - 2) * k**2 + (4 * h - 4 * g + 6) * k + (2 * h - 6) * H + 4 * h
    d = (2 * h + 6) * k**2 - 4 * h * k
    sigmasq = (a * N**3 + b * N**2 + c * N + d) / ((N - 1.0) * (N - 2.0) * (N - 3.0))
    m = k - 1
    A2 = (A2kN - m) / jnp.sqrt(sigmasq)

    # The b_i values are the interpolation coefficients from Table 2
    # of Scholz and Stephens 1987
    b0 = jnp.array([0.675, 1.281, 1.645, 1.96, 2.326, 2.573, 3.085])
    b1 = jnp.array([-0.245, 0.25, 0.678, 1.149, 1.822, 2.364, 3.615])
    b2 = jnp.array([-0.105, -0.305, -0.362, -0.391, -0.396, -0.345, -0.154])
    critical = b0 + b1 / jnp.sqrt(m) + b2 / m

    sig = jnp.array([0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001])
    if A2 < critical.min():
        p = sig.max()
        warnings.warn(f"p-value capped: true value larger than {p}", stacklevel=2)
    elif A2 > critical.max():
        p = sig.min()
        warnings.warn(f"p-value floored: true value smaller than {p}", stacklevel=2)
    else:
        # interpolation of probit of significance level
        pf = jnp.polyfit(critical, jnp.log(sig), 2)
        p = jnp.exp(jnp.polyval(pf, A2))

    res = Anderson_ksampResult(A2, critical, p)
    return res
