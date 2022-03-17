import math

import numpy as np

from .intervals import _interval_to_intervals


def norm_ccdf_approx(x):
    """
    Approximated CCDF (right tail probability) of a normal distribution.
    Bryc, W. 'A uniform approximation to the right normal tail integral' (2002)

    Args:
        x (float): Return the value at `x`. This value has to be positive.

    Returns:
        float: CCDF value at `x`.

    Raises:
        ValueError: If `x` is negative value.
    """
    if x < 0:
        raise ValueError("x has to be positive.")
    if np.isinf(x):
        return 0.0

    num = x ** 2 + 5.575192695 * x + 12.77436324
    denom = (
        math.sqrt(2 * math.pi) * (x ** 3)
        + 14.38718147 * (x ** 2)
        + 31.53531977 * x
        + 2 * 12.77436324
    )
    return (num / denom) * math.exp(-(x ** 2) / 2)


def t_ccdf_approx(x, df):
    """
    Approximated CCDF (right tail probability) of a t distribution.
    Ling, R. F. 'A study of the accuracy of some approximations for t, χ^2, and f tail
    probabilities' (1978)

    Args:
        x (float): Return the value at `x`. This value has to be positive.
        df (float): Degree of freedom.

    Returns:
        float: CCDF value at `x`.
    """
    z = (df - 2 / 3 + 1 / (10 * df)) * math.sqrt(
        math.log(1 + x ** 2 / df) / (df - 5 / 6)
    )
    return norm_ccdf_approx(z)


def chi2_ccdf_approx(x, df):
    """
    Approximated CCDF (right tail probability) of a chi-squared distribution.
    Canal, L. 'A normal approximation for the chi-square distribution' (2005)

    Args:
        x (float): Return the value at `x`.
        df (float): Degree of freedom.

    Returns:
        float: CCDF value at `x`.
    """
    if x <= 0:
        return 1.0
    if np.isinf(x):
        return 0.0

    z = (x / df) ** (1 / 6) - (x / df) ** (1 / 3) / 2 + (x / df) ** (1 / 2) / 3
    mean = 5 / 6 - 1 / (9 * df) - 7 / (648 * (df ** 2)) + 25 / (2187 * (df ** 3))
    var = 1 / (18 * df) + 1 / (162 * (df ** 2)) - 37 / (11664 * (df ** 3))
    z_norm = (z - mean) / math.sqrt(var)

    if z_norm >= 0:
        return norm_ccdf_approx(z_norm)
    else:
        return 1 - norm_ccdf_approx(-z_norm)


def f_ccdf_approx(x, df1, df2):
    """
    Approximated CCDF (right tail probability) of a F distribution.
    Ling, R. F. 'A study of the accuracy of some approximations for t, χ^2, and f tail
    probabilities' (1978)

    Args:
        x (float): Return the value at `x`.
        df1 (float): Degree of freedom.
        df2 (float): Degree of freedom.

    Returns:
        float: CCDF value at `x`.
    """
    if x <= 0:
        return 1.0
    if np.isinf(x):
        return 0.0

    R = (df1 - 1) / 2
    S = (df2 - 1) / 2
    N = (df1 + df2 - 2) / 2
    P = df2 / (df1 * x + df2)
    Q = 1 - P
    D = (
        S
        + 1 / 6
        - (N + 1 / 3) * P
        + 0.04 * (Q / df2 - P / df1 + (Q - 0.5) / (df1 + df2))
    )

    def g(z):
        if 0 < z < 1:
            return (1 - z ** 2 + 2 * z * math.log(z)) / (1 - z ** 2)
        elif z > 1:
            return (-(z ** 2) + 1 + 2 * z * math.log(z)) / (z ** 2 - 1)
        elif z == 0:
            return 1
        elif z == 1:
            return 0

    z = D * math.sqrt(
        (1 + Q * g(S / (N * P)) + P * g(R / (N * Q))) / ((N + 1 / 6) * P * Q)
    )

    if z >= 0:
        return norm_ccdf_approx(z)
    else:
        return 1 - norm_ccdf_approx(-z)


def _truncated_cdf_from_ccdf(ccdf_func, x, intervals):
    """
    Calculate CDF of a truncated distribution from a CCDF function.

    Args:
        ccdf_func (callable): CCDF function of a distribution.
        x (float): Return the value at `x`.
        intervals (array-like): Truncation intervals [[L1, U1], [L2, U2],...].

    Returns:
        float: CDF value at `x`.

    Raises:
        ValueError: If the value `x` is not located inside the `intervals`.
    """
    num = denom = 0
    inside_flag = False

    new_intervals = []
    for lower, upper in intervals:
        if lower < 0 and 0 < upper:
            new_intervals.append([lower, 0])
            new_intervals.append([0, upper])
        else:
            new_intervals.append([lower, upper])

    for lower, upper in new_intervals:
        if lower >= 0:
            diff = ccdf_func(lower) - ccdf_func(upper)
            denom += diff
            if lower <= x <= upper:
                num += ccdf_func(lower) - ccdf_func(x)
                inside_flag = True
            elif upper < x:
                num += diff
        elif upper <= 0:
            diff = ccdf_func(-upper) - ccdf_func(-lower)
            denom += diff
            if lower <= x <= upper:
                num += ccdf_func(-x) - ccdf_func(-lower)
                inside_flag = True
            elif upper < x:
                num += diff

    if not inside_flag:
        raise ValueError(f"Value x={x} is outside the intervals={intervals}")

    return num / denom


def tn_cdf_approx(x, interval):
    """
    Approximated CDF of a truncated normal distribution.

    Args:
        x (float): Return the value at `x`.
        interval (array-like): Truncation interval [L, U] or intervals
            [[L1, U1], [L2, U2],...].

    Returns:
        float: CDF value at `x`.
    """
    intervals = _interval_to_intervals(interval)
    return _truncated_cdf_from_ccdf(norm_ccdf_approx, x, intervals)


def tt_cdf_approx(x, interval, df):
    """
    Approximated CDF of a truncated t distribution.

    Args:
        x (float): Return the value at `x`.
        interval (array-like): Truncation interval [L, U] or intervals
            [[L1, U1], [L2, U2],...].
        df (float): Degree of freedom.

    Returns:
        float: CDF value at `x`.
    """
    intervals = _interval_to_intervals(interval)
    return _truncated_cdf_from_ccdf(lambda z: t_ccdf_approx(z, df), x, intervals)


def tc2_cdf_approx(x, interval, df):
    """
    Approximated CDF of a truncated chi-squared distribution.

    Args:
        x (float): Return the value at `x`.
        interval (array-like): Truncation interval [L, U] or intervals
            [[L1, U1], [L2, U2],...].
        df (float): Degree of freedom.

    Returns:
        float: CDF value at `x`.
    """
    intervals = _interval_to_intervals(interval)
    return _truncated_cdf_from_ccdf(lambda z: chi2_ccdf_approx(z, df), x, intervals)


def tf_cdf_approx(x, interval, df1, df2):
    """
    Approximated CDF of a truncated F distribution.

    Args:
        x (float): Return the value at `x`.
        interval (array-like): Truncation interval [L, U] or intervals
            [[L1, U1], [L2, U2],...].
        df1 (float): Degree of freedom.
        df2 (float): Degree of freedom.

    Returns:
        float: CDF value at `x`.
    """
    intervals = _interval_to_intervals(interval)
    return _truncated_cdf_from_ccdf(lambda z: f_ccdf_approx(z, df1, df2), x, intervals)
