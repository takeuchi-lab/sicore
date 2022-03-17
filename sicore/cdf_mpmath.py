import numpy as np
from mpmath import mp

from .intervals import _interval_to_intervals

mp.dps = 1000

INF = float("inf")
NINF = -INF


def t_cdf_mpmath(x, df):
    """
    CDF of a t distribution.

    Args:
        x (float): Return the value at `x`.
        df (float): Degree of freedom.

    Returns:
        float: CDF value at `x`.
    """
    z = df / (x ** 2 + df)
    abs_x_ccdf = mp.betainc(df / 2, 1 / 2, x2=z, regularized=True) / 2
    return 1 - abs_x_ccdf if x >= 0 else abs_x_ccdf


def chi2_cdf_mpmath(x, df):
    """
    CDF of a chi-squared distribution.

    Args:
        x (float): Return the value at `x`.
        df (float): Degree of freedom.

    Returns:
        float: CDF value at `x`.
    """
    return mp.gammainc(df / 2, a=0, b=x / 2, regularized=True)


def f_cdf_mpmath(x, df1, df2):
    """
    CDF of a F distribution.

    Args:
        x (float): Return the value at `x`.
        df1 (float): Degree of freedom.
        df2 (float): Degree of freedom.

    Returns:
        float: CDF value at `x`.
    """
    z = 1.0 if mp.isinf(x) else (df1 * x) / (df1 * x + df2)
    return mp.betainc(df1 / 2, df2 / 2, x2=z, regularized=True)


def _truncated_cdf_from_cdf(
    cdf_func, x, intervals, dps="auto", max_dps=5000, init_dps=30, scale=2, precision=15, out_log='test_log.log'
):
    """
    Calculate CDF of a truncated distribution from a CDF function.

    Args:
        cdf_func (callable): CDF function of a distribution.
        x (float): Return the value at `x`.
        intervals (array-like): Truncation intervals [[L1, U1], [L2, U2],...].
        dps (int, str, optional): dps value for mpmath. Set 'auto' to select dps
            automatically, although it will not work well when the interval is
            extremely narrow and the cdf values are almost the same. The auto selection
            requires some overheads. For faster calculation, set an interger value.
            Defaults to 'auto'.
        max_dps (int, optional): Maximum dps value for mpmath. This option is valid
            when `dps` is set to 'auto'. Defaults to 5000.
        init_dps (int, optional): Initial dps value. This option is valid when `dps` is
            set to 'auto'. Defaults to 30.
        scale (float, optional): This value will be multiplied to dps when increasing
            the precision. This value has to be greater than 1.0. This option is valid
            when `dps` is set to 'auto'. Defaults to 2.
        precision (int, optional): The minimum number of valid digits. This value has
            to be less than `init_dps`. This option is valid when `dps` is set to
            'auto'. Defaults to 15.

    Returns:
        float: CDF value at `x`.

    Raises:
        ValueError: If the value `x` is not located inside the `intervals`.
    """
    if dps == "auto":
        mp.dps = init_dps

        # If the cdf value of the maximum absolute truncation interval except INF and
        # NINF is '1.0', increase the precision
        flatten = np.ravel(intervals)
        max_tail = np.abs(flatten[np.isfinite(flatten)]).max()
        if (
            mp.nstr(
                cdf_func(max_tail), init_dps - precision, min_fixed=NINF, max_fixed=INF
            )
            == "1.0"
        ):
            next_dps = int(init_dps * scale)
            if next_dps <= max_dps:
                return _truncated_cdf_from_cdf(
                    cdf_func,
                    x,
                    intervals,
                    dps=dps,
                    init_dps=next_dps,
                    scale=scale,
                    precision=precision,
                )
    else:
        mp.dps = dps

    num = denom = 0
    inside_flag = False

    for lower, upper in intervals:
        diff = cdf_func(upper) - cdf_func(lower)
        denom += diff
        if lower <= x <= upper:
            num += cdf_func(x) - cdf_func(lower)
            inside_flag = True
        elif upper < x:
            num += diff

    if not inside_flag:
        raise ValueError(f"Value x={x} is outside the intervals={intervals}")

    try:
        cdf_value = num / denom
    except ZeroDivisionError:
        raise ZeroDivisionError(
            "Denominator has the value zero. Consider "
            "increasing the dps value to avoid it."
        )

    return cdf_value


def tn_cdf_mpmath(x, interval, **kwargs):
    """
    CDF of a truncated normal distribution.

    Args:
        x (float): Return the value at `x`.
        interval (array-like): Truncation interval [L, U] or intervals
            [[L1, U1], [L2, U2],...].

    Returns:
        float: CDF value at `x`.
    """
    intervals = _interval_to_intervals(interval)
    cdf_val = _truncated_cdf_from_cdf(mp.ncdf, x, intervals, **kwargs)
    return float(cdf_val)


def tt_cdf_mpmath(x, interval, df, **kwargs):
    """
    CDF of a truncated t distribution.

    Args:
        x (float): Return the value at `x`.
        interval (array-like): Truncation interval [L, U] or intervals
            [[L1, U1], [L2, U2],...].
        df (float): Degree of freedom.

    Returns:
        float: CDF value at `x`.
    """
    intervals = _interval_to_intervals(interval)
    cdf_val = _truncated_cdf_from_cdf(
        lambda z: t_cdf_mpmath(z, df), x, intervals, **kwargs
    )
    return float(cdf_val)


def tc2_cdf_mpmath(x, interval, df, **kwargs):
    """
    CDF of a truncated chi-squared distribution.

    Args:
        x (float): Return the value at `x`.
        interval (array-like): Truncation interval [L, U] or intervals
            [[L1, U1], [L2, U2],...].
        df (float): Degree of freedom.

    Returns:
        float: CDF value at `x`.
    """
    intervals = _interval_to_intervals(interval)
    cdf_val = _truncated_cdf_from_cdf(
        lambda z: chi2_cdf_mpmath(z, df), x, intervals, **kwargs
    )
    return float(cdf_val)


def tf_cdf_mpmath(x, interval, df1, df2, **kwargs):
    """
    CDF of a truncated t distribution.

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
    cdf_val = _truncated_cdf_from_cdf(
        lambda z: f_cdf_mpmath(z, df1, df2), x, intervals, **kwargs
    )
    return float(cdf_val)
