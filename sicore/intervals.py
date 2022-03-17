import itertools
from operator import itemgetter

import numpy as np

from .utils import is_int_or_float

INF = float("inf")
NINF = -INF


def _interval_to_intervals(interval):
    """
    Convert interval [l1, u1] or intervals [[l1, u1], [l2, u2], ...] to intervals
    [[L1, U1], [L2, U2], ...].
    """
    if len(interval) > 0 and is_int_or_float(interval[0]):
        return [interval]
    else:
        return interval


def _verify_and_raise(intervals):
    """
    Check if intervals are valid or not, and raise exception.

    Args:
        intervals (array-like): Intervals [[l1, u1], [l2, u2], ...].
    """
    for s, e in intervals:
        if not is_int_or_float(s) or not is_int_or_float(e):
            raise TypeError(f"Invalid data type exists in {[s, e]}")

        if s >= e:
            raise ValueError(f"Inverted or no-range interval found in {[s, e]}")


def intersection(interval1, interval2, verify=False):
    """
    Take intersection of two intervals.

    Args:
        interval1 (array-like): Interval [l1, u1] or the union of intervals
            [[l1, u1], [l2, u2], ...].
        interval2 (array-like): Interval [l1, u1] or the union of intervals
            [[l1, u1], [l2, u2], ...].
        verify (bool, optional): If True, check if the intervals are valid or not at
            the beginning, and raise exception. Set False for faster computation.
            Defaults to False.

    Returns:
        list: Intersection of the intervals [[L1, U1], [L2, U2], ...].
    """
    intervals1 = _interval_to_intervals(interval1)
    intervals2 = _interval_to_intervals(interval2)
    if verify:
        _verify_and_raise(intervals1)
        _verify_and_raise(intervals2)

    result_intervals = []
    for i1, i2 in itertools.product(intervals1, intervals2):
        lower = max(i1[0], i2[0])
        upper = min(i1[1], i2[1])
        if lower < upper:
            result_intervals.append([lower, upper])

    return result_intervals


def intersection_all(intervals, verify=False):
    """
    Take intersection of multiple intervals.

    Args:
        intervals (array-like): Intervals [[l1, u1], [l2, u2], ...].
        verify (bool, optional): If True, check if the intervals are valid or not at
            the beginning, and raise exception. Set False for faster computation.
            Defaults to False.

    Returns:
        list: Intersection of the intervals [] or [L, U].
    """
    if verify:
        _verify_and_raise(intervals)
    if len(intervals) == 0:
        return []
    lower = NINF
    upper = INF
    for low, up in intervals:
        lower = max(lower, low)
        upper = min(upper, up)
    if lower >= upper:
        return []
    else:
        return [lower, upper]


def union_all(intervals, tol=0.0, verify=False):
    """
    Take union of multiple intervals.

    Args:
        intervals (array-like): Intervals [[l1, u1], [l2, u2], ...].
        tol (float, optional): Tolerance error parameter. When `tol`>=0.1, the union of
            the intervals [[0, 1], [1.1, 2]] will be [[0, 2]]. Defaults to 0.0.
        verify (bool, optional): If True, check if the intervals are valid or not at
            the beginning, and raise exception. Set False for faster computation.
            Defaults to False.

    Returns:
        list: Union of the intervals [[L1, U1], [L2, U2], ...].
    """
    if verify:
        _verify_and_raise(intervals)
    if len(intervals) == 0:
        return []
    sorted_intervals = sorted(intervals, key=itemgetter(0))
    result_intervals = []
    s, e = sorted_intervals[0]
    for s_i, e_i in sorted_intervals[1:]:
        if s_i <= e + tol:
            e = max(e, e_i)
        else:
            result_intervals.append([s, e])
            s, e = s_i, e_i
    result_intervals.append([s, e])
    return result_intervals


def not_(interval, verify=False):
    """
    Take complement of interval in the real number field.

    Args:
        interval (array-like): Interval [l1, u1] or intervals [[l1, u1], [l2, u2], ...].
        verify (bool, optional): If True, check if the intervals are valid or not at
            the beginning, and raise exception. Set False for faster computation.
            Defaults to False.

    Returns:
        list: Complement of the intervals [[L1, U1], [L2, U2], ...].
    """
    intervals = _interval_to_intervals(interval)
    if verify:
        _verify_and_raise(intervals)
    if len(intervals) == 0:
        return [[NINF, INF]]
    sorted_intervals = sorted(intervals, key=itemgetter(0))
    result_intervals = []

    s = sorted_intervals[0][0]
    if not np.isneginf(s):
        result_intervals.append([NINF, s])

    for i in range(len(sorted_intervals) - 1):
        e = sorted_intervals[i][1]
        s = sorted_intervals[i + 1][0]
        if e < s:
            result_intervals.append([e, s])
        elif e > s:
            raise ValueError("Overlapping intervals exist")

    e = sorted_intervals[-1][1]
    if not np.isposinf(e):
        result_intervals.append([e, INF])

    return result_intervals


def poly_lt_zero(poly_or_coef, tol=1e-10):
    """
    Given a polynomial `f(x)`, find the intervals of `x` where `f(x) <= 0`.

    Args:
        poly_or_coef (np.poly1d, array-like): np.poly1d object or coefficients of the
            polynomial e.g. [a, b, c] for `f(x) = ax^2 + bx + c`.
        tol (float, optional): Tolerance error parameter. It is recommended to set a
            large value (about 1e-5) for high order polynomial (>= 3) or a polynomial
            with multiple root. Defaults to 1e-10.

    Returns:
        list: Intervals [[L1, U1], [L2, U2], ...] of `x` where `f(x) <= 0`.
    """
    if isinstance(poly_or_coef, np.poly1d):
        coef = poly_or_coef.coef
    else:
        coef = poly_or_coef
    coef = [0 if -tol < c < tol else c for c in coef]
    poly = np.poly1d(coef)

    if poly.order == 0:
        if poly.coef[0] <= 0:
            return [[NINF, INF]]
        else:
            return []

    roots = []
    if np.issubdtype(poly.roots.dtype, np.complex):
        for root in poly.roots:
            if -tol < root.imag < tol:
                roots.append(root.real)

        if len(roots) == 0:
            if poly(0) <= 0:
                return [[NINF, INF]]
            else:
                return []
    else:
        roots = poly.roots

    roots = np.unique(roots)  # the return value is sorted
    intervals = []

    if poly(roots[0] - 1) <= 0:
        intervals.append([NINF, roots[0]])
    for s, e in zip(roots, roots[1:]):
        if e - s < tol:
            continue
        mid = (s + e) / 2
        if poly(mid) <= 0:
            intervals.append([s, e])
    if poly(roots[-1] + 1) <= 0:
        intervals.append([roots[-1], INF])

    return union_all(intervals, tol=tol)
