import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from sicore.intervals import (
    _interval_to_intervals,
    _verify_and_raise,
    intersection,
    intersection_all,
    not_,
    poly_lt_zero,
    union_all,
)

INF = float("inf")
NINF = -INF


def test__interval_to_intervals():
    testcase = [([], []), ([0, 1], [[0, 1]]), ([[0, 1], [2, 3]], [[0, 1], [2, 3]])]

    for interval, expected in testcase:
        assert_equal(_interval_to_intervals(interval), expected)


def test__verify_and_raise():
    testcase = [[], [[1, 2]], [[1, 2], [3, 4]]]

    for intervals in testcase:
        _verify_and_raise(intervals)

    testcase_type_error = [
        [[1, None], [2, 3]],
    ]

    for intervals in testcase_type_error:
        with pytest.raises(TypeError):
            _verify_and_raise(intervals)

    testcase_value_error = [
        [[1, 2], [3, 2]],
        [[1, 1]],
        [[INF, INF]],
        [[1, 2], [3]],
    ]

    for intervals in testcase_value_error:
        with pytest.raises(ValueError):
            _verify_and_raise(intervals)


def test_intersection():
    testcase = [
        ([], [], []),
        ([], [1, 2], []),
        ([1, 2], [1, 2], [[1, 2]]),
        ([1, 3], [2, 4], [[2, 3]]),
        ([[1, 3]], [[2, 4]], [[2, 3]]),
        ([[NINF, 1], [2, INF]], [0, 3], [[0, 1], [2, 3]]),
        ([[NINF, 1], [2, INF]], [[0, 1], [3, INF]], [[0, 1], [3, INF]]),
        ([[NINF, 1], [2, INF]], [1, 2], []),
    ]

    for interval1, interval2, expected in testcase:
        assert_equal(intersection(interval1, interval2), expected)


def test_intersection_all():
    testcase = [
        ([], []),
        ([[1, 2]], [1, 2]),
        ([[0, 1], [2, 3]], []),
        ([[-5, 1], [-8, -7]], []),
        ([[NINF, 1], [1, INF]], []),
        ([[0, 5], [3, 7], [2, 4]], [3, 4]),
    ]

    for intervals, expected in testcase:
        assert_equal(intersection_all(intervals), expected)


def test_union_all():
    testcase = [
        ([], []),
        ([[1, 2]], [[1, 2]]),
        ([[1, INF], [1, INF], [1, INF]], [[1, INF]]),
        ([[3, 4], [1, 2]], [[1, 2], [3, 4]]),
        ([[8, 9], [2, 6], [1, 3], [4, 5], [7, 8]], [[1, 6], [7, 9]]),
    ]

    for intervals, expected in testcase:
        assert_equal(union_all(intervals), expected)

    testcase_merge = [
        ([[1, 1.999], [2, 3]], 0.001, [[1, 3]]),
        ([[1, 1.998], [2, 3]], 0.001, [[1, 1.998], [2, 3]]),
    ]

    for intervals, tol, expected in testcase_merge:
        assert_equal(union_all(intervals, tol=tol), expected)


def test_not_():
    testcase = [
        ([], [[NINF, INF]]),
        ([NINF, INF], []),
        ([[NINF, INF]], []),
        ([[NINF, 0], [1, 2], [3, INF]], [[0, 1], [2, 3]]),
        ([[3, INF], [NINF, 0], [1, 2]], [[0, 1], [2, 3]]),
        ([[0, 1]], [[NINF, 0], [1, INF]]),
        ([NINF, 0], [[0, INF]]),
        ([[NINF, 0]], [[0, INF]]),
        ([0, INF], [[NINF, 0]]),
        ([[0, INF]], [[NINF, 0]]),
        ([[NINF, 0], [0, INF]], []),
    ]

    for intervals, expected in testcase:
        assert_equal(not_(intervals), expected)

    with pytest.raises(ValueError):
        not_([[0, 2], [1, 3]])


def test_poly_lt_zero():
    testcase = [
        ([1], []),
        ([0], [[NINF, INF]]),
        ([-1], [[NINF, INF]]),
        ([1, 1], [[NINF, -1]]),
        ([-1, 1], [[1, INF]]),
        ([1, -2, 2], []),
        ([1, -2, 1], []),
        ([1, 0, -1], [[-1, 1]]),
        ([-1, 0, 1], [[NINF, -1], [1, INF]]),
        ([-1, 2, -1], [[NINF, INF]]),
        ([-1, 2, -2], [[NINF, INF]]),
        ([1, -6, 11, -6], [[NINF, 1], [2, 3]]),
        ([1, -3, 4, -2], [[NINF, 1]]),
        ([1, 0, 0, 0], [[NINF, 0]]),
    ]

    for coef, expected in testcase:
        assert_allclose(poly_lt_zero(coef), expected)

    for coef, expected in testcase:
        poly = np.poly1d(coef)
        assert_allclose(poly_lt_zero(poly), expected)
