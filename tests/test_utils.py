import numpy as np
from numpy.testing import assert_equal
from sicore.utils import OneVec, is_int_or_float


def test_is_int_or_float():
    testcase = [
        (1, True),
        (1.23, True),
        (float("inf"), True),
        (np.int64(1), True),
        (np.float64(1.23), True),
        ("1", False),
        (True, False),
        ([1, 2], False),
        (np.array([1, 2]), False),
    ]

    for value, expected in testcase:
        assert is_int_or_float(value) == expected


def test_OneVec():
    testcase = [
        ((1,), [1, 0, 0, 0, 0]),
        ((2,), [0, 1, 0, 0, 0]),
        ((5,), [0, 0, 0, 0, 1]),
        ((1, 1,), [1, 0, 0, 0, 0]),
        ((2, 2,), [0, 1, 0, 0, 0]),
        ((5, 5,), [0, 0, 0, 0, 1]),
        ((1, 3,), [1, 1, 1, 0, 0]),
        ((2, 3,), [0, 1, 1, 0, 0]),
    ]

    one = OneVec(5)
    for args, expected in testcase:
        assert_equal(one.get(*args), expected)
