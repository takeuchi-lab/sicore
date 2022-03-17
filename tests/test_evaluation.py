from numpy.testing import assert_allclose
from sicore.evaluation import (
    false_negative_rate,
    false_positive_rate,
    power,
    true_negative_rate,
    true_positive_rate,
    type1_error_rate,
    type2_error_rate,
)


def test_false_positive_rate():
    testcase = [
        ([0.01, 0.1, 0.05, 0.02, 0.5], 0.6),
        ([0, 0, 0, 0, 0], 1),
        ([1, 1, 1, 1, 1], 0),
    ]

    for pvalues, expected in testcase:
        assert_allclose(false_positive_rate(pvalues), expected)
        assert_allclose(type1_error_rate(pvalues), expected)


def test_false_negative_rate():
    testcase = [
        ([0.01, 0.1, 0.05, 0.02, 0.5], 0.4),
        ([0, 0, 0, 0, 0], 0),
        ([1, 1, 1, 1, 1], 1),
    ]

    for pvalues, expected in testcase:
        assert_allclose(false_negative_rate(pvalues), expected)
        assert_allclose(type2_error_rate(pvalues), expected)


def test_true_negative_rate():
    testcase = [
        ([0.01, 0.1, 0.05, 0.02, 0.5], 0.4),
        ([0, 0, 0, 0, 0], 0),
        ([1, 1, 1, 1, 1], 1),
    ]

    for pvalues, expected in testcase:
        assert_allclose(true_negative_rate(pvalues), expected)


def test_true_positive_rate():
    testcase = [
        ([0.01, 0.1, 0.05, 0.02, 0.5], 0.6),
        ([0, 0, 0, 0, 0], 1),
        ([1, 1, 1, 1, 1], 0),
    ]

    for pvalues, expected in testcase:
        assert_allclose(true_positive_rate(pvalues), expected)
        assert_allclose(power(pvalues), expected)
