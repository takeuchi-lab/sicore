from numpy.testing import assert_allclose
from sicore.cdf_approx import (
    chi2_ccdf_approx,
    f_ccdf_approx,
    norm_ccdf_approx,
    t_ccdf_approx,
    tc2_cdf_approx,
    tf_cdf_approx,
    tn_cdf_approx,
    tt_cdf_approx,
)

INF = float("inf")
NINF = -INF


def test_norm_ccdf_approx():
    testcase = [
        (0.0, 0.5),
        (1.0, 0.15865525393145707),
        (3.0, 0.0013498980316301035),
        (INF, 0.0),
    ]

    for x, expected in testcase:
        assert_allclose(norm_ccdf_approx(x), expected, atol=1e-3)


def test_t_ccdf_approx():
    testcase = [
        ((0.0, 2), 0.5),
        ((0.0, 3), 0.5),
        ((5.0, 2), 0.018874775675311862),
        ((5.0, 3), 0.007696219036651148),
        ((INF, 2), 0.0),
        ((INF, 3), 0.0),
    ]

    for args, expected in testcase:
        assert_allclose(t_ccdf_approx(*args), expected, atol=1e-3)


def test_chi2_ccdf_approx():
    testcase = [
        ((0.0, 2), 1.0),
        ((0.0, 3), 1.0),
        ((1.0, 2), 0.6065306597126335),
        ((1.0, 3), 0.80125195690120085),
        ((3.0, 2), 0.2231301601484298),
        ((3.0, 3), 0.3916251762710891),
        ((INF, 2), 0.0),
        ((INF, 3), 0.0),
    ]

    for args, expected in testcase:
        assert_allclose(chi2_ccdf_approx(*args), expected, atol=1e-2)


def test_f_ccdf_approx():
    testcase = [
        ((0.0, 2, 3), 1.0),
        ((1.0, 2, 3), 0.46475800154489),
        ((3.0, 2, 3), 0.1924500897298751),
        ((INF, 2, 3), 0.0),
    ]

    for args, expected in testcase:
        assert_allclose(f_ccdf_approx(*args), expected, atol=1e-1)


def test_tn_cdf_approx():
    testcase = [
        ((NINF, [[NINF, -1.5], [-1.0, -0.8], [-0.3, 0.5], [1.0, INF]]), 0.0),
        (
            (-1.7, [[NINF, -1.5], [-1.0, -0.8], [-0.3, 0.5], [1.0, INF]]),
            0.07578690102235282,
        ),
        (
            (0.0, [[NINF, -1.5], [-1.0, -0.8], [-0.3, 0.5], [1.0, INF]]),
            0.40459865137689516,
        ),
        (
            (0.3, [[NINF, -1.5], [-1.0, -0.8], [-0.3, 0.5], [1.0, INF]]),
            0.6051158395693588,
        ),
        ((INF, [[NINF, -1.5], [-1.0, -0.8], [-0.3, 0.5], [1.0, INF]]), 1.0),
    ]

    for args, expected in testcase:
        assert_allclose(tn_cdf_approx(*args), expected, atol=1e-3)


def test_tt_cdf_approx():
    testcase = [
        ((NINF, [[NINF, -1.5], [-1.0, -0.8], [-0.3, 0.5], [1.0, INF]], 2), 0.0),
        (
            (-1.7, [[NINF, -1.5], [-1.0, -0.8], [-0.3, 0.5], [1.0, INF]], 2),
            0.17506081601590198,
        ),
        (
            (0.0, [[NINF, -1.5], [-1.0, -0.8], [-0.3, 0.5], [1.0, INF]], 2),
            0.4276648740747664,
        ),
        (
            (0.3, [[NINF, -1.5], [-1.0, -0.8], [-0.3, 0.5], [1.0, INF]], 2),
            0.5847685858739919,
        ),
        ((INF, [[NINF, -1.5], [-1.0, -0.8], [-0.3, 0.5], [1.0, INF]], 2), 1.0),
    ]

    for args, expected in testcase:
        assert_allclose(tt_cdf_approx(*args), expected, atol=1e-2)


def test_tc2_cdf_approx():
    testcase = [
        ((0.0, [[0.0, 0.5], [1.0, 1.5], [2.0, INF]], 2), 0.0),
        ((0.3, [[0.0, 0.5], [1.0, 1.5], [2.0, INF]], 2), 0.19259373242557318),
        ((1.2, [[0.0, 0.5], [1.0, 1.5], [2.0, INF]], 2), 0.3856495412291721),
        ((INF, [[0.0, 0.5], [1.0, 1.5], [2.0, INF]], 2), 1.0),
    ]

    for args, expected in testcase:
        assert_allclose(tc2_cdf_approx(*args), expected, atol=1e-2)


def test_tf_cdf_approx():
    testcase = [
        ((0.0, [[0.0, 0.5], [1.0, 1.5], [2.0, INF]], 2, 3), 0.0),
        ((0.3, [[0.0, 0.5], [1.0, 1.5], [2.0, INF]], 2, 3), 0.3223627738673543),
        ((1.2, [[0.0, 0.5], [1.0, 1.5], [2.0, INF]], 2, 3), 0.5404533787680365),
        ((INF, [[0.0, 0.5], [1.0, 1.5], [2.0, INF]], 2, 3), 1.0),
    ]

    for args, expected in testcase:
        assert_allclose(tf_cdf_approx(*args), expected, atol=1e-1)
