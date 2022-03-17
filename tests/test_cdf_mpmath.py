from numpy.testing import assert_allclose
from sicore.cdf_mpmath import (
    chi2_cdf_mpmath,
    f_cdf_mpmath,
    mp,
    t_cdf_mpmath,
    tc2_cdf_mpmath,
    tf_cdf_mpmath,
    tn_cdf_mpmath,
    tt_cdf_mpmath,
)

mp.dps = 300

INF = float("inf")
NINF = -INF


def test_t_cdf_mpmath():
    testcase = [
        ((NINF, 2), 0.0),
        ((NINF, 3), 0.0),
        ((-5.0, 2), 0.018874775675311862),
        ((-5.0, 3), 0.007696219036651148),
        ((0.0, 2), 0.5),
        ((0.0, 3), 0.5),
        ((5.0, 2), 0.9811252243246881),
        ((5.0, 3), 0.9923037809633488),
        ((INF, 2), 1.0),
        ((INF, 3), 1.0),
    ]

    for args, expected in testcase:
        assert_allclose(float(t_cdf_mpmath(*args)), expected)


def test_chi2_cdf_mpmath():
    testcase = [
        ((0.0, 2), 0.0),
        ((0.0, 3), 0.0),
        ((1.0, 2), 0.3934693402873665),
        ((1.0, 3), 0.19874804309879915),
        ((3.0, 2), 0.7768698398515702),
        ((3.0, 3), 0.6083748237289109),
        ((INF, 2), 1.0),
        ((INF, 3), 1.0),
    ]

    for args, expected in testcase:
        assert_allclose(float(chi2_cdf_mpmath(*args)), expected)


def test_f_cdf_mpmath():
    testcase = [
        ((0.0, 2, 2), 0.0),
        ((0.0, 2, 3), 0.0),
        ((1.0, 2, 2), 0.5),
        ((1.0, 2, 3), 0.53524199845511),
        ((2.0, 2, 2), 0.6666666666666666),
        ((2.0, 2, 3), 0.7194341411251527),
        ((INF, 2, 2), 1.0),
        ((INF, 2, 3), 1.0),
    ]

    for args, expected in testcase:
        assert_allclose(float(f_cdf_mpmath(*args)), expected)


def test_tn_cdf_mpmath():
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
        assert_allclose(float(tn_cdf_mpmath(*args)), expected)


def test_tt_cdf_mpmath():
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
        assert_allclose(float(tt_cdf_mpmath(*args)), expected)


def test_tc2_cdf_mpmath():
    testcase = [
        ((0.0, [[0.0, 0.5], [1.0, 1.5], [2.0, INF]], 2), 0.0),
        ((0.3, [[0.0, 0.5], [1.0, 1.5], [2.0, INF]], 2), 0.19259373242557318),
        ((1.2, [[0.0, 0.5], [1.0, 1.5], [2.0, INF]], 2), 0.3856495412291721),
        ((INF, [[0.0, 0.5], [1.0, 1.5], [2.0, INF]], 2), 1.0),
    ]

    for args, expected in testcase:
        assert_allclose(float(tc2_cdf_mpmath(*args)), expected)


def test_tf_cdf_mpmath():
    testcase = [
        ((0.0, [[0.0, 0.5], [1.0, 1.5], [2.0, INF]], 2, 3), 0.0),
        ((0.3, [[0.0, 0.5], [1.0, 1.5], [2.0, INF]], 2, 3), 0.3223627738673543),
        ((1.2, [[0.0, 0.5], [1.0, 1.5], [2.0, INF]], 2, 3), 0.5404533787680365),
        ((INF, [[0.0, 0.5], [1.0, 1.5], [2.0, INF]], 2, 3), 1.0),
    ]

    for args, expected in testcase:
        assert_allclose(float(tf_cdf_mpmath(*args)), expected)
