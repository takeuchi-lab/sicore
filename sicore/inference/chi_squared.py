from abc import ABC, abstractmethod
import math
import numpy as np
from scipy.linalg import fractional_matrix_power
from ..utils import is_int_or_float
from ..intervals import intersection, not_, poly_lt_zero, union_all
from ..cdf_mpmath import chi2_cdf_mpmath, tc2_cdf_mpmath, tn_cdf_mpmath as tn_cdf
from .base import *


class InferenceChiSquared(ABC):
    """
    Base inference class for a test statistic which follows chi squared distribution under
    null.

    Args:
        data (array-like): Observation data of length `N`.
        var (float, array-like): Value of known variance, or `N`*`N` covariance matrix.
        basis (array-like): List of basis vector of length `N`.
    """

    def __init__(self, data, var, basis):
        self.data = data
        basis = np.array(basis)
        V = np.linalg.qr(basis.T)[0]
        P = np.dot(V, V.T)
        if np.sum(np.abs(np.dot(P, P) - P)) > 1e-5 or np.sum(np.abs(P.T - P)) > 1e-5:
            raise Exception(
                "The projection matrix is not constructed correctly")
        self.length = len(data)
        self.degree = len(basis)
        if is_int_or_float(var):
            self.cov = var * np.identity(self.length)
        else:
            self.cov = np.asarray(var)
        self.inv_sqrt_cov = fractional_matrix_power(self.cov, -0.5)
        self.P_data = np.dot(P, data)
        self.stat = np.linalg.norm(
            np.dot(self.inv_sqrt_cov, self.P_data), ord=2)

    @abstractmethod
    def test(self, *args, **kwargs):
        """Perform statistical testing."""
        pass


class NaiveInferenceChiSquared(InferenceChiSquared):
    """
    Naive inference for a test statistic which follows chi squared distribution under null.

    Args:
        data (array-like): Observation data of length `N`.
        var (float, array-like): Value of known variance, or `N`*`N`covariance matrix.
        basis (array-like): List of basis vector of length `N`.
    """

    def test(self, tail="right"):
        """
        Perform naive statistical testing.

        Args:
            tail (str, optional): 'double' for double-tailed test, 'right' for
                right-tailed test, and 'left' for left-tailed test. Defaults to
                'right'.

        Returns:
            float: p-value
        """

        chi = chi2_cdf_mpmath(self.stat ** 2, self.degree)
        return calc_pvalue(chi, tail=tail)


class SelectiveInferenceChiSquared(InferenceChiSquared):
    """
    Selective inference for a test statistic which follows chi squared distribution under
    null.

    Args:
        data (array-like): Observation data of length `N`.
        var (float, array-like): Value of known variance, or `N`*`N` covariance matrix.
        basis (array-like): List of basis vector of length `N`.
    """

    def __init__(self, data, var, basis):
        super().__init__(data, var, basis)
        self.c = self.P_data / self.stat
        self.z = self.data - self.P_data
        self.intervals = [[NINF, INF]]
        self.searched_intervals = None
        self.mappings = None  # {interval1: model1, interval2: model2, ...}
        self.tol = None

    @property
    def parametric_data(self):
        return [np.poly1d([a, b]) for a, b in zip(self.c, self.z)]

    def add_polytope(self, A=None, b=None, c=None, tol=1e-10):
        """
        Add a polytope `{x'Ax+b'x+c<=0}` as a selection event.

        Args:
            A (array-like, optional): `N`*`N` matrix. Set None if `A` is unused.
                Defaults to None.
            b (array-like, optional): `N` dimensional vector. Set None if `b` is unused.
                Defaults to None.
            c (float, optional): Constant. Set None if `c` is unused. Defaults to None.
            tol (float, optional): Tolerance error parameter. Defaults to 1e-10.
        """
        alp = beta = gam = 0

        if A is not None:
            cA = np.dot(self.c, A)
            zA = np.dot(self.z, A)
            alp += np.dot(cA, self.c)
            beta += np.dot(zA, self.c) + np.dot(cA, self.z)
            gam += np.dot(zA, self.z)

        if b is not None:
            beta += np.dot(b, self.c)
            gam += np.dot(b, self.z)

        if c is not None:
            gam += c

        self.add_polynomial([alp, beta, gam], tol=tol)

    def add_polynomial(self, poly_or_coef, tol=1e-10):
        """
        Add a polynomial `f(x)` as a selection event.

        Args:
            poly_or_coef (np.poly1d, array-like): np.poly1d object or coefficients of
                the polynomial e.g. [a, b, c] for `f(x) = ax^2 + bx + c`.
            tol (float, optional): Tolerance error parameter. It is recommended to set
                a large value (about 1e-5) for high order polynomial (>= 3) or a
                polynomial with multiple root. Defaults to 1e-10.
        """
        intervals = poly_lt_zero(poly_or_coef, tol=tol)
        self.intervals = intersection(self.intervals, intervals)

    def add_interval(self, interval):
        """
        Add an interval as a selection event.

        Args:
            interval (array-like): Interval [l1, u1] or the union of intervals
                [[l1, u1], [l2, u2], ...].
        """
        self.intervals = intersection(self.intervals, interval)

    def _next_search_data(self):
        intervals = not_(self.searched_intervals)
        if len(intervals) == 0:
            return None
        s, e = random.choice(intervals)
        param = (e + s) / 2
        return self.c * param + self.z

    def parametric_search(self, algorithm, max_tail=1000, tol=1e-10):
        """
        Perform parametric search.

        Args:
            algorithm (callable): Callable function which takes a new data vector of
                length `N` as single argument, and returns the selected model (any) and
                the truncation intervals (array-like). A closure function might be
                helpful to implement this.
            max_tail (float, optional): Maximum tail value to be searched. Defaults to
                1000.
            tol (float, optional): Tolerance error parameter. Defaults to 1e-10.
        """
        if self.intervals == [[NINF, INF]]:
            raise Exception("Initial intervals are not set")

        self.tol = tol
        self.mappings = dict()
        self.searched_intervals = union_all(
            [[NINF, -max_tail]] + list(self.intervals) + [[max_tail, INF]], tol=self.tol
        )

        while True:
            data = self._next_search_data()
            if data is None:
                break
            model, intervals = algorithm(data)
            for interval in intervals:
                interval = tuple(interval)
                if interval in self.mappings:
                    raise Exception(
                        "An interval appeared a second time. Usually, numerical error "
                        "causes this exception. Consider increasing the tol parameter "
                        "or decreasing max_tail parameter to avoid it."
                    )
                self.mappings[interval] = model
            self.searched_intervals = union_all(
                self.searched_intervals + list(intervals), tol=self.tol
            )

    def test(
        self, intervals=None, model_selector=None, tail="right", dps="auto", out_log='test_log.log', max_dps=5000
    ):
        """
        Perform selective statistical testing.

        Args:
            model_selector (callable): Callable function which takes a selected model
                (any) as single argument, and returns True if the model is used for the
                testing, and False otherwise. This option is valid after calling
                ``self.parametric_search()``.
            tail (str, optional): 'double' for double-tailed test, 'right' for
                right-tailed test, and 'left' for left-tailed test. Defaults to
                righte'.
            dps (int, str, optional): dps value for mpmath. Set 'auto' to select dps
                automatically. Defaults to 'auto'.
            max_dps (int, optional): Maximum dps value for mpmath. This option is valid
                when `dps` is set to 'auto'. Defaults to 5000.

        Returns:
            float: p-value
        """
        if intervals is None:
            if model_selector is None:
                intervals = self.intervals
            else:
                if self.mappings is None:
                    raise Exception("Parametric search has not been performed")
                result_intervals = list(self.intervals)
                for interval, model in self.mappings.items():
                    if model_selector(model):
                        result_intervals.append(interval)
                intervals = union_all(result_intervals, tol=self.tol)
        else:
            self.interval = np.asarray(intervals)
            intervals = self.interval

        stat = self.stat ** 2
        chi_intervals = intersection(
            intervals, [[1e-5, INF]])
        chi_squared_intervals = np.power(chi_intervals, 2)
        chi = tc2_cdf_mpmath(stat, chi_squared_intervals,
                             self.degree, dps=dps, out_log=out_log)

        return calc_pvalue(chi, tail=tail)
    

