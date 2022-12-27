from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import fractional_matrix_power
from ..utils import is_int_or_float
from ..intervals import intersection, not_, poly_lt_zero, union_all, _interval_to_intervals
from ..cdf_mpmath import chi2_cdf_mpmath, tc2_cdf_mpmath, tn_cdf_mpmath as tn_cdf
from .base import *


class InferenceChiSquared(ABC):
    """
    Base inference class for a test statistic which follows chi squared distribution under null.

    Args:
        data (array-like): Observation data of length `N`.
        var (float, array-like): Value of known variance, or `N`*`N` covariance matrix.
        basis (array-like): List of basis vector of length `N`.
        use_tf (boolean, optional): Whether to use tensorflow or not. Defaults to False.
    """

    def __init__(self, data, var, basis, use_tf=False):
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

        if use_tf:
            try:
                import tensorflow as tf
            except ModuleNotFoundError:
                raise Exception('use_tf is True, but package not found.')

            assert isinstance(data, tf.Tensor)

            self.inv_sqrt_cov = tf.constant(
                self.inv_sqrt_cov, dtype=tf.float64)
            self.P_data = tf.tensordot(P, data, axes=1)
            self.stat = tf.norm(
                tf.tensordot(self.inv_sqrt_cov, self.P_data, axes=1), ord=2)

        else:
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
        use_tf (boolean, optional): Whether to use tensorflow or not. Defaults to False.
    """

    def test(self, tail="right"):
        """
        Perform naive statistical testing.

        Args:
            tail (str, optional): 'double' for double-tailed test, 'right' for
                right-tailed test, and 'left' for left-tailed test. Defaults to 'right'.

        Returns:
            float: p-value
        """

        chi = chi2_cdf_mpmath(np.asarray(self.stat) ** 2, self.degree)
        return calc_pvalue(chi, tail=tail)


class SelectiveInferenceChiSquared(InferenceChiSquared):
    """
    Selective inference for a test statistic which follows chi squared distribution under null.

    Args:
        data (array-like): Observation data of length `N`.
        var (float, array-like): Value of known variance, or `N`*`N` covariance matrix.
        basis (array-like): List of basis vector of length `N`.
        use_tf (boolean, optional): Whether to use tensorflow or not. Defaults to False.
    """

    def __init__(self, data, var, basis, use_tf=False):
        super().__init__(data, var, basis, use_tf)
        self.c = self.P_data / self.stat  # `b` vector in para si.
        self.z = self.data - self.P_data  # `a` vector in para si.
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

    def _next_search_data(self, line_search):
        intervals = not_(self.searched_intervals)
        if len(intervals) == 0:
            return None
        if line_search:
            param = intervals[0][0] + self.step
        else:
            s, e = random.choice(intervals)
            param = (e + s) / 2
        return param

    def parametric_search(self, algorithm, max_tail=1000, tol=1e-10, model_selector=None, line_search=True, step=1e-10):
        """
        Perform parametric search.

        Args:
            algorithm (callable): Callable function which takes two vectors (`a`, `b`)
                and a scalar `z` that can satisfy `data = a + b * z`
                as arguments, and returns the selected model (any) and
                the truncation intervals (array-like). A closure function might be
                helpful to implement this.
            max_tail (float, optional): Maximum tail value to be searched. Defaults to 1000.
            tol (float, optional): Tolerance error parameter. Defaults to 1e-10.
            model_selector (callable, optional): Callable function which takes
                a selected model (any) as single argument, and returns True
                if the model is used for the testing, and False otherwise.
                If this option is activated, the truncation intervals for testing is
                calculated within this method. Defaults to None.
            line_search (boolean, optional): Whether to perform a line search or a random search
                from unexplored regions. Defaults to True.
            step (float, optional): Step width for line search. Defaults to 1e-10.
        """
        self.tol = tol
        self.step = step
        self.searched_intervals = union_all(
            [[NINF, 1e-10], [float(max_tail), INF]], tol=self.tol
        )
        self.mappings = dict()
        result_intervals = list()

        self.count = 0
        self.detect_count = 0

        z = self._next_search_data(line_search)
        while True:
            if z is None:
                break
            self.count += 1

            model, interval = algorithm(self.z, self.c, z)
            interval = np.asarray(interval)
            intervals = _interval_to_intervals(interval)

            if model_selector is None:
                for interval in intervals:
                    interval = tuple(interval)
                    if interval in self.mappings:
                        raise Exception(
                            "An interval appeared a second time. Usually, numerical error "
                            "causes this exception. Consider increasing the tol parameter "
                            "or decreasing max_tail parameter to avoid it."
                        )
                    self.mappings[interval] = model
            else:
                if model_selector(model):
                    result_intervals += intervals
                    self.detect_count += 1

            self.searched_intervals = union_all(
                self.searched_intervals + intervals, tol=self.tol)

            z = self._next_search_data(line_search)

        if model_selector is not None:
            self.intervals = union_all(result_intervals, tol=self.tol)

    def test(
        self, intervals=None, model_selector=None, tail="right", dps="auto", out_log='test_log.log', max_dps=5000
    ):
        """
        Perform selective statistical testing.

        Args:
            model_selector (callable, optional): Callable function which takes
                a selected model (any) as single argument, and returns True
                if the model is used for the testing, and False otherwise.
                This option is valid after calling ``self.parametric_search()``
                with model_selector None.
            tail (str, optional): 'double' for double-tailed test, 'right' for
                right-tailed test, and 'left' for left-tailed test. Defaults to 'right'.
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

        stat = np.asarray(self.stat) ** 2
        chi_intervals = intersection(
            intervals, [[1e-5, INF]])
        chi_squared_intervals = np.power(chi_intervals, 2)
        F = tc2_cdf_mpmath(stat, chi_squared_intervals, self.degree,
                           dps=dps, max_dps=max_dps, out_log=out_log)

        return calc_pvalue(F, tail=tail)

    def only_check_reject_or_not(
        self, algorithm, model_selector, significance_level=0.05, tol=1e-10, step=1e-10, tail="double", popmean=0, dps="auto", out_log='test_log.log', max_dps=5000
    ):
        """
        Only check whether the null hypothesis is rejected or not in selective statistical test.

        Args:
            algorithm (callable): Callable function which takes two vectors (`a`, `b`)
                and a scalar `z` that can satisfy `data = a + b * z`
                as arguments, and returns the selected model (any) and
                the truncation intervals (array-like). A closure function might be
                helpful to implement this.
            model_selector (callable): Callable function which takes
                a selected model (any) as single argument, and returns True
                if the model is used for the testing, and False otherwise.
            significance_level (float, optional): Significance level value for
                selective statistical tests. Defaults to 0.05.
            tol (float, optional): Tolerance error parameter. Defaults to 1e-10.
            step (float, optional): Step width for next search. Defaults to 1e-10.
            tail (str, optional): 'double' for double-tailed test, 'right' for
                right-tailed test, and 'left' for left-tailed test. Defaults to 'double'.
            popmean (float, optional): Population mean of `η^T x` under null hypothesis.
                Defaults to 0.
            dps (int, str, optional): dps value for mpmath. Set 'auto' to select dps
                automatically. Defaults to 'auto'.
            max_dps (int, optional): Maximum dps value for mpmath. This option is valid
                when `dps` is set to 'auto'. Defaults to 5000.

        Returns:
            (boolean, float, float): (reject or not, lower of p-value, upper of p-value)
        """
        self.tol = tol
        self.step = step
        self.searched_intervals = list()
        truncated_intervals = list()

        self.count = 0
        self.detect_count = 0

        stat = float(self.stat) ** 2

        z = self.stat
        while True:
            if self.count > 1e5:
                raise Exception(
                    'The number of searches exceeds 10,000 times, suggesting an infinite loop.')
            self.count += 1

            model, interval = algorithm(self.z, self.c, z)
            interval = np.asarray(interval)
            intervals = _interval_to_intervals(interval)

            if model_selector(model):
                truncated_intervals += intervals
                self.detect_count += 1

            self.searched_intervals = union_all(
                self.searched_intervals + intervals, tol=self.tol)

            unsearched_intervals = not_(self.searched_intervals)
            s = intersection(unsearched_intervals, [
                             NINF, float(self.stat)])[-1][1]
            e = intersection(unsearched_intervals, [
                             float(self.stat), INF])[0][0]

            sup_intervals = union_all(
                truncated_intervals + [[NINF, s]], tol=self.tol)
            inf_intervals = union_all(
                truncated_intervals + [[e, INF]], tol=self.tol)

            chi_sup_intervals = intersection(
                sup_intervals, [[1e-5, INF]])
            chi_inf_intervals = intersection(
                inf_intervals, [[1e-5, INF]])

            chi_squared_sup_intervals = np.power(chi_sup_intervals, 2)
            chi_squared_inf_intervals = np.power(chi_inf_intervals, 2)

            sup_F = tc2_cdf_mpmath(stat, chi_squared_sup_intervals, self.degree,
                                   dps=dps, max_dps=max_dps, out_log=out_log)
            inf_F = tc2_cdf_mpmath(stat, chi_squared_inf_intervals, self.degree,
                                   dps=dps, max_dps=max_dps, out_log=out_log)

            inf_p, sup_p = calc_p_range(inf_F, sup_F, tail=tail)

            if sup_p <= significance_level:
                return True, inf_p, sup_p
            if inf_p > significance_level:
                return False, inf_p, sup_p

            z_l = s - self.step
            z_r = e + self.step

            if float(self.stat) - z_l < z_r - float(self.stat):
                if z_l >= 1e-5:
                    z = z_l
            else:
                z = z_r
