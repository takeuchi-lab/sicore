from abc import ABC, abstractmethod
import math
import numpy as np
from ..utils import is_int_or_float
from ..intervals import intersection, not_, poly_lt_zero, union_all, _interval_to_intervals
from ..cdf_mpmath import tn_cdf_mpmath as tn_cdf
from .base import *

from scipy.stats import norm
from typing import Callable, List, Dict, Tuple, Type


class InferenceNorm(ABC):
    """
    Base inference class for a test statistic which follows normal distribution under null.

    Args:
        data (array-like): Observation data of length `N`.
        var (float, array-like): Value of known variance, or `N`*`N` covariance matrix.
        eta (array-like): Contrast vector of length `N`.
        use_tf (boolean, optional): Whether to use tensorflow or not. Defaults to False.
    """

    def __init__(self, data, var, eta, use_tf=False):
        self.data = data
        self.eta = eta
        self.length = len(data)

        if use_tf:
            try:
                import tensorflow as tf
            except ModuleNotFoundError:
                raise Exception('use_tf is True, but package not found.')

            assert isinstance(data, tf.Tensor)

            if is_int_or_float(var):
                self.cov = var * tf.linalg.diag(tf.ones_like(data))
            else:
                self.cov = tf.constant(var)
            self.stat = tf.tensordot(eta, data, axes=1)
            self.sigma_eta = tf.tensordot(self.cov, eta, axes=1)
            self.eta_sigma_eta = tf.tensordot(eta, self.sigma_eta, axes=1)

        else:
            if is_int_or_float(var):
                self.cov = var * np.identity(self.length)
            else:
                self.cov = np.asarray(var)
            self.stat = np.dot(eta, data)
            self.sigma_eta = np.dot(self.cov, eta)
            self.eta_sigma_eta = np.dot(eta, self.sigma_eta)

    @abstractmethod
    def test(self, *args, **kwargs):
        """Perform statistical testing."""
        pass


class NaiveInferenceNorm(InferenceNorm):
    """
    Naive inference for a test statistic which follows normal distribution under null.

    Args:
        data (array-like): Observation data of length `N`.
        var (float, array-like): Value of known variance, or `N`*`N`covariance matrix.
        eta (array-like): Contrast vector of length `N`.
        use_tf (boolean, optional): Whether to use tensorflow or not. Defaults to False.
    """

    def test(self, tail="double", popmean=0):
        """
        Perform naive statistical testing.

        Args:
            tail (str, optional): 'double' for double-tailed test, 'right' for
                right-tailed test, and 'left' for left-tailed test. Defaults to 'double'.
            popmean (float, optional): Population mean of `η'x` under null hypothesis.
                Defaults to 0.

        Returns:
            float: p-value
        """
        stat = standardize(self.stat, popmean, self.eta_sigma_eta)
        F = norm.cdf(stat)
        return calc_pvalue(F, tail=tail)


class SelectiveInferenceNorm(InferenceNorm):
    """
    Selective inference for a test statistic which follows normal distribution under null.

    Args:
        data (array-like): Observation data of length `N`.
        var (float, array-like): Value of known variance, or `N`*`N` covariance matrix.
        eta (array-like): Contrast vector of length `N`.
        use_tf (boolean, optional): Whether to use tensorflow or not. Defaults to False.
    """

    def __init__(self, data, var, eta, use_tf=False):
        super().__init__(data, var, eta, use_tf)
        self.c = self.sigma_eta / self.eta_sigma_eta  # `b` vector in para si.
        self.z = data - self.stat * self.c  # `a` vecotr in para si.
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
            [[NINF, -float(max_tail)], [float(max_tail), INF]], tol=self.tol
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
        self, intervals=None, model_selector=None, tail="double", popmean=0, dps="auto", out_log='test_log.log', max_dps=5000
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
                right-tailed test, and 'left' for left-tailed test. Defaults to 'double'.
            popmean (float, optional): Population mean of `η^T x` under null hypothesis.
                Defaults to 0.
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

        stat = standardize(self.stat, popmean, self.eta_sigma_eta)
        norm_intervals = standardize(intervals, popmean, self.eta_sigma_eta)
        F = tn_cdf(stat, norm_intervals,
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
            (bool, float, float): (reject or not, lower of p-value, upper of p-value)
        """
        self.tol = tol
        self.step = step
        self.searched_intervals = list()
        truncated_intervals = list()

        self.count = 0
        self.detect_count = 0

        stat = standardize(self.stat, popmean, self.eta_sigma_eta)

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

            norm_sup_intervals = standardize(
                sup_intervals, popmean, self.eta_sigma_eta)
            norm_inf_intervals = standardize(
                inf_intervals, popmean, self.eta_sigma_eta)

            sup_F = tn_cdf(stat, norm_sup_intervals,
                           dps=dps, max_dps=max_dps, out_log=out_log)
            inf_F = tn_cdf(stat, norm_inf_intervals,
                           dps=dps, max_dps=max_dps, out_log=out_log)

            inf_p, sup_p = calc_p_range(inf_F, sup_F, tail=tail)

            if sup_p <= significance_level:
                return True, inf_p, sup_p
            if inf_p > significance_level:
                return False, inf_p, sup_p

            z_l = s - self.step
            z_r = e + self.step

            if float(self.stat) - z_l < z_r - float(self.stat):
                z = z_l
            else:
                z = z_r

    def calc_range_of_cdf_value(self, stat, truncated_intervals, searched_intervals):

        unsearched_intervals = not_(searched_intervals)
        s = intersection(unsearched_intervals, [
            NINF, float(stat)])[-1][-1]
        e = intersection(unsearched_intervals, [
            float(stat), INF])[0][0]

        sup_intervals = union_all(
            truncated_intervals + [[NINF, s]], tol=self.tol)
        inf_intervals = union_all(
            truncated_intervals + [[e, INF]], tol=self.tol)

        norm_sup_intervals = standardize(
            sup_intervals, 0, self.eta_sigma_eta)
        norm_inf_intervals = standardize(
            inf_intervals, 0, self.eta_sigma_eta)

        sup_F = tn_cdf(stat, norm_sup_intervals,
                       dps=self.dps, max_dps=self.max_dps, out_log=self.out_log)
        inf_F = tn_cdf(stat, norm_inf_intervals,
                       dps=self.dps, max_dps=self.max_dps, out_log=self.out_log)
        return inf_F, sup_F

    def determine_next_search_data(self, choose_method, *args):
        if choose_method == 'near_stat':
            def method(z): return -np.abs(z - float(self.stat))
        if choose_method == 'high_pdf':
            def method(z): return norm.pdf(z, 0, np.sqrt(self.eta_sigma_eta))
        if choose_method == 'random':
            return random.choice(args)
        return max(args, key=method)

    def inference(
        self,
        algorithm: Callable[[np.ndarray, np.ndarray, float], Tuple[List[List[float]], Any]],
        model_selector: Callable[[Any], bool],
        significance_level: float = 0.05,
        tail: str = 'double',
        tol: float = 1e-10,
        step: float = 1e-10,
        check_only_reject_or_not: bool = False,
        over_conditioning: bool = False,
        line_search: bool = True,
        max_tail: float = 1e3,
        choose_method: str = 'near_stat',
        retain_selected_model: bool = False,
        retain_mappings: bool = False,
        dps: int | str = 'auto',
        max_dps: int = 5000,
        out_log: str = 'test_log.log'
    ) -> Type[SelectiveInferenceResult]:

        if over_conditioning and check_only_reject_or_not:
            raise Exception(
                'The two options, check_only_reject_or_not and over-conditioning, cannot be activated at the same time.'
            )

        if over_conditioning:
            result = self._over_conditioned_inference(
                algorithm, significance_level, tail, retain_selected_model,
                tol, dps, max_dps, out_log)
            return result

        if check_only_reject_or_not:
            result = self._rejectability_only_inference(
                algorithm, model_selector, significance_level, tail, choose_method,
                retain_selected_model, retain_mappings, tol, step,
                dps, max_dps, out_log)
        else:
            result = self._parametric_inference(
                algorithm, model_selector, significance_level, tail, line_search, max_tail,
                retain_selected_model, retain_mappings, tol, step, dps, max_dps, out_log)

        return result

    def _parametric_inference(
        self, algorithm, model_selector, significance_level=0.05, tail='double',
        line_search=True, max_tail=1000, retain_selected_model=False, retain_mappings=False,
        tol=1e-10, step=1e-10, dps='auto', max_dps=5000, out_log='test_log.log'
    ):

        self.tol = tol
        self.dps = dps
        self.max_dps = max_dps
        self.out_log = out_log

        self.searched_intervals = union_all(
            [[NINF, -float(max_tail)], [float(max_tail), INF]], tol=self.tol
        )
        self.step = step
        mappings = dict() if retain_mappings else None
        result_intervals = list()

        search_count = 0
        detect_count = 0

        stat = standardize(self.stat, 0, self.eta_sigma_eta)
        z = self._next_search_data(line_search)

        while True:

            if z is None:
                break

            search_count += 1
            if search_count > 1e6:
                raise Exception(
                    'The number of searches exceeds 100,000 times, suggesting an infinite loop.')

            model, interval = algorithm(self.z, self.c, z)
            interval = np.asarray(interval)
            intervals = _interval_to_intervals(interval)

            if retain_mappings:
                for interval in intervals:
                    interval = tuple(interval)
                    if interval in mappings:
                        raise Exception(
                            "An interval appeared a second time. Usually, numerical error "
                            "causes this exception. Consider increasing the tol parameter "
                            "or decreasing max_tail parameter to avoid it."
                        )
                    mappings[interval] = model

            if model_selector(model):
                selected_model = model if retain_selected_model else None
                result_intervals += intervals
                detect_count += 1

            self.searched_intervals = union_all(
                self.searched_intervals + intervals, tol=tol)

            z = self._next_search_data(line_search)

        truncated_intervals = union_all(result_intervals, tol=self.tol)

        p_value = self.test(truncated_intervals, dps=dps,
                            max_dps=max_dps, out_log=out_log)

        inf_F, sup_F = self.calc_range_of_cdf_value(
            stat, truncated_intervals, [[-float(max_tail), float(max_tail)]])
        inf_p, sup_p = calc_p_range(inf_F, sup_F, tail=tail)

        return SelectiveInferenceResult(
            stat, significance_level, p_value, inf_p, sup_p,
            (p_value <= significance_level),
            standardize(truncated_intervals, 0, self.eta_sigma_eta),
            search_count, detect_count, selected_model, mappings)

    def _rejectability_only_inference(
        self, algorithm, model_selector, significance_level=0.05, tail='double', choose_method='near_stat',
        retain_selected_model=False, retain_mappings=False, tol=1e-10, step=1e-10,
        dps='auto', max_dps=5000, out_log='test_log.log'
    ):
        self.tol = tol
        self.step = step
        self.searched_intervals = list()
        truncated_intervals = list()
        mappings = dict() if retain_mappings else None

        search_count = 0
        detect_count = 0

        stat = standardize(self.stat, 0, self.eta_sigma_eta)

        z = self.stat
        while True:
            if search_count > 1e6:
                raise Exception(
                    'The number of searches exceeds 100,000 times, suggesting an infinite loop.')
            search_count += 1

            model, interval = algorithm(self.z, self.c, z)
            interval = np.asarray(interval)
            intervals = _interval_to_intervals(interval)

            if retain_mappings:
                for interval in intervals:
                    interval = tuple(interval)
                    if interval in mappings:
                        raise Exception(
                            "An interval appeared a second time. Usually, numerical error "
                            "causes this exception. Consider increasing the tol parameter "
                            "or decreasing max_tail parameter to avoid it."
                        )
                    mappings[interval] = model

            if model_selector(model):
                selected_model = model if retain_selected_model else None
                truncated_intervals += intervals
                detect_count += 1

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

            norm_sup_intervals = standardize(
                sup_intervals, 0, self.eta_sigma_eta)
            norm_inf_intervals = standardize(
                inf_intervals, 0, self.eta_sigma_eta)

            sup_F = tn_cdf(stat, norm_sup_intervals,
                           dps=dps, max_dps=max_dps, out_log=out_log)
            inf_F = tn_cdf(stat, norm_inf_intervals,
                           dps=dps, max_dps=max_dps, out_log=out_log)

            inf_p, sup_p = calc_p_range(inf_F, sup_F, tail=tail)

            if sup_p <= significance_level:
                reject_or_not = True
                break
            if inf_p > significance_level:
                reject_or_not = False
                break

            z_l = s - self.step
            z_r = e + self.step

            z = self.determine_next_search_data(choose_method, z_l, z_r)

        return SelectiveInferenceResult(
            stat, significance_level, None, inf_p, sup_p, reject_or_not,
            standardize(union_all(truncated_intervals), 0, self.eta_sigma_eta),
            search_count, detect_count, selected_model, mappings)

    def _over_conditioned_inference(
        self, algorithm, significance_level=0.05, tail='double', retain_selected_model=False,
        tol=1e-10, dps='auto', max_dps=5000, out_log='test_log.log'
    ):

        self.tol = tol
        self.dps = dps
        self.max_dps = max_dps
        self.out_log = out_log

        stat = standardize(self.stat, 0, self.eta_sigma_eta)

        model, interval = algorithm(self.z, self.c, self.stat)
        interval = np.asarray(interval)
        intervals = _interval_to_intervals(interval)

        p_value = self.test(intervals)

        unsearched_intervals = not_(intervals)
        s = intersection(unsearched_intervals, [
            NINF, float(self.stat)])[-1][1]
        e = intersection(unsearched_intervals, [
            float(self.stat), INF])[0][0]

        sup_intervals = union_all(
            intervals + [[NINF, s]], tol=self.tol)
        inf_intervals = union_all(
            intervals + [[e, INF]], tol=self.tol)

        norm_sup_intervals = standardize(
            sup_intervals, 0, self.eta_sigma_eta)
        norm_inf_intervals = standardize(
            inf_intervals, 0, self.eta_sigma_eta)

        sup_F = tn_cdf(stat, norm_sup_intervals,
                       dps=dps, max_dps=max_dps, out_log=out_log)
        inf_F = tn_cdf(stat, norm_inf_intervals,
                       dps=dps, max_dps=max_dps, out_log=out_log)
        inf_p, sup_p = calc_p_range(inf_F, sup_F, tail=tail)

        return SelectiveInferenceResult(
            stat, significance_level, p_value, inf_p, sup_p,
            (p_value <= significance_level),
            standardize(intervals, 0, self.eta_sigma_eta), 1, 1,
            None, model if retain_selected_model else None)


class SelectiveInferenceNormSE(SelectiveInferenceNorm):
    """
    Selective inference for a test statistic which follows normal distribution under
    null. Truncation intervals are calculated from selection events using the method
        proposed by Lee et al.

    Args:
        data (array-like): Observation data of length `N`.
        var (float, array-like): Value of known variance, or `N`*`N` covariance matrix.
        eta (array-like): Contrast vector of length `N`.
        init_lower (float, optional): Initial lower interval. Defaults to -inf.
        init_upper (float, optional): Initial upper interval. Defaults to inf.
    """

    def __init__(self, data, var, eta, init_lower=NINF, init_upper=INF):
        super().__init__(data, var, eta)
        self.__lower = init_lower
        self.__upper = init_upper
        self.__concave_intervals = []
        self.summary = {"linear": 0, "convex": 0, "concave": 0}

    def add_selection_event(self, A=None, b=None, c=None):
        """
        Add a selection event `{x'Ax+b'x+c<=0}`.

        Args:
            A (array-like, optional): `N`*`N` matrix. Set None if `A` is unused.
                Defaults to None.
            b (array-like, optional): `N` dimensional vector. Set None if `b` is unused.
                Defaults to None.
            c (float, optional): Constant. Set None if `c` is unused. Defaults to None.
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

        self.cut_interval(alp, beta, gam)

    def cut_interval(self, a, b, c, tau=False):
        """
        Truncate the interval with a quadratic inequality `ατ^2+βτ+γ<=0`, where `τ` is
        test statistic, and `β`, `γ` are function of `c`, `z` respectively. We can also
        use a auadratic inequality `ατ^2+κτ+λ<=0`, where `κ`, `λ` are function of `c`,
        `x`. This method truncates the interval only when the inequality is convex in
        order to reduce calculation cost. Truncation intervals for concave inequalities
        are stored in ``self.__concave_intervals``. The final truncation intervals are
        calculated when ``self.get_intervals()`` is called.

        Args:
            a (float): `α`. Set 0 if the inequality is linear.
            b (float): `β` or `κ`.
            c (float): `γ` or `λ`.
            tau (bool, optional): Set False when the inputs are `β` and `γ`, and True
                when they are `κ` and `λ`. Defaults to False.

        Raises:
            ValueError: If the test direction of interest does not intersect with the
                inequality or the polytope.
        """
        tau = self.stat if tau else 0

        threshold = 1e-10
        if -threshold < a < threshold:
            a = 0
        if -threshold < b < threshold:
            b = 0
        if -threshold < c < threshold:
            c = 0

        if a == 0:
            if b == 0:
                if c <= 0:
                    return
                else:
                    raise ValueError(
                        "Test direction of interest does not "
                        "intersect with the inequality."
                    )
            elif b < 0:
                self.__lower = max(self.__lower, -c / b + tau)
            elif b > 0:
                self.__upper = min(self.__upper, -c / b + tau)
            self.summary["linear"] += 1
        elif a > 0:
            disc = b ** 2 - 4 * a * c  # discriminant
            # if -threshold < disc < threshold:
            #     disc = 0
            if disc <= 0:
                raise ValueError(
                    "Test direction of interest does not "
                    "intersect with the inequality."
                )
            self.__lower = max(
                self.__lower, (-b - math.sqrt(disc)) / (2 * a) + tau)
            self.__upper = min(
                self.__upper, (-b + math.sqrt(disc)) / (2 * a) + tau)
            self.summary["convex"] += 1
        else:
            disc = b ** 2 - 4 * a * c  # discriminant
            # if -threshold < disc < threshold:
            #     disc = 0
            if disc <= 0:
                return

            lower = (-b + math.sqrt(disc)) / (2 * a) + tau
            upper = (-b - math.sqrt(disc)) / (2 * a) + tau
            self.__concave_intervals.append((lower, upper))
            self.summary["concave"] += 1

        if self.__lower >= self.__upper:
            raise ValueError(
                "Test direction of interest does not intersect " "with the polytope."
            )

    def get_intervals(self):
        """
        Get truncation intervals.

        Returns:
            list: List of truncation intervals [[L1, U1], [L2, U2],...].
        """
        intervals = [
            [self.__lower, self.__upper],
        ]

        for lower, upper in self.__concave_intervals:
            if lower <= intervals[0][0] < upper < intervals[-1][1]:
                # truncate the left side of the intervals
                for i in range(len(intervals)):
                    if upper <= intervals[i][0]:
                        intervals = intervals[i:]
                        break
                    elif intervals[i][0] < upper < intervals[i][1]:
                        intervals = intervals[i:]
                        intervals[0][0] = upper
                        break
            elif intervals[0][0] < lower < intervals[-1][1] <= upper:
                # truncate the right side of the intervals
                for i in range(len(intervals) - 1, -1, -1):
                    if intervals[i][1] <= lower:
                        intervals = intervals[: i + 1]
                        break
                    elif intervals[i][0] < lower < intervals[i][1]:
                        intervals = intervals[: i + 1]
                        intervals[-1][1] = lower
                        break
            elif intervals[0][0] < lower and upper < intervals[-1][1]:
                # truncate the middle part of the intervals
                for i in range(len(intervals)):
                    if upper <= intervals[i][0]:
                        right_intervals = intervals[i:]
                        break
                    elif intervals[i][0] < upper < intervals[i][1]:
                        right_intervals = [[upper, intervals[i][1]]] + intervals[
                            i + 1:
                        ]
                        break
                for i in range(len(intervals) - 1, -1, -1):
                    if intervals[i][1] <= lower:
                        left_intervals = intervals[: i + 1]
                        break
                    elif intervals[i][0] < lower < intervals[i][1]:
                        left_intervals = intervals[:i] + \
                            [[intervals[i][0], lower]]
                        break
                intervals = left_intervals + right_intervals
            elif lower <= intervals[0][0] and intervals[-1][1] <= upper:
                raise ValueError(
                    "Test direction of interest does not "
                    "intersect with the polytope."
                )

        return intervals

    def test(self, intervals=None, tail='double', popmean=0, dps='auto', out_log='test_log.log', max_dps=5000):
        """
        Perform selective statistical testing.

        Args:
            tail (str, optional): 'double' for double-tailed test, 'right' for
                right-tailed test, and 'left' for left-tailed test. Defaults to
                'double'.
            popmean (float, optional): Population mean of `η'x` under null hypothesis.
                Defaults to 0.
            dps (int, str, optional): dps value for mpmath. Set 'auto' to select dps
                automatically. Defaults to 'auto'.
            max_dps (int, optional): Maximum dps value for mpmath. This option is valid
                when `dps` is set to 'auto'. Defaults to 5000.

        Returns:
            float: p-value
        """
        if intervals is None:
            self.add_interval(self.get_intervals())
            return super().test(tail=tail, popmean=popmean, dps=dps, max_dps=max_dps)
        else:
            return super().test(intervals=intervals, tail=tail, popmean=popmean, dps=dps, out_log=out_log)
