import math
import random
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm, ttest_ind, ttest_1samp
from ..cdf_mpmath import tn_cdf_mpmath as tn_cdf, tc2_cdf_mpmath as tc2_cdf, chi2_cdf_mpmath as chi2_cdf
from ..intervals import intersection, not_, poly_lt_zero, union_all
from ..utils import is_int_or_float
from .base import *


class PCI(ABC):
    """
    Base inference class for a test statistic which follows normal
    distribution under null.

    Args:
        data (array-like): Observation data of length `N`.
        var (float, list): Value of known variance, or `N` * `N` covariance matrix.
        eta (array-like): `N` dimensional vector of test direction.
    """
    def __init__(self, data, var, eta, delta):
        self.data = data.T.flatten()
        self.var = var
        self.eta = eta
        self.delta = delta
        self.length = data.shape[0]
        
        if is_int_or_float(var):
            self.cov = var * np.identity(self.length)
        else:
            self.cov = np.asarray(var)
        
        if is_int_or_float(delta):
            self.stat = 1
            self.sigma_eta = 1
            self.eta_sigma_eta = 1
        else:
            self.xi = np.identity(data.shape[1])
            self.stat = np.dot(np.kron(delta, eta).T, self.data)[0]
            sig_eta = np.dot(self.cov, eta)
            xi_del = np.dot(self.xi, delta)
            self.sigma_eta = np.kron(xi_del, sig_eta)
            self.eta_sigma_eta = np.dot(delta.T, xi_del)*np.dot(eta.T, sig_eta)

    @abstractmethod
    def test(self, *args, **kwargs):
        """Perform statistical testing."""
        pass


class NaivePCINorm(PCI):
    """
    Naive inference for a test statistic which follows normal distribution under null.

    Args:
        data (array-like): Observation data of length `N`.
        var (float, list): Value of known variance, or `N` * `N`covariance matrix.
        eta (array-like): `N` dimensional vector of test direction. Unlike ``SelectiveInference.test()``, the vector input here has to be constructed without taking absolute value of the test statistic.
    """
    def test(self, tail='double', popmean=0):
        """
        Perform naive statistical testing.

        Args:
            tail (str, optional): 'double' for double-tailed test, 'right' for right-tailed test, and 'left' for left-tailed test. Defaults to 'double'.
            popmean (float, optional): Population mean of `η'x` under null hypothesis is true. Defaults to 0.

        Returns:
            float: p-value
        """
        stat = standardize(self.stat, popmean, self.eta_sigma_eta)
        F = norm.cdf(stat)
        return calc_pvalue(F, tail=tail)


class SelectivePCINorm(PCI):
    """
    Selective inference for a test statistic which follows normal distribution under null. 
    Use this class if you already have truncation intervals at hand.

    Args:
        data (array-like): Observation data of length `N`.
        var (float, list): Value of known variance, or `N` * `N` covariance matrix.
        eta (array-like): `N` dimensional vector of test direction.
    """
    def __init__(self, data, var, eta, delta):
        super().__init__(data, var, eta, delta)
        if is_int_or_float(delta):
            self.c = 1
            self.z = 1
        else:
            self.c = (self.sigma_eta / self.eta_sigma_eta).reshape(self.sigma_eta.shape[0])
            self.z = data.T.flatten() - self.stat * self.c


    def test(self, intervals, tail='double', popmean=0, dps='auto', out_log='test_log.log', max_dps=5000):
        """
        Perform selective statistical testing.

        Args:
            interval (array-like): Truncation interval [L, U] or [[L1, U1], [L2, U2],...].
            model_selector (callable): Callable function which takes a selected model (any) as single argument, and returns True if the model is used for the testing, and False otherwise.
                This option is valid after calling ``self.parametric_search()``.
            tail (str, optional): 'double' for double-tailed test, 'right' for right-tailed test, and 'left' for left-tailed test. 
                Defaults to 'double'.
            popmean (float, optional): Population mean of `η^T x` under null hypothesis is true. 
                Defaults to 0.
            dps (int, str, optional): dps value for mpmath. 
                Set 'auto' to select dps automatically. 
                Defaults to 'auto'.
            max_dps (int, optional): Maximum dps value for mpmath. 
                This option is valid when `dps` is set to 'auto'.
                Defaults to 5000.

        Returns:
            float: p-value
        """

        self.interval = np.asarray(intervals)
        
        stat = standardize(self.stat, popmean, self.eta_sigma_eta)
        norm_intervals = standardize(self.interval, popmean, self.eta_sigma_eta)
        F = tn_cdf(stat, norm_intervals, dps=dps, max_dps=max_dps, out_log=out_log)
        return calc_pvalue(F, tail=tail)


class SelectivePCINormSE(SelectivePCINorm):
    """
    Selective inference for a test statistic which follows normal distribution under null. 
    Truncation intervals are calculated from selection events using the method proposed by Lee et al.

    Args:
        data (array-like): Observation data of length `N`.
        var (float, list): Value of known variance, or `N` * `N` covariance matrix.
        eta (array-like): `N` dimensional vector of test direction.
        init_lower (float, optional): Initial lower interval. 
            Defaults to -inf.
        init_upper (float, optional): Initial upper interval. 
            Defaults to inf.
    """
    def __init__(self, data, var, eta, delta, init_lower=NINF, init_upper=INF):
        super().__init__(data, var, eta, delta)
        self.__lower = init_lower
        self.__upper = init_upper
        self.__concave_intervals = []
        self.summary = {
            'linear': 0,
            'convex': 0,
            'concave': 0
        }


    def parametric_test(self, interval, tail='double', popmean=0, dps='auto'):
        return self.test(interval, tail=tail, popmean=popmean, dps=dps)


    def add_selection_event(self, A=None, b=None, c=None):
        """
        Add a selection event `{x'Ax + b'x + c ≦ 0}`.

        Args:
            A (array-like, optional): `N`*`N` matrix.
                Set None if `A` is unused.
                Defaults to None.
            b (array-like, optional): `N` dimensional vector. 
                Set None if `b` is unused. 
                Defaults to None.
            c (float, optional): Constant.
                Set None if `c` is unused.
                Defaults to None.
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
        Truncate the interval with a quadratic inequality `ατ^2 + βτ + γ ≦ 0`, where `τ` is test statistic, and `β`, `γ` are function of `c`, `z` respectively.
        We can also use a auadratic inequality `ατ^2 + κτ + λ ≦ 0`, where `κ`, `λ` are function of `c`, `x`.
        This method truncates the interval only when the inequality is convex in order to reduce calculation cost.
        Truncation intervals for concave inequalities are stored in ``self.__concave_intervals``.
        The final truncation intervals are calculated when ``self.get_intervals()`` is called.
        
        Args:
            a (float): `α`. Set 0 if the inequality is linear.
            b (float): `β` or `κ`.
            c (float): `γ` or `λ`.
            tau (bool, optional): Set False when the inputs are `β` and `γ`, and True when they are `κ` and `λ`.
                Defaults to False.

        Raises:
            ValueError: If the test direction of interest does not intersect with the inequality or the polytope.
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
            self.summary['linear'] += 1
        elif a > 0:
            disc = b**2 - 4 * a * c  # discriminant
            # if -threshold < disc < threshold:
            #     disc = 0
            if disc <= 0:
                raise ValueError(
                    "Test direction of interest does not "
                    "intersect with the inequality."
                )
            self.__lower = max(self.__lower, (-b - math.sqrt(disc)) / (2 * a) + tau)
            self.__upper = min(self.__upper, (-b + math.sqrt(disc)) / (2 * a) + tau)
            self.summary['convex'] += 1
        else:
            disc = b ** 2 - 4 * a * c  # discriminant
            # if -threshold < disc < threshold:
            #     disc = 0
            if disc <= 0:
                return

            lower = (-b + math.sqrt(disc)) / (2 * a) + tau
            upper = (-b - math.sqrt(disc)) / (2 * a) + tau
            self.__concave_intervals.append((lower, upper))
            self.summary['concave'] += 1

        if self.__lower >= self.__upper:
            raise ValueError(
                "Test direction of interest does not intersect "
                "with the polytope."
            )

    def get_intervals(self):
        """
        Get truncation intervals.

        Returns:
            list: List of truncation intervals [[L1, U1], [L2, U2],...].
        """
        intervals = [[self.__lower, self.__upper], ]

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
                        intervals = intervals[:i + 1]
                        break
                    elif intervals[i][0] < lower < intervals[i][1]:
                        intervals = intervals[:i + 1]
                        intervals[-1][1] = lower
                        break
            elif intervals[0][0] < lower and upper < intervals[-1][1]:
                # truncate the middle part of the intervals
                for i in range(len(intervals)):
                    if upper <= intervals[i][0]:
                        right_intervals = intervals[i:]
                        break
                    elif intervals[i][0] < upper < intervals[i][1]:
                        right_intervals = [[upper, intervals[i][1]]] + intervals[i + 1:]
                        break
                for i in range(len(intervals) - 1, -1, -1):
                    if intervals[i][1] <= lower:
                        left_intervals = intervals[:i + 1]
                        break
                    elif intervals[i][0] < lower < intervals[i][1]:
                        left_intervals = intervals[:i] + [[intervals[i][0], lower]]
                        break
                intervals = left_intervals + right_intervals
            elif lower <= intervals[0][0] and intervals[-1][1] <= upper:
                raise ValueError(
                    "Test direction of interest does not "
                    "intersect with the polytope."
                )

        return intervals

    def test(self, intervals, tail='double', popmean=0, dps='auto', out_log='test_log.log', max_dps=5000):
        """
        Perform selective statistical testing.

        Args:
            tail (str, optional): 'double' for double-tailed test, 'right' for right-tailed test, and 'left' for left-tailed test. 
                Defaults to 'double'.
            popmean (float, optional): Population mean of `η'x` under null hypothesis is true. Defaults to 0.
            dps (int, str, optional): dps value for mpmath. Set 'auto' to select dps automatically. Defaults to 'auto'.
            max_dps (int, optional): Maximum dps value for mpmath.
                This option is valid when `dps` is set to 'auto'. 
                Defaults to 5000.

        Returns:
            float: p-value
        """
        
        return super().test(intervals, tail=tail, popmean=popmean, dps=dps, out_log=out_log)


class NaivePCIChiSquared(PCI):
    """
    Naive inference for a test statistic which follows chi squared distribution under null.

    Args:
        data (array-like): Observation data of length `N`.
        var (float, list): Value of known variance, or `N`*`N`covariance matrix.
        eta (array-like): `N` dimensional vector of test direction. Unlike ``SelectiveInference.test()``, the vector input here has to be constructed without taking absolute value of the test statistic.
    """
    def test(self, data, tail='double'):
        """
        chi2 test
        """
        if is_int_or_float(self.var):
            self.cov = self.var * np.identity(data.shape[0])
        else:
            self.cov = np.asarray(self.var)
        self.sigma_eta = np.dot(self.cov, self.eta)
        self.eta_sigma_eta = np.dot(self.eta.T, self.sigma_eta)
        sigma_hat = self.eta_sigma_eta / np.dot(self.eta.T, self.eta)

        p = np.dot(self.eta, self.eta.T) / np.dot(self.eta.T, self.eta)
        eta_data = np.dot(p, data)
        P_vecx = eta_data.T.flatten()
        self.stat = np.linalg.norm(P_vecx, ord=2) / math.sqrt(sigma_hat[0][0])
        
        chi = chi2_cdf(self.stat ** 2, data.shape[1])
        return calc_pvalue(chi, tail=tail)


class SelectivePCIChiSquared(PCI):
    """
    Selective inference for a test statistic which follows chi squared distribution under null. 
    Use this class if you already have truncation intervals at hand.

    Args:
        data (array-like): Observation data of length `N`.
        var (float, list): Value of known variance, or `N` * `N` covariance matrix.
        eta (array-like): `N` dimensional vector of test direction.
    """
    def __init__(self, data, var, eta, delta):
        super().__init__(data, var, eta, delta)
        if is_int_or_float(delta):
            # TODO: 1 の理由
            self.c = 1
            self.z = 1
        else:
            self.c = (self.sigma_eta / self.eta_sigma_eta).reshape(self.sigma_eta.shape[0])
            self.z = data.T.flatten() - self.stat * self.c


    def test(self, intervals, tail='double', popmean=0, dps='auto', out_log='test_log.log'):
        """
        Perform selective statistical testing.

        Args:
            interval (array-like): Truncation interval [L, U] or [[L1, U1], [L2, U2],...].
            tail (str, optional): 'double' for double-tailed test, 'right' for right-tailed test, and 'left' for left-tailed test. 
                Defaults to 'double'.
            popmean (float, optional): Population mean of `η^T x` under null hypothesis is true. 
                Defaults to 0.
            dps (int, str, optional): dps value for mpmath. 
                Set 'auto' to select dps automatically. 
                Defaults to 'auto'.

        Returns:
            float: p-value
        """

        self.interval = np.asarray(intervals)
        
        stat = standardize(self.stat, popmean, self.eta_sigma_eta)
        norm_intervals = standardize(self.interval, popmean, self.eta_sigma_eta)
        F = tn_cdf(stat, norm_intervals, dps=dps, out_log=out_log)
        return calc_pvalue(F, tail=tail)


class SelectivePCIChiSquaredSE(SelectivePCIChiSquared):
    """
    Selective inference for a test statistic which follows chi squared distribution under null. 
    Truncation intervals are calculated from selection events using the method proposed by Lee et al.

    Args:
        data (array-like): Observation data of length `N`.
        var (float, list): Value of known variance, or `N` * `N` covariance matrix.
        eta (array-like): `N` dimensional vector of test direction.
        init_lower (float, optional): Initial lower interval. 
            Defaults to -inf.
        init_upper (float, optional): Initial upper interval. 
            Defaults to inf.
    """
    def __init__(self, data, var, eta, delta, init_lower=NINF, init_upper=INF):
        super().__init__(data, var, eta, delta)
        self.__lower = init_lower
        self.__upper = init_upper
        self.__concave_intervals = []
        self.summary = {
            'linear': 0,
            'convex': 0,
            'concave': 0
        }


    def make_sigma_hat(self, data, eta):
        if isinstance(self.var, (int, float)):
            self.cov = self.var * np.identity(data.shape[0])
        else:
            self.cov = np.asarray(self.var)
        self.sigma_eta = np.dot(self.cov, eta)
        self.eta_sigma_eta = np.dot(eta.T, self.sigma_eta)
        sigma_hat = self.eta_sigma_eta/np.dot(eta.T, eta)

        p = np.dot(eta, eta.T)/np.dot(eta.T, eta)
        eta_data = np.dot(p, data)
        P_vecx = eta_data.T.flatten()
        self.stat = np.linalg.norm(P_vecx, ord=2)/math.sqrt(sigma_hat[0][0])
        self.z = self.data - P_vecx
        self.c = P_vecx/np.linalg.norm(P_vecx, ord=2)

        return sigma_hat


    def test_chi2(self, data, intervals, tail='double', dps='auto', out_log='test_log.log'):
        """
        truncated chi2 test

        """

        self.interval = np.asarray(intervals)
        chi = tc2_cdf(self.stat**2, np.power(self.interval, 2), data.shape[1], dps=dps, out_log=out_log)
        return calc_pvalue(chi, tail=tail)


    def parametric_test(self, interval, tail='double', popmean=0, dps='auto'):
        return self.test(interval, tail=tail, popmean=popmean, dps=dps)


    def add_selection_event(self, A=None, b=None, c=None):
        """
        Add a selection event `{x'Ax + b'x + c ≦ 0}`.

        Args:
            A (array-like, optional): `N`*`N` matrix.
                Set None if `A` is unused.
                Defaults to None.
            b (array-like, optional): `N` dimensional vector. 
                Set None if `b` is unused. 
                Defaults to None.
            c (float, optional): Constant.
                Set None if `c` is unused.
                Defaults to None.
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
        Truncate the interval with a quadratic inequality `ατ^2 + βτ + γ ≦ 0`, where `τ` is test statistic, and `β`, `γ` are function of `c`, `z` respectively.
        We can also use a auadratic inequality `ατ^2 + κτ + λ ≦ 0`, where `κ`, `λ` are function of `c`, `x`.
        This method truncates the interval only when the inequality is convex in order to reduce calculation cost.
        Truncation intervals for concave inequalities are stored in ``self.__concave_intervals``.
        The final truncation intervals are calculated when ``self.get_intervals()`` is called.
        
        Args:
            a (float): `α`. Set 0 if the inequality is linear.
            b (float): `β` or `κ`.
            c (float): `γ` or `λ`.
            tau (bool, optional): Set False when the inputs are `β` and `γ`, and True when they are `κ` and `λ`.
                Defaults to False.

        Raises:
            ValueError: If the test direction of interest does not intersect with the inequality or the polytope.
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
            self.summary['linear'] += 1
        elif a > 0:
            disc = b**2 - 4 * a * c  # discriminant
            # if -threshold < disc < threshold:
            #     disc = 0
            if disc <= 0:
                raise ValueError(
                    "Test direction of interest does not "
                    "intersect with the inequality."
                )
            self.__lower = max(self.__lower, (-b - math.sqrt(disc)) / (2 * a) + tau)
            self.__upper = min(self.__upper, (-b + math.sqrt(disc)) / (2 * a) + tau)
            self.summary['convex'] += 1
        else:
            disc = b ** 2 - 4 * a * c  # discriminant
            # if -threshold < disc < threshold:
            #     disc = 0
            if disc <= 0:
                return

            lower = (-b + math.sqrt(disc)) / (2 * a) + tau
            upper = (-b - math.sqrt(disc)) / (2 * a) + tau
            self.__concave_intervals.append((lower, upper))
            self.summary['concave'] += 1

        if self.__lower >= self.__upper:
            raise ValueError(
                "Test direction of interest does not intersect "
                "with the polytope."
            )

    def get_intervals(self):
        """
        Get truncation intervals.

        Returns:
            list: List of truncation intervals [[L1, U1], [L2, U2],...].
        """
        intervals = [[self.__lower, self.__upper], ]

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
                        intervals = intervals[:i + 1]
                        break
                    elif intervals[i][0] < lower < intervals[i][1]:
                        intervals = intervals[:i + 1]
                        intervals[-1][1] = lower
                        break
            elif intervals[0][0] < lower and upper < intervals[-1][1]:
                # truncate the middle part of the intervals
                for i in range(len(intervals)):
                    if upper <= intervals[i][0]:
                        right_intervals = intervals[i:]
                        break
                    elif intervals[i][0] < upper < intervals[i][1]:
                        right_intervals = [[upper, intervals[i][1]]] + intervals[i + 1:]
                        break
                for i in range(len(intervals) - 1, -1, -1):
                    if intervals[i][1] <= lower:
                        left_intervals = intervals[:i + 1]
                        break
                    elif intervals[i][0] < lower < intervals[i][1]:
                        left_intervals = intervals[:i] + [[intervals[i][0], lower]]
                        break
                intervals = left_intervals + right_intervals
            elif lower <= intervals[0][0] and intervals[-1][1] <= upper:
                raise ValueError(
                    "Test direction of interest does not "
                    "intersect with the polytope."
                )

        return intervals
