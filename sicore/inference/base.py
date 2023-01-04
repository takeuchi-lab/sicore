import math
import random
import numpy as np
from scipy.stats import norm, ttest_ind, ttest_1samp

from ..utils import is_int_or_float

from typing import List, Any, Dict
from dataclasses import dataclass


@dataclass
class SelectiveInferenceResult():
    """
    This class contains the results of selective inference.

    Attributes:
        stat (float): Observed value of test statistic.
        alpha (float): Significance level.
        p_value (float | None): p-value from test performed.
        inf_p (float): Lower bound of p-value.
        sup_p (float): Upper bound of p-value.
        reject_or_not (bool): Whether or not to reject the null hypothesis.
        truncated_intervals (List[List[float]]): Intervals from which the selected model is obtained.
        search_count (int): Number of times the truncated intervals were searched.
        detect_count (int): Number of times that the same model as the observed model was obtained.
        selected_model (Any | None): The model selected for the observed test statistic.
        mappings (Dict[tuple[float], Any] | None): A dictionary that holds the model obtained at any point.
    """
    stat: float
    alpha: float
    p_value: float | None
    inf_p: float
    sup_p: float
    reject_or_not: bool
    truncated_intervals: List[List[float]]
    search_count: int
    detect_count: int
    selected_model: Any | None
    mappings: Dict[tuple[float], Any] | None


INF = float('inf')
NINF = -INF

random.seed(0)


def calc_pvalue(F, tail='double'):
    """
    Calculate p-value.

    Args:
        F (float): CDF value at the observed test statistic.
        tail (str, optional): Set 'double' for double-tailed test, 'right' for right-tailed test, and 'left' for left-tailed test. Defaults to 'double'.

    Returns:
        float: p-value
    """
    if tail == 'double':
        return float(2 * min(F, 1 - F))
    elif tail == 'right':
        return float(1 - F)
    elif tail == 'left':
        return float(F)


def calc_p_range(inf_F, sup_F, tail='double'):
    """
    Calculate possible range of p-value.

    Args:
        inf_F (float): Infimum of CDF value at the observed test statistic.
        sup_F (float): Supremum of CDF value at the observed test statistic.
        tail (str, optional): Set 'double' for double-tailed test, 'right' for right-tailed test, and 'left' for left-tailed test. Defaults to 'double'.

    Returns:
        (float, float): (Infimum of p-value, Supremum of p-value)
    """
    if tail == 'double':
        sup_p = float(2 * min(sup_F, 1 - inf_F))
        inf_p = float(2 * min(inf_F, 1 - sup_F))
    elif tail == 'right':
        sup_p = float(1 - inf_F)
        inf_p = float(1 - sup_F)
    elif tail == 'left':
        sup_p = float(sup_F)
        inf_p = float(inf_F)
    return (inf_p, sup_p)


def standardize(x, mean=0, var=1):
    """
    Standardize a random variable.

    Args:
        x (float, list): The value of random variable.
        mean (float, optional): Mean. Defaults to 0.
        var (float, optional): Variance. Defaults to 1.
    """
    sd = math.sqrt(var)
    return (np.asarray(x) - mean) / sd


def one_sample_test(data, popmean, var=None, tail='double'):
    """
    One sample hypothesis testing for population mean.

    var=float: Z-test
    var=None: T-test

    Args:
        data (array-like): Dataset.
        popmean (float): Population mean of `data` under null hypothesis is
            true.
        var (float, optional): Known population variance of the dataset. If
            None, the population variance is unknown. Defaults to None.
        tail (str, optional): 'double' for double-tailed test, 'right' for
            right-tailed test, and 'left' for left-tailed test. Defaults to
            'double'.

    Returns:
        float: p-value
    """
    if var is None:
        pvalue, stat = ttest_1samp(data, popmean)
        F = pvalue / 2 if stat < 0 else 1 - pvalue / 2
        return calc_pvalue(F, tail=tail)
    else:
        estimator = np.mean(data)
        var = var / len(data)
        stat = standardize(estimator, popmean, var)
        F = norm.cdf(stat)
        return calc_pvalue(F, tail=tail)


def two_sample_test(data1, data2, var=None, equal_var=True, tail='double'):
    """
    Two sample hypothesis testing for the difference between population means.

    var=float, list: Z-test
    var=None & equal_var=True: T-test
    var=None & equal_var=False: Welch's T-test

    Args:
        data1 (array-like): Dataset1.
        data2 (array-like): Dataset2.
        var (float, list, optional): Known population variance of each dataset
            in the form of single value or tuple `(var1, var2)`. If None, the
            population variance is unknown. Defaults to None.
        equal_var (bool, optional): If True, two population variances are
            assumed to be the same. Defaults to True.
        tail (str, optional): 'double' for double-tailed test, 'right' for
            right-tailed test, and 'left' for left-tailed test. Defaults to
            'double'.

    Returns:
        float: p-value
    """
    if var is None:
        pvalue, stat = ttest_ind(data1, data2, equal_var=equal_var)
        F = pvalue / 2 if stat < 0 else 1 - pvalue / 2
        return calc_pvalue(F, tail=tail)
    else:
        if is_int_or_float(var):
            var1 = var2 = var
        else:
            var1, var2 = var

        estimator = np.mean(data1) - np.mean(data2)
        var = var1 / len(data1) + var2 / len(data2)
        stat = standardize(estimator, var=var)
        F = norm.cdf(stat)
        return calc_pvalue(F, tail=tail)
