import math
import random
import numpy as np
from scipy.stats import norm, ttest_ind, ttest_1samp

from ..utils import is_int_or_float


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
