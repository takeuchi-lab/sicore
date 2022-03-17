import numpy as np


def false_positive_rate(pvalues, alpha=0.05):
    """
    Calculate false positive rate of p-values under null.

    Args:
        pvalues (array-like): List of p-values.
        alpha (float, optional): Significance level.

    Returns:
        float: False positive rate.
    """
    pvalues = np.array(pvalues)
    return np.count_nonzero(pvalues <= alpha) / len(pvalues)


def false_negative_rate(pvalues, alpha=0.05):
    """
    Calculate false negative rate of p-values under alternative.

    Args:
        pvalues (array-like): List of p-values.
        alpha (float, optional): Significance level.

    Returns:
        float: False negative rate.
    """
    pvalues = np.array(pvalues)
    return np.count_nonzero(pvalues > alpha) / len(pvalues)


def true_negative_rate(pvalues, alpha=0.05):
    """
    Calculate true negative rate of p-values under null.

    Args:
        pvalues (array-like): List of p-values.
        alpha (float, optional): Significance level.

    Returns:
        float: True negative rate.
    """
    return 1 - false_positive_rate(pvalues, alpha=alpha)


def true_positive_rate(pvalues, alpha=0.05):
    """
    Calculate true positive rate of p-values under alternative.

    Args:
        pvalues (array-like): List of p-values.
        alpha (float, optional): Significance level.

    Returns:
        float: True positive rate.
    """
    return 1 - false_negative_rate(pvalues, alpha=alpha)


type1_error_rate = false_positive_rate
type2_error_rate = false_negative_rate
power = true_positive_rate
