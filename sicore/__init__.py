"""
Core package for selective inference.
"""
from . import intervals, utils
from .cdf_mpmath import tc2_cdf_mpmath as tc2_cdf
from .cdf_mpmath import tf_cdf_mpmath as tf_cdf
from .cdf_mpmath import tn_cdf_mpmath as tn_cdf
from .cdf_mpmath import tt_cdf_mpmath as tt_cdf
from .evaluation import (
    false_negative_rate,
    false_positive_rate,
    power,
    true_negative_rate,
    true_positive_rate,
    type1_error_rate,
    type2_error_rate,
)
from .figures import FprFigure, PowerFigure, pvalues_hist, pvalues_qqplot
from .inference.base import (
    one_sample_test,
    two_sample_test,
)
from .inference.norm import (
    NaiveInferenceNorm,
    SelectiveInferenceNorm,
    SelectiveInferenceNormSE,
)
from .inference.pci import (
    NaivePCINorm,
    SelectivePCINorm,
    SelectivePCINormSE,
    NaivePCIChiSquared,
    SelectivePCIChiSquared,
    SelectivePCIChiSquaredSE,
)

__all__ = [
    "NaiveInferenceNorm",
    "SelectiveInferenceNorm",
    "SelectiveInferenceNormSE",
    "NaivePCINorm",
    "SelectivePCINorm",
    "SelectivePCINormSE",
    "NaivePCIChiSquared",
    "SelectivePCIChiSquared",
    "SelectivePCIChiSquaredSE",
    "one_sample_test",
    "two_sample_test",
    "tn_cdf",
    "tt_cdf",
    "tc2_cdf",
    "tf_cdf",
    "false_positive_rate",
    "false_negative_rate",
    "true_negative_rate",
    "true_positive_rate",
    "type1_error_rate",
    "type2_error_rate",
    "power",
    "pvalues_hist",
    "pvalues_qqplot",
    "FprFigure",
    "PowerFigure",
    "intervals",
    "utils",
]
