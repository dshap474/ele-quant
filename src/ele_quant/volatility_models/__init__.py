"""
Volatility Models for Quantitative Finance.

This module includes implementations of common volatility models like GARCH.
"""

from .garch import (
    garch_1_1_conditional_variance,
    garch_1_1_log_likelihood,
    estimate_garch_1_1_parameters,
)
from .realized_volatility import (
    calculate_realized_variance_from_sum_sq_returns,
    estimate_variance_mle_diffusion,
)
from .state_space_vol import (
    ewma_variance_forecast,
    muth_model_variance_estimation,
    harvey_shephard_volatility_estimation,
)

__all__ = [
    "ewma_variance_forecast",
    "muth_model_variance_estimation",
    "harvey_shephard_volatility_estimation",
    "garch_1_1_conditional_variance",
    "garch_1_1_log_likelihood",
    "estimate_garch_1_1_parameters",
    "calculate_realized_variance_from_sum_sq_returns",
    "estimate_variance_mle_diffusion",
]
