# quant_elements_lib/risk_evaluation/__init__.py

"""
Risk Evaluation Submodule

This submodule provides functions for evaluating risk models, including loss functions,
matrix evaluation techniques, and other ancillary tests related to risk and portfolio construction.
"""

from .loss_functions import (
    calculate_qlike_loss,
    calculate_mse_variance_ratio_loss
)
from .matrix_evaluation import (
    evaluate_random_portfolio_variance,
    evaluate_mvo_portfolio_variance,
    calculate_qdist_likelihood,
    test_minimum_variance_portfolios,
    calculate_malv_statistic
)
from .ancillary_tests import (
    calculate_fmp_turnover,
    calculate_beta_prediction_error
)

__all__ = [
    # From loss_functions
    'calculate_qlike_loss',
    'calculate_mse_variance_ratio_loss',
    # From matrix_evaluation
    'evaluate_random_portfolio_variance',
    'evaluate_mvo_portfolio_variance',
    'calculate_qdist_likelihood',
    'test_minimum_variance_portfolios',
    'calculate_malv_statistic',
    # From ancillary_tests
    'calculate_fmp_turnover',
    'calculate_beta_prediction_error',
]
