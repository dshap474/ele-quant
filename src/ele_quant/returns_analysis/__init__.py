from .calculations import (
    calculate_simple_returns,
    calculate_dividend_adjusted_returns,
    calculate_log_returns,
    calculate_excess_returns,
    calculate_compounded_return_series,
    calculate_total_compounded_return,
)
from .price_estimation import estimate_true_price_roll_model
from .currency_hedging import calculate_hedged_returns, calculate_forward_premium


__all__ = [
    "estimate_true_price_roll_model",
    "calculate_simple_returns",
    "calculate_dividend_adjusted_returns",
    "calculate_log_returns",
    "calculate_excess_returns",
    "calculate_compounded_return_series",
    "calculate_total_compounded_return",
    "calculate_hedged_returns",
    "calculate_forward_premium",
]
