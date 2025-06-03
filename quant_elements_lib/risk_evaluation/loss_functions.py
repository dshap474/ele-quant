import pandas as pd
import numpy as np
from typing import Union

# Epsilon for numerical stability
EPSILON = 1e-8

def calculate_qlike_loss(realized_variances: pd.Series, predicted_variances: pd.Series) -> float:
    """
    Calculate the QLIKE loss function.

    The QLIKE loss function is a robust loss function for evaluating variance forecasts.
    It is defined as: (1/T) Σ (r_t²/σ̂_t² - log(r_t²/σ̂_t²) - 1).

    Parameters
    ----------
    realized_variances : pd.Series
        Series of realized variances (r_t²).
    predicted_variances : pd.Series
        Series of predicted variances (σ̂_t²).

    Returns
    -------
    float
        The QLIKE loss value.

    Notes
    -----
    - Inputs are aligned using their index.
    - Potential division by zero or log of zero/negative in `predicted_variances`
      is handled by adding a small epsilon (1e-8) to `predicted_variances`
      before division and log operations.
    - Reference: Page 178 of "Elements of Quantitative Investing".
    """
    # Align inputs
    aligned_realized, aligned_predicted = realized_variances.align(predicted_variances, join='inner')

    # Handle potential invalid values in predicted_variances
    aligned_predicted = aligned_predicted + EPSILON

    # Calculate variance ratio
    variance_ratio = aligned_realized / aligned_predicted

    # Calculate QLIKE loss
    qlike_loss = np.mean(variance_ratio - np.log(variance_ratio) - 1)

    return float(qlike_loss)

def calculate_mse_variance_ratio_loss(realized_variances: pd.Series, predicted_variances: pd.Series) -> float:
    """
    Calculate the Mean Squared Error (MSE) of variance ratios.

    The MSE of variance ratios is a loss function for evaluating variance forecasts.
    It is defined as: (1/T) Σ (r_t²/σ̂_t² - 1)².

    Parameters
    ----------
    realized_variances : pd.Series
        Series of realized variances (r_t²).
    predicted_variances : pd.Series
        Series of predicted variances (σ̂_t²).

    Returns
    -------
    float
        The MSE of variance ratios loss value.

    Notes
    -----
    - Inputs are aligned using their index.
    - Potential division by zero in `predicted_variances` is handled by adding
      a small epsilon (1e-8) to `predicted_variances` before division.
    - Reference: Page 178 of "Elements of Quantitative Investing".
    """
    # Align inputs
    aligned_realized, aligned_predicted = realized_variances.align(predicted_variances, join='inner')

    # Handle potential division by zero in predicted_variances
    aligned_predicted = aligned_predicted + EPSILON

    # Calculate variance ratio
    variance_ratio = aligned_realized / aligned_predicted

    # Calculate MSE of variance ratios loss
    mse_loss = np.mean((variance_ratio - 1)**2)

    return float(mse_loss)
