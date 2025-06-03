import pandas as pd
import numpy as np
from typing import Union

def winsorize_series(
    input_series: pd.Series,
    method: str = 'quantile',
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    z_threshold: float = 3.0
) -> pd.Series:
    """
    Winsorizes a pandas Series using either quantile-based or robust z-score method.

    Args:
        input_series: The pandas Series to winsorize.
        method: The winsorization method. Options:
                'quantile': Caps values based on specified quantiles.
                'z_score_robust': Caps values based on robust z-scores (using median and MAD).
        lower_quantile: The lower quantile for the 'quantile' method (e.g., 0.01 for 1st percentile).
        upper_quantile: The upper quantile for the 'quantile' method (e.g., 0.99 for 99th percentile).
        z_threshold: The z-score threshold for the 'z_score_robust' method.
                     Values with absolute robust z-score greater than this will be capped.

    Returns:
        A new pandas Series with winsorized values.

    Raises:
        ValueError: If an invalid method is specified or quantile values are out of [0,1] range.
    """
    if not isinstance(input_series, pd.Series):
        raise TypeError("input_series must be a pandas Series.")

    series_copy = input_series.copy()

    # Drop NaNs for calculations, but preserve their original positions for the output
    # Alternatively, decide if NaNs should be filled or handled by quantile/median functions
    # For now, calculations are on non-NaN data, and NaNs remain NaNs in the output.
    # If input_series contains NaNs, series_copy will also. Operations below should handle NaNs appropriately.
    # For example, .quantile() ignores NaNs by default. Median also ignores NaNs.

    if method == 'quantile':
        if not (0 <= lower_quantile <= 1 and 0 <= upper_quantile <= 1):
            raise ValueError("Quantiles must be between 0 and 1.")
        if lower_quantile >= upper_quantile:
            raise ValueError("Lower quantile must be less than upper quantile.")

        lower_val = series_copy.quantile(lower_quantile)
        upper_val = series_copy.quantile(upper_quantile)

        # Check if lower_val or upper_val are NaN (e.g. if series is all NaN or empty after dropna)
        if pd.isna(lower_val) or pd.isna(upper_val):
            # If series was all NaNs, or empty, quantiles are NaN. Return copy.
            return series_copy

        series_copy[series_copy < lower_val] = lower_val
        series_copy[series_copy > upper_val] = upper_val

    elif method == 'z_score_robust':
        if z_threshold <= 0:
            raise ValueError("z_threshold must be positive.")

        median_val = series_copy.median()

        if pd.isna(median_val): # Handles all-NaN series
            return series_copy

        abs_dev = abs(series_copy - median_val)
        mad = abs_dev.median() # Median Absolute Deviation

        # If MAD is zero (e.g., all non-NaN values are the same), no scaling is possible/meaningful.
        if np.isclose(mad, 0):
            return series_copy # Return original copy as no meaningful winsorization can be done

        scaled_mad = mad * 1.4826 # Scale factor to make MAD comparable to std dev for normal data

        if np.isclose(scaled_mad, 0): # Should be caught by mad check, but as a safeguard
             return series_copy

        robust_z_scores = (series_copy - median_val) / scaled_mad

        # Identify outliers
        lower_cap = median_val - z_threshold * scaled_mad
        upper_cap = median_val + z_threshold * scaled_mad

        series_copy[robust_z_scores < -z_threshold] = lower_cap
        series_copy[robust_z_scores > z_threshold] = upper_cap

    else:
        raise ValueError(f"Invalid method: '{method}'. Choose 'quantile' or 'z_score_robust'.")

    return series_copy
