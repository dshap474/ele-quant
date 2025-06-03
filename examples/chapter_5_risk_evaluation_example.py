"""
Example script for Chapter 5: Risk Evaluation.

This script demonstrates the usage of various risk evaluation functions
from the quant_elements_lib.risk_evaluation submodule.
It covers:
- Fetching stock price data using yfinance.
- Calculating log returns.
- Estimating covariance matrices (simple sample and basic shrinkage).
- Applying loss functions (QLIKE, MSE of variance ratios).
- Evaluating covariance matrices using random portfolios (evaluate_random_portfolio_variance).
- Calculating QDIST likelihood (calculate_qdist_likelihood).
- Testing minimum variance portfolios (test_minimum_variance_portfolios).
- Calculating MALV statistic (calculate_malv_statistic).
"""

import yfinance as yf
import pandas as pd
import numpy as np

from quant_elements_lib.risk_evaluation import (
    calculate_qlike_loss,
    calculate_mse_variance_ratio_loss,
    evaluate_random_portfolio_variance,
    calculate_qdist_likelihood,
    test_minimum_variance_portfolios,
    calculate_malv_statistic
)

# Attempt to import a specific covariance estimator, e.g., Ledoit-Wolf
# from quant_elements_lib.utils.covariance_estimation.
# If not available, a simple sample covariance and basic shrinkage will be used.
try:
    # Assuming a hypothetical LedoitWolfCovarianceEstimator class/function might exist
    # This path is based on the PLAN.md for Ch 6.
    from quant_elements_lib.utils.covariance_estimation import LedoitWolfCovarianceEstimator
    # A more generic import if the exact name is unknown:
    # from quant_elements_lib.utils import covariance_estimation as ce
    # And then check for attributes like ce.LedoitWolfCovarianceEstimator
    LW_AVAILABLE = True
    print("Ledoit-Wolf covariance estimator found and will be used for Omega_B.\n")
except ImportError:
    LW_AVAILABLE = False
    print("Ledoit-Wolf covariance estimator not found in quant_elements_lib.utils.covariance_estimation.")
    print("Using simple sample covariance for Omega_A and basic shrinkage for Omega_B.\n")

# Small epsilon for ensuring positive definiteness and numerical stability
EPSILON = 1e-8

def ensure_positive_definite(matrix: pd.DataFrame) -> pd.DataFrame:
    """Ensures a covariance matrix is positive definite by adding epsilon to diagonal."""
    if not np.all(np.linalg.eigvals(matrix.values) > 0):
        print(f"Matrix not positive definite. Adding epsilon ({EPSILON}) to diagonal.")
        matrix_pd = matrix + np.eye(matrix.shape[0]) * EPSILON
        # Re-check (optional, for verbosity)
        if not np.all(np.linalg.eigvals(matrix_pd.values) > 0):
             print("Warning: Matrix still not positive definite after adding epsilon.")
        return matrix_pd
    return matrix

def main():
    """Main function to run the risk evaluation examples."""

    # 1. Data Fetching and Preparation
    print("="*50)
    print("1. Data Fetching and Preparation")
    print("="*50)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'V', 'JNJ', 'PG', 'NVDA', 'TSLA']
    # Using a shorter period for faster execution in an example setting
    start_date = '2020-01-01'
    end_date = '2023-12-31'

    print(f"Fetching daily closing prices for: {tickers}")
    print(f"Period: {start_date} to {end_date}\n")
    try:
        prices = yf.download(tickers, start=start_date, end=end_date)['Close']
    except Exception as e:
        print(f"Error downloading data from yfinance: {e}")
        print("Please ensure yfinance is installed and you have an internet connection.")
        # Create dummy data for offline testing if yfinance fails
        print("Using dummy data for demonstration as yfinance failed.")
        dates = pd.date_range(start_date, end_date, freq='B')
        prices = pd.DataFrame(
            np.random.rand(len(dates), len(tickers)) * 100 + 50,
            index=dates,
            columns=tickers
        )
        prices = prices.cumprod() # Make it look a bit like prices


    if prices.empty or prices.isnull().all().all():
        print("Failed to download price data or data is all NaNs. Exiting.")
        return

    # Handle cases where some tickers might not have data for the full period
    prices = prices.dropna(axis=1, how='any') # Drop columns with any NaNs
    if prices.empty:
        print("No tickers with complete data for the period. Exiting.")
        return

    print("Downloaded prices (showing head):")
    print(prices.head())

    log_returns = np.log(prices / prices.shift(1)).dropna()
    print("\nCalculated log returns (showing head):")
    print(log_returns.head())

    if log_returns.empty:
        print("Log returns are empty. Check input data. Exiting.")
        return

    # Split data: 70% estimation, 30% evaluation
    split_ratio = 0.7
    split_index = int(len(log_returns) * split_ratio)
    log_returns_estimation = log_returns.iloc[:split_index]
    log_returns_evaluation = log_returns.iloc[split_index:]

    print(f"\nData split into estimation ({len(log_returns_estimation)} days) and "
          f"evaluation ({len(log_returns_evaluation)} days) periods.")

    if log_returns_estimation.empty or log_returns_evaluation.empty:
        print("One of the data splits is empty. Need more data or different split. Exiting.")
        return

    # Update tickers list based on available data after potential drops
    available_tickers = log_returns.columns.tolist()
    print(f"\nUsing data for the following tickers: {available_tickers}")


    # 2. Covariance Matrix Estimation
    print("\n" + "="*50)
    print("2. Covariance Matrix Estimation (from estimation period)")
    print("="*50)

    # Omega_A: Simple Sample Covariance
    omega_a_df = log_returns_estimation.cov()
    print("\nOmega_A (Sample Covariance Matrix - head):")
    print(omega_a_df.head())
    omega_a_df = ensure_positive_definite(omega_a_df)


    # Omega_B: Alternative Covariance (Shrinkage or Ledoit-Wolf)
    if LW_AVAILABLE:
        try:
            # This assumes LedoitWolfCovarianceEstimator has a fit method
            # and returns a DataFrame or can be converted to one.
            # Adjust based on actual API of the estimator.
            # For example, if it's a scikit-learn style estimator:
            # lw_estimator = LedoitWolfCovarianceEstimator()
            # lw_estimator.fit(log_returns_estimation.values)
            # omega_b_values = lw_estimator.covariance_
            # omega_b_df = pd.DataFrame(omega_b_values, index=omega_a_df.index, columns=omega_a_df.columns)

            # Placeholder for actual LW usage - this will depend on the actual class
            # For now, let's assume it's a function that takes returns and returns a DataFrame
            # omega_b_df = LedoitWolfCovarianceEstimator(log_returns_estimation) # Fictional direct usage

            # If it's a class like sklearn.covariance.LedoitWolf
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(log_returns_estimation.values)
            omega_b_values = lw.covariance_
            omega_b_df = pd.DataFrame(omega_b_values, index=omega_a_df.index, columns=omega_a_df.columns)
            print("\nOmega_B (Ledoit-Wolf Shrinkage Covariance Matrix - head):")

        except Exception as e:
            print(f"Error using LedoitWolfCovarianceEstimator: {e}")
            print("Falling back to simple shrinkage for Omega_B.")
            rho = 0.1
            omega_b_df = (1 - rho) * omega_a_df + rho * pd.DataFrame(np.diag(np.diag(omega_a_df.values)), index=omega_a_df.index, columns=omega_a_df.columns)
            print("\nOmega_B (Simple Shrinkage Covariance Matrix - head):")
    else:
        rho = 0.1 # Simple shrinkage factor
        # Ensure diagonal part is also a DataFrame to maintain labels
        diag_omega_a = pd.DataFrame(np.diag(np.diag(omega_a_df.values)), index=omega_a_df.index, columns=omega_a_df.columns)
        omega_b_df = (1 - rho) * omega_a_df + rho * diag_omega_a
        print("\nOmega_B (Simple Shrinkage Covariance Matrix - head):")

    print(omega_b_df.head())
    omega_b_df = ensure_positive_definite(omega_b_df)


    # 3. Demonstrate Loss Functions on Variances
    print("\n" + "="*50)
    print("3. Demonstrate Loss Functions on Variances (using evaluation period)")
    print("="*50)

    variances_a = pd.Series(np.diag(omega_a_df.values), index=omega_a_df.index)
    variances_b = pd.Series(np.diag(omega_b_df.values), index=omega_b_df.index)

    # Calculate sample variances from the "evaluation period"
    # Ensure var is calculated per asset (column-wise)
    realized_variances_eval = log_returns_evaluation.var(axis=0)

    # Align (should already be aligned by ticker names if .var() preserves them)
    variances_a_aligned, realized_variances_eval_a = variances_a.align(realized_variances_eval, join='inner')
    variances_b_aligned, realized_variances_eval_b = variances_b.align(realized_variances_eval, join='inner')

    if variances_a_aligned.empty or variances_b_aligned.empty:
        print("Error: Alignment resulted in empty series for variance loss functions. Check ticker consistency.")
    else:
        qlike_loss_a = calculate_qlike_loss(realized_variances_eval_a, variances_a_aligned)
        qlike_loss_b = calculate_qlike_loss(realized_variances_eval_b, variances_b_aligned)
        print(f"\nQLIKE Loss for Omega_A variances: {qlike_loss_a:.6f}")
        print(f"QLIKE Loss for Omega_B variances: {qlike_loss_b:.6f}")

        mse_loss_a = calculate_mse_variance_ratio_loss(realized_variances_eval_a, variances_a_aligned)
        mse_loss_b = calculate_mse_variance_ratio_loss(realized_variances_eval_b, variances_b_aligned)
        print(f"\nMSE Variance Ratio Loss for Omega_A variances: {mse_loss_a:.6f}")
        print(f"MSE Variance Ratio Loss for Omega_B variances: {mse_loss_b:.6f}")


    # 4. Demonstrate evaluate_random_portfolio_variance
    print("\n" + "="*50)
    print("4. Demonstrate evaluate_random_portfolio_variance (using QLIKE loss)")
    print("="*50)
    # Use a smaller number of portfolios for faster example execution
    num_portfolios_example = 100

    avg_loss_rand_a = evaluate_random_portfolio_variance(
        omega_a_df, log_returns_evaluation, calculate_qlike_loss, num_random_portfolios=num_portfolios_example
    )
    avg_loss_rand_b = evaluate_random_portfolio_variance(
        omega_b_df, log_returns_evaluation, calculate_qlike_loss, num_random_portfolios=num_portfolios_example
    )
    print(f"\nAverage QLIKE Loss (Random Portfolios) for Omega_A: {avg_loss_rand_a:.6f}")
    print(f"Average QLIKE Loss (Random Portfolios) for Omega_B: {avg_loss_rand_b:.6f}")


    # 5. Demonstrate calculate_qdist_likelihood
    print("\n" + "="*50)
    print("5. Demonstrate calculate_qdist_likelihood (using evaluation period returns)")
    print("="*50)

    qdist_a = calculate_qdist_likelihood(omega_a_df, log_returns_evaluation)
    qdist_b = calculate_qdist_likelihood(omega_b_df, log_returns_evaluation)
    print(f"\nQDIST Likelihood for Omega_A: {qdist_a:.4f}")
    print(f"QDIST Likelihood for Omega_B: {qdist_b:.4f}")
    print("(Lower QDIST value suggests a better fit, up to a constant)")


    # 6. Demonstrate test_minimum_variance_portfolios
    print("\n" + "="*50)
    print("6. Demonstrate test_minimum_variance_portfolios")
    print("="*50)

    # Select a few arbitrary days from the evaluation period
    sample_days_indices = np.linspace(0, len(log_returns_evaluation) - 1, 5, dtype=int)
    sample_days_returns = log_returns_evaluation.iloc[sample_days_indices]

    print("\nTesting MVP for a few sample days from evaluation period:")
    for i, (day_idx, r_t) in enumerate(sample_days_returns.iterrows()):
        print(f"\n--- Sample Day (Index from eval period: {day_idx}, Date: {r_t.name.date()}) ---")
        # r_t is already a pd.Series with asset names as index

        real_var_a, w_mvp_a = test_minimum_variance_portfolios(omega_a_df, r_t)
        real_var_b, w_mvp_b = test_minimum_variance_portfolios(omega_b_df, r_t)

        print(f"Omega_A MVP: Realized Variance = {real_var_a:.8f}")
        print(f"Omega_B MVP: Realized Variance = {real_var_b:.8f}")
        if i == 0: # Print weights only for the first sample day for brevity
            print("\nMVP Weights for Omega_A (first sample day):")
            print(w_mvp_a.round(4))
            print("\nMVP Weights for Omega_B (first sample day):")
            print(w_mvp_b.round(4))


    # 7. Demonstrate calculate_malv_statistic
    print("\n" + "="*50)
    print("7. Demonstrate calculate_malv_statistic (using evaluation period returns)")
    print("="*50)

    try:
        precision_a = pd.DataFrame(np.linalg.pinv(omega_a_df.values + EPSILON * np.eye(omega_a_df.shape[0])),
                                   columns=omega_a_df.columns, index=omega_a_df.index)
        precision_b = pd.DataFrame(np.linalg.pinv(omega_b_df.values + EPSILON * np.eye(omega_b_df.shape[0])),
                                   columns=omega_b_df.columns, index=omega_b_df.index)

        malv_a = calculate_malv_statistic(precision_a, log_returns_evaluation)
        malv_b = calculate_malv_statistic(precision_b, log_returns_evaluation)
        print(f"\nMALV Statistic for Omega_A (using its precision matrix): {malv_a:.6f}")
        print(f"MALV Statistic for Omega_B (using its precision matrix): {malv_b:.6f}")
        print("(MALV is (1/T) Sum r_t' Omega^-1 r_t. Lower values might indicate better fit under certain assumptions.)")

    except np.linalg.LinAlgError:
        print("\nError calculating pseudo-inverse for MALV statistic. Skipping.")

    print("\n" + "="*50)
    print("Example script finished.")
    print("="*50)


if __name__ == "__main__":
    # Check if yfinance is installed
    try:
        import yfinance
    except ImportError:
        print("Error: yfinance is not installed. Please install it to run this example:")
        print("pip install yfinance")
        # Fallback to dummy data generation path if yfinance is not installed
        # This is handled inside main() now by checking yf.download() output
        # but good to have a top-level check too.

    # Check if scikit-learn is installed (for LedoitWolf as a fallback if our internal one isn't there)
    try:
        import sklearn
    except ImportError:
        if LW_AVAILABLE: # If our internal LW was supposed to be available but failed
            pass # No sklearn needed
        else: # If we intended to use sklearn's LW as a fallback
             print("\nNote: scikit-learn is not installed. If a Ledoit-Wolf estimator from")
             print("quant_elements_lib.utils.covariance_estimation is not found,")
             print("this script uses simple shrinkage. Installing scikit-learn would enable")
             print("using sklearn.covariance.LedoitWolf as an alternative for Omega_B.")
             print("pip install scikit-learn\n")


    main()
