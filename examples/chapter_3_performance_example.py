# examples/chapter_3_performance_example.py

import pandas as pd
import numpy as np
import yfinance as yf
from ele_quant.performance_metrics import (
    calculate_expected_return,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_information_ratio,
    calculate_sharpe_ratio_se,
    adjust_sharpe_ratio_for_autocorrelation,
)

def main():
    """
    Main function to demonstrate performance metrics calculations.
    """
    # --- 1. Configuration & Data Fetching ---
    print("Fetching data for AAPL and SPY...")
    tickers = ['AAPL', 'SPY']
    start_date = '2020-01-01'
    end_date = '2022-12-31'
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    if data.empty:
        print("Failed to download data. Please check your internet connection or ticker symbols.")
        return

    # yfinance now defaults to auto_adjust=True, so 'Adj Close' might not be present.
    # Using 'Close' as it will be adjusted.
    aapl_prices = data['Close']['AAPL'].dropna()
    spy_prices = data['Close']['SPY'].dropna()

    # Define risk-free rate
    annual_risk_free_rate = 0.01
    daily_risk_free_rate = (1 + annual_risk_free_rate)**(1/252) - 1
    trading_periods_per_year = 252

    print(f"Data fetched for {len(aapl_prices)} trading days for AAPL.")
    print(f"Data fetched for {len(spy_prices)} trading days for SPY.")
    print(f"Annual risk-free rate: {annual_risk_free_rate:.2%}")
    print(f"Daily risk-free rate: {daily_risk_free_rate:.6f}")
    print("-" * 50)

    # --- 2. Preprocessing: Calculate Returns ---
    print("Calculating returns...")
    aapl_returns = aapl_prices.pct_change().dropna()
    spy_returns = spy_prices.pct_change().dropna()

    # Align returns before calculating active returns (important if dates don't match perfectly)
    common_index = aapl_returns.index.intersection(spy_returns.index)
    aapl_returns_aligned = aapl_returns.loc[common_index]
    spy_returns_aligned = spy_returns.loc[common_index]

    active_returns = aapl_returns_aligned - spy_returns_aligned

    if aapl_returns.empty or spy_returns.empty:
        print("Not enough data to calculate returns after processing.")
        return

    print(f"Calculated {len(aapl_returns)} daily returns for AAPL.")
    print(f"Calculated {len(spy_returns)} daily returns for SPY.")
    print(f"Calculated {len(active_returns)} daily active returns for AAPL vs SPY.")
    print("-" * 50)

    # --- 3. Demonstrate Implemented Library Functions ---
    print("Calculating performance metrics for AAPL...")

    # Expected Return
    exp_ret_aapl_daily = calculate_expected_return(aapl_returns)
    exp_ret_aapl_annualized = (1 + exp_ret_aapl_daily)**trading_periods_per_year - 1
    print(f"\nExpected Daily Return (AAPL): {exp_ret_aapl_daily:.6f}")
    print(f"Expected Annualized Return (AAPL): {exp_ret_aapl_annualized:.2%}")

    # Volatility
    vol_aapl_annualized = calculate_volatility(
        aapl_returns,
        trading_periods_per_year=trading_periods_per_year,
        annualize=True
    )
    vol_aapl_daily = calculate_volatility(aapl_returns, annualize=False)
    print(f"\nAnnualized Volatility (AAPL): {vol_aapl_annualized:.2%}")
    print(f"Daily Volatility (AAPL): {vol_aapl_daily:.6f}")

    # Sharpe Ratio
    sr_aapl_annualized = calculate_sharpe_ratio(
        aapl_returns,
        risk_free_rate=daily_risk_free_rate,
        trading_periods_per_year=trading_periods_per_year,
        annualize=True
    )
    print(f"\nAnnualized Sharpe Ratio (AAPL): {sr_aapl_annualized:.4f}")

    # Information Ratio
    ir_aapl_annualized = calculate_information_ratio(
        active_returns,
        trading_periods_per_year=trading_periods_per_year,
        annualize=True
    )
    print(f"\nAnnualized Information Ratio (AAPL vs SPY): {ir_aapl_annualized:.4f}")

    # Sharpe Ratio Standard Error
    # First, calculate non-annualized SR for AAPL
    sr_aapl_non_annualized_for_se = calculate_sharpe_ratio(
        aapl_returns,
        risk_free_rate=daily_risk_free_rate,
        annualize=False # Crucial: SE calculation needs non-annualized SR
    )
    num_observations = len(aapl_returns)
    se_sr_aapl_annualized = calculate_sharpe_ratio_se(
        sr_aapl_non_annualized_for_se,
        num_observations=num_observations,
        trading_periods_per_year=trading_periods_per_year,
        annualize_se=True # Annualize the SE itself
    )
    print(f"\nNon-Annualized Sharpe Ratio (for SE calc): {sr_aapl_non_annualized_for_se:.4f}")
    print(f"Number of Observations for SE: {num_observations}")
    print(f"Annualized Standard Error of Sharpe Ratio (AAPL): {se_sr_aapl_annualized:.4f}")

    # Confidence Interval for Sharpe Ratio (example)
    if not np.isnan(sr_aapl_annualized) and not np.isnan(se_sr_aapl_annualized):
        sr_lower_bound = sr_aapl_annualized - 1.96 * se_sr_aapl_annualized
        sr_upper_bound = sr_aapl_annualized + 1.96 * se_sr_aapl_annualized
        print(f"Approx. 95% Confidence Interval for Annualized SR (AAPL): [{sr_lower_bound:.4f}, {sr_upper_bound:.4f}]")


    # Adjusted Sharpe Ratio for Autocorrelation
    autocorr_aapl_lag1 = aapl_returns.autocorr(lag=1)
    # We adjust the annualized SR using the autocorrelation
    sr_aapl_adj_annualized = adjust_sharpe_ratio_for_autocorrelation(
        sr_aapl_annualized, # Use the annualized SR here
        lag_one_autocorrelation=autocorr_aapl_lag1
    )
    print(f"\nLag-1 Autocorrelation of AAPL Returns: {autocorr_aapl_lag1:.4f}")
    print(f"Annualized Sharpe Ratio (AAPL) Adjusted for Autocorrelation: {sr_aapl_adj_annualized:.4f}")

    print("-" * 50)
    print("Example script finished.")

if __name__ == "__main__":
    main()
