# examples/chapter_8_evaluating_excess_returns_example.py

import numpy as np
import pandas as pd
import yfinance as yf

# Assuming the library is installed or PYTHONPATH is set correctly
from quant_elements_lib.backtesting.framework import (
    generate_k_fold_cv_indices_ts,
    generate_walk_forward_indices,
)
from quant_elements_lib.backtesting.ras_evaluation import (
    calculate_empirical_rademacher_complexity,
    calculate_ras_lower_bound_ic,
)
# from quant_elements_lib.performance_metrics.metrics import calculate_information_coefficient # Assuming this exists from Ch3

def calculate_simple_momentum_signal(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Generates a simple momentum signal (return over past 'window' days)."""
    return prices.pct_change(window).shift(1) # Shift to avoid lookahead for IC calc

def calculate_daily_ics(signals: pd.DataFrame, future_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates daily Information Coefficients for multiple signals.
    Aligns signals and future_returns, then calculates IC row-wise.
    For IC_t = corr(signal_t, future_return_{t+1}), signal_t is known at t, future_return_{t+1} is return from t to t+1.
    If signals are already shifted (i.e., signal for day t is based on data up to t-1),
    then future_returns should be returns for day t (e.g. open_t to open_{t+1} or close_t to close_{t+1}).
    """
    common_index = signals.index.intersection(future_returns.index)
    signals_aligned = signals.loc[common_index]
    future_returns_aligned = future_returns.loc[common_index]

    daily_ic_values = {}
    for col in signals_aligned.columns: # Iterate through each signal
        # Calculate correlation between each signal column and each corresponding future_return column
        # This is a simplification. True IC is often asset-by-asset cross-sectional correlation.
        # Here, we do a time-series correlation of the signal values vs future returns,
        # which is more like a predictive R-squared if returns are regressed on signal.
        # For a more "textbook" IC (cross-sectional correlation of signal_t vs return_t+1):
        # We'd need signals and returns where columns are assets, rows are time.
        # This example uses signals where columns are different signal types for the same asset(s).
        # Let's assume signals and future_returns are for a single asset for simplicity here.
        # If multi-asset, this would need reshaping or cross-sectional logic.

        # Simplified: if signals are T x N_signals and future_returns is T x N_assets,
        # we'd need to decide how to match them. Assume for now they are matched 1-to-1 (e.g. signal for asset X, return for asset X)
        # Or, if performance_matrix_X for RAS is T x N_strategies, each column is the IC of *one* strategy over time.

        # Let's make performance_matrix_X where each column is a time series of ICs for one signal strategy.
        # This requires calculating IC at each time point t.
        # IC_t = correlation(signal_values_at_t_across_assets, future_returns_at_t_across_assets)
        # This example is simplified as we only have one asset for future_returns.
        # So, we make each "signal" pretend it's an IC time series.

        # For the purpose of this example, let's assume `signals` are already ICs or a similar performance metric.
        # If `signals` contains raw signal values and `future_returns` contains returns,
        # the IC calculation would be:
        # temp_df = pd.concat([signals_aligned[col], future_returns_aligned.iloc[:, 0]], axis=1).dropna()
        # if len(temp_df) > 1:
        #     daily_ic_values[col] = temp_df.iloc[:, 0].rolling(window=20).corr(temp_df.iloc[:, 1]).fillna(0) # rolling IC
        # else:
        #     daily_ic_values[col] = pd.Series(0.0, index=signals_aligned.index)

        # To keep example simple and focused on RAS, let's assume signals *are* the performance time series (e.g. daily ICs)
        # Normally, these would be calculated from a proper backtest.
        # For demonstration, we'll use the signal values themselves as proxies for daily ICs, scaled down.
        daily_ic_values[f"IC_{col}"] = signals_aligned[col] * 0.1 # Scale to make them look like ICs

    return pd.DataFrame(daily_ic_values, index=signals_aligned.index)


def main():
    print("Chapter 8: Evaluating Excess Returns Example")

    # 1. Data Fetching and Preparation
    tickers = ['AAPL', 'MSFT', 'GOOGL'] # Using a few tickers for a hypothetical multi-asset signal scenario
    start_date = "2010-01-01"
    end_date = "2023-01-01"

    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        if data.empty:
            print(f"No data downloaded for {tickers}. Exiting.")
            return
        if len(tickers) > 1 and isinstance(data.columns, pd.MultiIndex): # yfinance >=0.2.41
             # Flatten MultiIndex if necessary, or select one asset for simplicity
             # For this example, let's average prices to get a single series for simplicity
             # In a real scenario, signals would be generated per asset.
             prices = data.mean(axis=1).dropna().to_frame(name="AvgPrice")
        elif len(tickers) == 1:
            prices = data.to_frame(name=tickers[0])
        else: # Older yfinance or single ticker not in list
            prices = data.mean(axis=1).dropna().to_frame(name="AvgPrice")


    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    log_returns = np.log(prices / prices.shift(1)).dropna()
    # For IC calculation, future_returns are typically next-period returns
    future_returns = log_returns.shift(-1).dropna() # log_returns aligned with signal day 't'

    # Align prices for signal calculation with future_returns for IC calculation
    # signals based on prices up to t-1, IC uses return from t to t+1
    aligned_prices = prices.loc[future_returns.index]


    # 2. Simulate Signals
    signal_momentum_20d = calculate_simple_momentum_signal(aligned_prices, window=20)
    signal_momentum_60d = calculate_simple_momentum_signal(aligned_prices, window=60)

    # Create a DataFrame of signals (ensure they are on the same index)
    signals_df = pd.concat([
        signal_momentum_20d.rename(columns={"AvgPrice": "Mom20D"}),
        signal_momentum_60d.rename(columns={"AvgPrice": "Mom60D"})
    ], axis=1).dropna()

    # 3. Calculate Daily ICs (or use signals as performance metric proxies)
    # Here, we use the simplified approach where scaled signals act as ICs
    # In a real scenario, you'd compute actual cross-sectional ICs daily.
    # performance_matrix_X columns: strategy1_IC_t, strategy2_IC_t, ...

    # Align signals with future_returns for IC calculation
    common_idx = signals_df.index.intersection(future_returns.index)
    signals_df_aligned = signals_df.loc[common_idx]
    future_returns_aligned = future_returns.loc[common_idx]

    # This is a placeholder for actual IC calculation.
    # For a real IC, you'd correlate signal_t with return_t (if signal is for t, return is t to t+1)
    # For simplicity, let's assume `signals_df_aligned` columns are already daily ICs or similar performance metrics.
    # If these were raw signals, you would need a function here to convert them to daily ICs.
    # E.g., by correlating stock signals with stock returns cross-sectionally each day.
    # For this example, we'll just use the signal values as proxies for the performance matrix.
    performance_matrix_X = signals_df_aligned.copy()
    # Let's scale them to be more like ICs (e.g. between -0.1 and 0.1)
    performance_matrix_X = performance_matrix_X / performance_matrix_X.abs().max().max() * 0.1
    performance_matrix_X = performance_matrix_X.dropna()


    if performance_matrix_X.empty:
        print("Performance matrix is empty after processing signals. Exiting.")
        return

    num_observations = len(performance_matrix_X)
    print(f"\nNumber of observations (time periods): {num_observations}")
    print(f"Number of signals/strategies: {performance_matrix_X.shape[1]}")
    print("Performance Matrix X (first 5 rows):")
    print(performance_matrix_X.head())

    # 4. Demonstrate Backtesting Protocol Setups
    print("\n--- Backtesting Protocol Setups ---")
    if num_observations > 10 : # Need enough obs for meaningful splits
        try:
            k_folds = 5
            gap = 1
            print(f"\nGenerating {k_folds}-Fold Cross-Validation Indices (gap={gap}):")
            cv_indices = generate_k_fold_cv_indices_ts(num_observations, k_folds=k_folds, gap_between_folds=gap)
            for i, (train_idx, test_idx) in enumerate(cv_indices):
                print(f"Fold {i+1}: Train size={len(train_idx)}, Test size={len(test_idx)}")
                # print(f"  Train: {train_idx[:3]}...{train_idx[-3:]}, Test: {test_idx[:3]}...{test_idx[-3:]}")
        except ValueError as e:
            print(f"Could not generate K-Fold CV indices: {e}")


        try:
            train_window = num_observations // 3 # Approx 1/3 for training
            test_window = num_observations // 10 # Approx 1/10 for testing
            if train_window > 0 and test_window > 0:
                print(f"\nGenerating Walk-Forward Indices (fixed train_window={train_window}, test_window={test_window}):")
                wf_indices_fixed = generate_walk_forward_indices(num_observations, train_window_size=train_window, test_window_size=test_window, fixed_train_window=True)
                if wf_indices_fixed:
                    for i, (train_idx, test_idx) in enumerate(wf_indices_fixed[:3]): # Print first 3
                        print(f"Split {i+1}: Train size={len(train_idx)}, Test size={len(test_idx)}")
                else:
                    print("No splits generated for fixed walk-forward with current parameters.")
            else:
                print("Skipping walk-forward due to small num_observations relative to window sizes.")

        except ValueError as e:
            print(f"Could not generate Walk-Forward indices: {e}")
    else:
        print("Skipping CV/Walk-Forward examples due to insufficient observations.")


    # 5. Calculate Empirical Rademacher Complexity
    print("\n--- Rademacher Complexity and RAS Bounds ---")
    try:
        r_hat_T = calculate_empirical_rademacher_complexity(performance_matrix_X, num_rademacher_samples=500) # Reduced samples for speed
        print(f"\nEmpirical Rademacher Complexity (R_hat_T): {r_hat_T:.4f}")

        # 6. Calculate Empirical Average ICs and RAS Lower Bounds
        empirical_avg_ics = performance_matrix_X.mean()
        print("\nEmpirical Average ICs (theta_hat_S):")
        print(empirical_avg_ics)

        ras_bounds_ic = calculate_ras_lower_bound_ic(
            empirical_ic_series=empirical_avg_ics,
            rademacher_complexity_R_hat_T=r_hat_T,
            num_observations_T=num_observations,
            delta_confidence=0.05  # 95% confidence
        )
        print("\nRAS Lower Bounds for ICs (95% confidence):")
        print(ras_bounds_ic)

        # Example for SR (if performance_matrix_X contained SRs instead of ICs)
        # empirical_avg_srs = performance_matrix_X.mean() # Assuming these are non-annualized SRs
        # ras_bounds_sr = calculate_ras_lower_bound_sr(
        #     empirical_sr_series=empirical_avg_srs,
        #     rademacher_complexity_R_hat_T=r_hat_T,
        #     num_observations_T=num_observations,
        #     delta_confidence=0.05
        # )
        # print("\nRAS Lower Bounds for SRs (95% confidence) - Example:")
        # print(ras_bounds_sr)

    except ValueError as e:
        print(f"Error in RAS calculations: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during RAS calculations: {e}")


if __name__ == "__main__":
    main()
