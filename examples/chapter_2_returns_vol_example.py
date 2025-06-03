import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Import from ele_quant library
from ele_quant.returns_analysis.calculations import (
    calculate_simple_returns,
    calculate_log_returns,
    calculate_dividend_adjusted_returns,
    calculate_excess_returns,
    calculate_compounded_return_series,
    calculate_total_compounded_return,
)
from ele_quant.returns_analysis.price_estimation import (
    estimate_true_price_roll_model,
)
from ele_quant.volatility_models.garch import (
    garch_1_1_conditional_variance,
    garch_1_1_log_likelihood, # Not directly used in example, but good for completeness
    estimate_garch_1_1_parameters,
)
from ele_quant.volatility_models.realized_volatility import (
    calculate_realized_variance_from_sum_sq_returns,
    estimate_variance_mle_diffusion,
)
from ele_quant.volatility_models.state_space_vol import (
    ewma_variance_forecast,
    muth_model_variance_estimation,
    harvey_shephard_volatility_estimation,
)

def run_chapter_2_examples():
    print("--- Chapter 2 Examples: Returns and Volatility ---")

    # --- 1. Data Fetching ---
    print("\n--- 1. Data Fetching ---")
    daily_tickers = ['AAPL', 'MSFT', 'SPY']
    start_date = '2020-01-01'
    end_date = '2022-12-31'
    spy_intraday_data = None
    aapl_daily_data = None
    spy_daily_data = None

    try:
        print(f"Fetching daily data for {', '.join(daily_tickers)} from {start_date} to {end_date}...")
        all_daily_data = yf.download(daily_tickers, start=start_date, end=end_date, progress=False)
        if all_daily_data.empty:
            print("Failed to download daily data. Examples relying on it will be skipped.")
        else:
            aapl_daily_data = all_daily_data['Close']['AAPL'].dropna()
            aapl_dividends = yf.Ticker('AAPL').dividends.loc[start_date:end_date] # Fetch dividends
            spy_daily_data = all_daily_data['Close']['SPY'].dropna()
            print("Daily data fetched successfully.")
    except Exception as e:
        print(f"Error fetching daily data: {e}")

    try:
        print("\nFetching 5-minute intraday data for SPY (last 60 days)...")
        # yfinance intraday data is limited to last 60 days for 1m, 5m might be similar
        spy_intraday_data_raw = yf.download('SPY', period="60d", interval="5m", progress=False)
        if spy_intraday_data_raw.empty or 'Close' not in spy_intraday_data_raw:
            print("Failed to download SPY 5-minute intraday data or data is incomplete. RV examples might be skipped.")
        else:
            spy_intraday_data = spy_intraday_data_raw['Close'].dropna()
            print(f"SPY 5-minute intraday data fetched successfully. Shape: {spy_intraday_data.shape}")
    except Exception as e:
        print(f"Error fetching SPY 5-minute intraday data: {e}. RV examples might be skipped.")

    # --- 2. Preprocessing ---
    print("\n--- 2. Preprocessing ---")
    if aapl_daily_data is not None and not aapl_daily_data.empty:
        print("Preprocessing AAPL daily data...")
        aapl_simple_returns = calculate_simple_returns(aapl_daily_data)
        aapl_log_returns = calculate_log_returns(aapl_daily_data) # from prices
        # Alternative: aapl_log_returns_from_simple = calculate_log_returns(aapl_simple_returns)
        aapl_sq_log_returns = aapl_log_returns**2
        print("AAPL daily preprocessing complete.")
    else:
        print("AAPL daily data not available, skipping related preprocessing.")
        aapl_simple_returns, aapl_log_returns, aapl_sq_log_returns = None, None, None

    spy_daily_sum_sq_log_returns = None
    spy_daily_n_obs = None
    sample_day_spy_intraday_returns = None

    if spy_intraday_data is not None and not spy_intraday_data.empty:
        print("\nPreprocessing SPY 5-minute intraday data...")
        spy_intraday_log_returns = calculate_log_returns(spy_intraday_data) # from prices
        spy_intraday_log_returns = spy_intraday_log_returns.dropna()

        if not spy_intraday_log_returns.empty:
            spy_daily_sum_sq_log_returns = spy_intraday_log_returns.groupby(spy_intraday_log_returns.index.date).apply(lambda x: (x**2).sum())
            spy_daily_n_obs = spy_intraday_log_returns.groupby(spy_intraday_log_returns.index.date).count()

            # Get intraday returns for a sample day (first available day)
            if not spy_daily_sum_sq_log_returns.empty:
                sample_date = spy_daily_sum_sq_log_returns.index[0]
                sample_day_spy_intraday_returns = spy_intraday_log_returns[spy_intraday_log_returns.index.date == sample_date]
                print(f"SPY intraday preprocessing complete. Sample day for MLE: {sample_date}")
            else:
                print("No full days of intraday data after processing.")
        else:
            print("SPY intraday log returns are empty after calculation and NaN drop.")
    else:
        print("SPY 5-minute intraday data not available, skipping related preprocessing.")


    # --- 3. Demonstrate Implemented Library Functions ---
    print("\n--- 3. Demonstrate Library Functions ---")

    # ** Return Calculations (AAPL daily) **
    if aapl_simple_returns is not None:
        print("\n-- Return Calculations (AAPL Daily) --")
        print("Sample Simple Returns (AAPL):\n", aapl_simple_returns.head())
        print("\nSample Log Returns (AAPL):\n", aapl_log_returns.head())

        # Dividend Adjusted Returns
        if aapl_dividends is not None and not aapl_dividends.empty:
            # Align dividends with prices (forward fill, or sum if multiple on same day)
            # yf dividends are timestamped to payment date. Adjustment needs care.
            # For simplicity, let's try a basic alignment.
            aligned_dividends = aapl_dividends.reindex(aapl_daily_data.index).fillna(0)
            if aligned_dividends.sum() > 0 :
                aapl_div_adj_returns = calculate_dividend_adjusted_returns(aapl_daily_data, aligned_dividends)
                print("\nSample Dividend Adjusted Returns (AAPL):\n", aapl_div_adj_returns.head())
            else:
                print("\nNote: No dividends for AAPL in the fetched period or alignment failed. Skipping dividend-adjusted returns demo.")
        else:
            print("\nNote: Dividend data for AAPL not available/fetched. Skipping dividend-adjusted returns demo.")

        # Excess Returns
        annual_risk_free_rate = 0.01
        daily_risk_free_rate = (1 + annual_risk_free_rate)**(1/252) - 1 # Assuming 252 trading days
        aapl_excess_returns = calculate_excess_returns(aapl_simple_returns, daily_risk_free_rate)
        print(f"\nSample Excess Returns (AAPL, daily RFR: {daily_risk_free_rate:.6f}):\n", aapl_excess_returns.head())

        # Compounded Returns
        aapl_comp_ret_series = calculate_compounded_return_series(aapl_simple_returns.dropna())
        aapl_total_comp_ret = calculate_total_compounded_return(aapl_simple_returns.dropna())
        print("\nCompounded Return Series (AAPL, start of series):\n", aapl_comp_ret_series.head())
        print(f"\nTotal Compounded Return (AAPL): {aapl_total_comp_ret:.4f}")

    # ** Roll Model (AAPL daily prices) **
    if aapl_daily_data is not None:
        print("\n-- Roll Model (AAPL Daily Prices) --")
        sigma_m_sq_roll = 0.01**2  # Example: 1% daily true price std dev
        sigma_eta_sq_roll = 0.005**2 # Example: 0.5% daily bid-ask bounce std dev

        # Ensure there are enough prices for the model
        if len(aapl_daily_data) > 1:
            aapl_roll_true_prices = estimate_true_price_roll_model(
                aapl_daily_data, sigma_m_sq_roll, sigma_eta_sq_roll
            )
            plt.figure(figsize=(10, 6))
            plt.plot(aapl_daily_data.index, aapl_daily_data, label='Observed Prices (AAPL)')
            plt.plot(aapl_roll_true_prices.index, aapl_roll_true_prices, label='Roll Model Estimated True Prices', linestyle='--')
            plt.title('AAPL: Observed vs. Roll Model Estimated True Prices')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
        else:
            print("Not enough AAPL daily data points to run Roll Model example.")


    # ** GARCH(1,1) (AAPL daily log returns) **
    if aapl_log_returns is not None and not aapl_log_returns.dropna().empty:
        print("\n-- GARCH(1,1) (AAPL Daily Log Returns) --")
        garch_params = estimate_garch_1_1_parameters(aapl_log_returns.dropna())
        print("Estimated GARCH(1,1) Parameters (AAPL):")
        for k, v in garch_params.items(): print(f"  {k}: {v:.6f}")

        if not np.isnan(garch_params['alpha_0']):
            garch_cond_var = garch_1_1_conditional_variance(
                aapl_log_returns.dropna(),
                garch_params['alpha_0'],
                garch_params['alpha_1'],
                garch_params['beta_1']
            )
            garch_cond_vol = np.sqrt(garch_cond_var)

            plt.figure(figsize=(10, 6))
            plt.plot(aapl_log_returns.index, aapl_log_returns, label='AAPL Log Returns', alpha=0.7)
            plt.plot(garch_cond_vol.index, garch_cond_vol, label='GARCH(1,1) Conditional Volatility', color='red')
            plt.title('AAPL: Log Returns and GARCH(1,1) Conditional Volatility')
            plt.xlabel('Date')
            plt.ylabel('Return / Volatility')
            plt.legend()
        else:
            print("GARCH parameter estimation failed for AAPL. Skipping GARCH plot.")
    else:
        print("AAPL log returns not available. Skipping GARCH example.")


    # ** Realized Volatility (SPY intraday) **
    if spy_daily_sum_sq_log_returns is not None and not spy_daily_sum_sq_log_returns.empty:
        print("\n-- Realized Volatility (SPY Intraday) --")
        # Using calculate_realized_variance_from_sum_sq_returns
        # Example: Daily RV (T=1 day). n is number of intraday returns.
        # The formula in book is (n/T) * sum_sq_ret. If T=1, it's n * sum_sq_ret.
        # This gives variance for "1 day" scaled by n.
        # A more common definition of daily RV is just sum_sq_ret.
        # Let's use T=1 as per prompt's interpretation of p.78.

        # Take the first available day for detailed example
        day_idx = spy_daily_sum_sq_log_returns.index[0]
        sum_sq_ret_day0 = spy_daily_sum_sq_log_returns.iloc[0]
        n_obs_day0 = spy_daily_n_obs.iloc[0]

        # Using textbook formula as implemented: (n/T) * sum_sq_ret
        # With T=1 (for daily variance from intraday returns), result is n * sum_sq_ret
        rv_day0_book = calculate_realized_variance_from_sum_sq_returns(sum_sq_ret_day0, n_obs_day0, period_length_T=1.0)
        # More common definition: sum_sq_ret
        rv_day0_common = sum_sq_ret_day0

        print(f"Realized Variance (SPY, {day_idx}, Book's (n/T) formula, T=1): {rv_day0_book:.6f} (vol: {np.sqrt(rv_day0_book):.4f})")
        print(f"Realized Variance (SPY, {day_idx}, Common sum_sq_ret): {rv_day0_common:.6f} (vol: {np.sqrt(rv_day0_common):.4f})")

        # Using estimate_variance_mle_diffusion
        if sample_day_spy_intraday_returns is not None and not sample_day_spy_intraday_returns.empty:
            mle_var_day0 = estimate_variance_mle_diffusion(sample_day_spy_intraday_returns, period_length_T=1.0)
            print(f"MLE Variance Estimate (SPY, {day_idx}, T=1): {mle_var_day0:.6f} (vol: {np.sqrt(mle_var_day0):.4f})")
        else:
            print("No sample day intraday returns for MLE demo.")

        # Plot series of daily realized volatility (using common definition: sum_sq_returns)
        spy_daily_realized_vol = np.sqrt(spy_daily_sum_sq_log_returns)
        plt.figure(figsize=(10,6))
        plt.plot(spy_daily_realized_vol.index, spy_daily_realized_vol, label='SPY Daily Realized Volatility (sqrt of sum of sq 5-min returns)')
        if spy_daily_data is not None: # Overlay SPY daily log returns if available
             spy_log_returns_for_rv_period = calculate_log_returns(spy_daily_data).reindex(spy_daily_realized_vol.index)
             plt.plot(spy_log_returns_for_rv_period.index, spy_log_returns_for_rv_period, label='SPY Daily Log Returns', alpha=0.5)
        plt.title('SPY: Daily Realized Volatility and Log Returns')
        plt.xlabel('Date')
        plt.ylabel('Volatility / Return')
        plt.legend()
    else:
        print("SPY intraday data not processed for Realized Volatility examples.")

    # ** EWMA (AAPL daily squared log returns) **
    if aapl_sq_log_returns is not None and not aapl_sq_log_returns.dropna().empty:
        print("\n-- EWMA (AAPL Daily Squared Log Returns) --")
        ewma_var = ewma_variance_forecast(aapl_sq_log_returns.dropna(), smoothing_factor_K=0.94)
        ewma_vol = np.sqrt(ewma_var)

        plt.figure(figsize=(10, 6))
        # Sqrt of squared log returns is abs(log_returns) - for visual comparison scale
        plt.plot(aapl_sq_log_returns.index, np.sqrt(aapl_sq_log_returns), label='AAPL Abs Log Returns', alpha=0.7)
        plt.plot(ewma_vol.index, ewma_vol, label='EWMA Volatility (K=0.94)', color='red')
        plt.title('AAPL: Abs Log Returns and EWMA Volatility')
        plt.xlabel('Date')
        plt.ylabel('Return / Volatility')
        plt.legend()
    else:
        print("AAPL squared log returns not available. Skipping EWMA example.")


    # ** State-Space Models (AAPL daily) **
    if aapl_log_returns is not None and not aapl_log_returns.dropna().empty:
        print("\n-- State-Space Models (AAPL Daily) --")
        # Muth Model (on squared log returns)
        tau_w_sq_muth = np.var(aapl_sq_log_returns.dropna().diff().dropna()) * 0.1 # Heuristic
        if pd.isna(tau_w_sq_muth) or tau_w_sq_muth <=1e-8 : tau_w_sq_muth = 1e-7
        tau_v_sq_muth = np.var(aapl_sq_log_returns.dropna()) * 0.5 # Heuristic
        if pd.isna(tau_v_sq_muth) or tau_v_sq_muth <=1e-8 : tau_v_sq_muth = 1e-6

        print(f"Muth model params: tau_w_sq={tau_w_sq_muth:.2e}, tau_v_sq={tau_v_sq_muth:.2e}")

        muth_est_true_var = muth_model_variance_estimation(
            aapl_sq_log_returns.dropna(), tau_w_sq_muth, tau_v_sq_muth
        )
        muth_est_true_vol = np.sqrt(muth_est_true_var)

        plt.figure(figsize=(10, 6))
        plt.plot(aapl_sq_log_returns.index, np.sqrt(aapl_sq_log_returns), label='AAPL Abs Log Returns', alpha=0.7)
        plt.plot(muth_est_true_vol.index, muth_est_true_vol, label='Muth Model Estimated True Volatility', color='purple')
        plt.title('AAPL: Abs Log Returns and Muth Model Estimated Volatility')
        plt.xlabel('Date')
        plt.ylabel('Return / Volatility')
        plt.legend()

        # Harvey-Shepherd Model (on log returns)
        # Parameters can be estimated using MLE, here we use plausible ones.
        # x_t = b + a*x_{t-1} + eps_t_state. x_t = log(h_t^2)
        # Typical a_state (persistence) is high, e.g., 0.9 to 0.99
        # b_state related to unconditional mean: b_state = log(omega_uncond) * (1-a_state)
        log_var_uncond = np.log(np.var(aapl_log_returns.dropna()))
        if pd.isna(log_var_uncond): log_var_uncond = np.log(1e-4)

        hs_a_state = 0.97
        hs_b_state = log_var_uncond * (1 - hs_a_state)
        hs_sigma_eps_sq = np.var(aapl_log_returns.dropna().diff().dropna()) * 0.1 # Heuristic for state noise
        if pd.isna(hs_sigma_eps_sq) or hs_sigma_eps_sq <= 1e-8: hs_sigma_eps_sq = 1e-7
        hs_beta_hs = -1.27 # E[log(chi_1_sq)] for standard normal innovations

        print(f"Harvey-Shephard params: b_state={hs_b_state:.4f}, a_state={hs_a_state:.4f}, sigma_eps_sq={hs_sigma_eps_sq:.2e}, beta_hs={hs_beta_hs:.2f}")

        harvey_s_vol = harvey_shephard_volatility_estimation(
            aapl_log_returns.dropna(),
            b_state=hs_b_state,
            a_state=hs_a_state,
            sigma_epsilon_sq_state=hs_sigma_eps_sq,
            beta_hs=hs_beta_hs
        )

        plt.figure(figsize=(10, 6))
        plt.plot(aapl_log_returns.index, aapl_log_returns.abs(), label='AAPL Abs Log Returns', alpha=0.7)
        plt.plot(harvey_s_vol.index, harvey_s_vol, label='Harvey-Shepherd Estimated Volatility', color='green')
        plt.title('AAPL: Abs Log Returns and Harvey-Shepherd Estimated Volatility')
        plt.xlabel('Date')
        plt.ylabel('Return / Volatility')
        plt.legend()
    else:
        print("AAPL log returns not available. Skipping State-Space model examples.")


    # --- 4. Show Plots ---
    print("\n--- 4. Displaying Plots ---")
    plt.tight_layout()
    plt.show()
    print("\n--- Chapter 2 Examples Complete ---")


if __name__ == '__main__':
    run_chapter_2_examples()
