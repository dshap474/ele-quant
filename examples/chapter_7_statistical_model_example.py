import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from quant_elements_lib.factor_models import StatisticalFactorModel, choose_num_factors_threshold

def run_chapter_7_example():
    """
    Demonstrates the use of StatisticalFactorModel (PCA method) from Chapter 7.
    """
    # 1. Data fetching and preparation
    # Using a smaller list for quicker example, ideally 30-50 for PCA
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "BRK-B", "JPM", "JNJ", "V",
        "PG", "UNH", "HD", "MA", "BAC", "DIS", "ADBE", "CRM", "NFLX", "PFE"
    ]
    # Reduce for CI/testing speed if necessary:
    # tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    print(f"Fetching data for {len(tickers)} tickers...")
    try:
        data = yf.download(tickers, start="2020-01-01", end="2022-12-31", progress=False, timeout=10)
        if data.empty or data['Adj Close'].isnull().all().all():
            print("Failed to download valid data with yfinance (empty or all NaNs).")
            data = None # Explicitly set to None to trigger fallback
        else:
            # Check for partial data (e.g. some tickers failed)
            adj_close_check = data['Adj Close'].dropna(axis=1, how='all')
            if adj_close_check.shape[1] < 2 : # Need at least 2 assets
                 print(f"Downloaded data for only {adj_close_check.shape[1]} assets. Insufficient.")
                 data = None
    except Exception as e:
        print(f"Exception during yfinance download: {e}")
        data = None # Trigger fallback

    if data is None:
        print("Using dummy data for demonstration as yfinance failed or returned insufficient data.")
        num_assets = len(tickers)
        num_periods = 252 * 3 # Approx 3 years
        np.random.seed(42)
        log_returns = pd.DataFrame(
            np.random.randn(num_periods, num_assets) * 0.01,
            index=pd.date_range(start="2020-01-01", periods=num_periods, freq='B'),
            columns=tickers
        )
        # Ensure tickers variable matches the columns of log_returns
        tickers = log_returns.columns.tolist()

    else:
        adj_close = data['Adj Close'].dropna(axis=1, how='all') # Drop columns that are all NaN
        adj_close = adj_close.dropna(axis=0) # Drop rows with any NaN from remaining assets

        valid_tickers = adj_close.columns.tolist()
        if len(valid_tickers) < 2: # Need at least 2 assets for PCA
             print(f"Not enough valid ticker data after cleaning (found {len(valid_tickers)}). Using dummy data.")
             num_assets = len(tickers) if len(tickers) >=2 else 2
             original_tickers = tickers # save original list for dummy data columns
             num_periods = 252 * 3
             np.random.seed(42)
             log_returns = pd.DataFrame(
                 np.random.randn(num_periods, num_assets) * 0.01,
                 index=pd.date_range(start="2020-01-01", periods=num_periods, freq='B'),
                 columns=original_tickers[:num_assets] # Use original tickers for consistency
             )
             tickers = log_returns.columns.tolist() # Update tickers to actual columns used
        else:
            log_returns = np.log(adj_close / adj_close.shift(1)).dropna()
            tickers = valid_tickers # Update tickers list to only include those with valid data

    print(f"Using data for {len(tickers)} tickers for PCA: {tickers}")
    print(f"Log returns shape: {log_returns.shape}")

    if log_returns.shape[0] < 5 or log_returns.shape[1] < 2: # Min T, Min N
        print("Not enough data points or assets for a meaningful PCA. Exiting example.")
        return

    # 2. Demean returns (StatisticalFactorModel will also do this, but good for direct calcs)
    returns_demeaned_for_scree = log_returns - log_returns.mean(axis=0)

    # 3. Determine number of factors
    # Using .values to ensure it's a numpy array for np.linalg.eigh
    sample_cov_matrix = returns_demeaned_for_scree.cov().values
    eigenvalues, _ = np.linalg.eigh(sample_cov_matrix)
    sorted_eigenvalues = np.sort(eigenvalues)[::-1] # Descending

    N_assets = log_returns.shape[1]
    T_observations = log_returns.shape[0]

    num_factors_chosen_threshold = 0
    if T_observations > 0: # Avoid division by zero for gamma
        num_factors_chosen_threshold = choose_num_factors_threshold(sorted_eigenvalues, N_assets, T_observations)

    # For stability in example, let's cap it or pick a fixed small number
    num_factors = min(max(1, num_factors_chosen_threshold), 10)

    # Ensure K < N for PCA/Statistical models
    if N_assets <= num_factors :
        num_factors = max(1, N_assets -1) # K must be less than N

    if num_factors == 0 and N_assets > 0: # Ensure at least one factor if possible
        num_factors = 1

    if N_assets == 0: # No assets, no factors.
        print("No assets to process after data loading/cleaning. Exiting.")
        return
    if num_factors == 0 and N_assets > 0: # If somehow num_factors became 0 with assets present
        print("Warning: num_factors is 0, but assets are present. Setting to 1.")
        num_factors = 1


    print(f"Eigenvalues from sample covariance (top 15): {sorted_eigenvalues[:min(15, len(sorted_eigenvalues))]}")
    print(f"Number of factors suggested by threshold rule: {num_factors_chosen_threshold}")
    print(f"Number of factors chosen for the model: {num_factors}")

    # 4. Instantiate and fit StatisticalFactorModel using PCA
    # asset_universe list must match columns of returns_data
    stat_model = StatisticalFactorModel(asset_universe=list(log_returns.columns), num_factors_to_extract=num_factors)

    print("\nFitting StatisticalFactorModel using PCA...")
    stat_model.fit(returns_data=log_returns, estimation_method='PCA')
    print("Model fitting complete.")

    # 5. Print key attributes of the fitted model
    print("\n--- Fitted Statistical Factor Model (PCA) ---")
    if stat_model.alpha is not None:
        print("\nAlpha (first 5 assets):")
        print(stat_model.alpha.head())
    else:
        print("\nAlpha: Not computed or None")

    if stat_model.B_loadings is not None:
        print("\nB_loadings (Factor Loadings - first 5 assets, first min(3, num_factors) factors):")
        print(stat_model.B_loadings.iloc[:5, :min(3, num_factors)])
    else:
        print("\nB_loadings: Not computed or None")

    if stat_model.factor_returns is not None:
        print("\nFactor Returns (first 5 periods, first min(3, num_factors) factors):")
        print(stat_model.factor_returns.iloc[:5, :min(3, num_factors)])
    else:
        print("\nFactor Returns: Not computed or None")

    if stat_model.factor_covariance is not None:
        print("\nFactor Covariance (first min(3, num_factors) x min(3, num_factors) factors):")
        print(stat_model.factor_covariance.iloc[:min(3, num_factors), :min(3, num_factors)])
    else:
        print("\nFactor Covariance: Not computed or None")

    if stat_model.idiosyncratic_covariance is not None:
        print("\nIdiosyncratic Variances (first 5 assets):")
        print(stat_model.idiosyncratic_covariance.head())
    else:
        print("\nIdiosyncratic Covariance: Not computed or None")

    # 6. Plot Scree Plot
    if len(sorted_eigenvalues) > 0 and T_observations > 0: # Ensure there are eigenvalues to plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(sorted_eigenvalues) + 1), sorted_eigenvalues, 'o-')
        plt.title('Scree Plot of Eigenvalues (Sample Covariance Matrix)')
        plt.xlabel('Principal Component Number')
        plt.ylabel('Eigenvalue')

        threshold_val = 1 + np.sqrt(N_assets / T_observations) if T_observations > 0 else np.inf
        plt.axhline(y=threshold_val, color='r', linestyle='--',
                    label=f'Threshold (1 + sqrt(N/T)) = {threshold_val:.2f}')
        if num_factors > 0:
            plt.axvline(x=num_factors, color='g', linestyle='--',
                        label=f'Num Factors Chosen = {num_factors}')
        plt.legend()
        plt.grid(True)

        plot_filename = "chapter_7_scree_plot.png"
        try:
            plt.savefig(plot_filename)
            print(f"\nScree plot saved to {plot_filename}")
        except Exception as e:
            print(f"\nCould not save scree plot: {e}. Attempting to display if possible.")
            try:
                plt.show()
            except Exception as display_e:
                print(f"Could not display scree plot: {display_e}")
    else:
        print("\nScree plot not generated due to lack of eigenvalues or T_observations=0.")


if __name__ == "__main__":
    run_chapter_7_example()
