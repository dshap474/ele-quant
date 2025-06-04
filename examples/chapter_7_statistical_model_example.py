import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from quant_elements_lib.factor_models import StatisticalFactorModel
from quant_elements_lib.factor_models.utils import choose_num_factors_threshold
# from quant_elements_lib.utils import shrink_eigenvalues_spiked_model # Optional import

def run_chapter_7_example():
    print("Running Chapter 7: Statistical Factor Models Example\n")

    # 1. Data Fetching and Preparation
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V',
        'PG', 'UNH', 'HD', 'MA', 'BAC', 'DIS', 'ADBE', 'CRM', 'NFLX', 'XOM',
        'CVX', 'PFE', 'MRK', 'KO', 'PEP', 'WMT', 'MCD', 'COST', 'INTC', 'CSCO'
        # Add more or use a broad index component list if feasible
    ]
    # Reduce list for faster example run in CI/testing if needed
    # tickers = tickers[:15]

    start_date = '2020-01-01'
    end_date = '2023-12-31'

    print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
        if data.empty:
            # This case might happen if yf.download returns an empty DataFrame but no exception
            # e.g. for very specific ticker lists or network issues not caught by yf's internal retries.
            raise ValueError("No data returned from yfinance. Check tickers or date range.")
    except Exception as e:
        print(f"Error fetching data with yfinance: {e}")
        # Fallback for environments where yfinance might fail (e.g. no internet)
        # Create dummy data for N assets, T observations
        T_obs_dummy, N_assets_dummy = 252*3, len(tickers) # Approx 3 years
        dummy_returns_np = np.random.randn(T_obs_dummy, N_assets_dummy) * 0.01 # Daily returns ~ N(0, 0.01^2)
        # Create price series from returns
        dummy_prices = np.exp(np.cumsum(dummy_returns_np, axis=0)) * 100 # Start price 100
        data = pd.DataFrame(dummy_prices,
                            index=pd.date_range(start_date, periods=T_obs_dummy, freq='B'),
                            columns=tickers) # Ensure columns match original ticker list
        print("Using dummy data as yfinance fallback.")


    if data.empty:
        print("No data fetched or generated. Exiting.")
        return

    # Drop columns (assets) that have no price data at all (e.g. yf returns NaNs for a ticker)
    data = data.dropna(axis=1, how='all')
    # Forward fill and then backfill to handle missing values for individual assets
    data = data.ffill().bfill()
    # Drop again if any asset is still all NaN (e.g. if it had no data from yf to begin with and ffill/bfill couldn't help)
    data = data.dropna(axis=1, how='all')

    if data.shape[1] == 0:
        print("No valid asset data remaining after cleaning. Exiting.")
        return

    final_tickers = data.columns.tolist()
    print(f"Using data for {len(final_tickers)} assets.")

    log_returns = np.log(data / data.shift(1)).dropna() # T x N

    if log_returns.empty or log_returns.shape[0] < 2 or log_returns.shape[1] < 2:
        print("Not enough data points or assets after processing for a meaningful model. Exiting example.")
        return

    # 2. Determine Number of Factors
    returns_demeaned_for_eig = log_returns - log_returns.mean(axis=0)
    T_obs, N_assets = returns_demeaned_for_eig.shape

    sample_eigenvalues = np.array([1.0]) # Default dummy in case all calcs fail
    if T_obs <= 1: # Should be caught by log_returns.shape[0] < 2 earlier
        print("Not enough observations (T<=1) to calculate eigenvalues reliably. Using dummy eigenvalues.")
        sample_eigenvalues = np.sort(np.random.rand(max(1,N_assets)) * max(1,N_assets))[::-1] + 0.1
    elif T_obs <= N_assets :
        try:
            # SVD on R_demeaned (T x N = U S Vh)
            # Eigenvalues of (1/(T-1)) * R_demeaned.T @ R_demeaned are s**2 / (T-1)
            _, s_vals, _ = np.linalg.svd(returns_demeaned_for_eig.values, full_matrices=False)
            sample_eigenvalues = (s_vals**2) / (T_obs - 1)
        except np.linalg.LinAlgError as e:
            print(f"SVD failed for eigenvalue calculation: {e}. Using dummy eigenvalues.")
            sample_eigenvalues = np.sort(np.random.rand(N_assets) * N_assets)[::-1] + 0.1
    else: # T > N
        try:
            sample_cov_matrix = np.cov(returns_demeaned_for_eig.values, rowvar=False, ddof=1)
            sample_eigenvalues = np.linalg.eigvalsh(sample_cov_matrix)
            sample_eigenvalues = np.sort(sample_eigenvalues)[::-1]
        except np.linalg.LinAlgError as e:
            print(f"Eigendecomposition failed: {e}. Using dummy eigenvalues.")
            sample_eigenvalues = np.sort(np.random.rand(N_assets) * N_assets)[::-1] + 0.1

    print(f"\nTop 5 sample eigenvalues: {sample_eigenvalues[:5]}")

    if N_assets <=1 :
        num_factors_selected = N_assets # 1 if N_assets=1, 0 if N_assets=0
    else:
        num_factors_selected = choose_num_factors_threshold(sample_eigenvalues, N_assets, T_obs)

    # Cap number of factors for practical example
    if N_assets > 1:
        # Ensure num_factors < N_assets for some model properties (esp. PPCA later)
        # Also cap at a reasonable number like 10 for example display
        num_factors_selected = min(num_factors_selected, N_assets - 1, 10)
    elif N_assets == 1:
        num_factors_selected = 1

    if num_factors_selected == 0 and N_assets > 0 :
        num_factors_selected = 1 # Default to 1 factor if assets exist but threshold gave 0

    print(f"Selected number of factors using threshold method (capped): {num_factors_selected}")

    # 3. Fit Statistical Factor Model (PCA)
    if num_factors_selected > 0:
        print("\nFitting StatisticalFactorModel using PCA...")
        statistical_model_pca = StatisticalFactorModel(
            asset_universe=final_tickers,
            num_factors_to_extract=num_factors_selected
        )
        try:
            statistical_model_pca.fit(log_returns, estimation_method='PCA')
            print("PCA Model fitting complete.")
            print("\nPCA Model - Alpha (Mean Returns):")
            print(statistical_model_pca.alpha.head())
            print("\nPCA Model - Factor Loadings (B_loadings) - First 5 assets, all factors:")
            print(statistical_model_pca.B_loadings.head(5))
            print("\nPCA Model - Factor Returns (first 5 periods, all factors):")
            print(statistical_model_pca.factor_returns.head(5))
            print("\nPCA Model - Factor Covariance Matrix:")
            print(statistical_model_pca.factor_covariance)
            print("\nPCA Model - Idiosyncratic Covariance (Variances - first 5 assets):")
            print(statistical_model_pca.idiosyncratic_covariance.head(5))
        except Exception as e:
            print(f"Error fitting PCA model or accessing attributes: {e}")
    else:
        print("Skipping PCA model fitting as num_factors_selected is 0.")

    # 4. Scree Plot
    if N_assets > 0 and T_obs > 1 and len(sample_eigenvalues) > 0 and sample_eigenvalues[0] != 1.0: # Avoid plotting dummy
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(sample_eigenvalues) + 1), sample_eigenvalues, 'o-', label='Eigenvalues')
        plt.title('Scree Plot of Eigenvalues (from Sample Covariance of Demeaned Returns)')
        plt.xlabel('Rank')
        plt.ylabel('Eigenvalue')

        if N_assets > 0 and T_obs > 0: # Check to avoid division by zero if T_obs became 0
            threshold_val = 1 + np.sqrt(N_assets / T_obs)
            plt.axhline(threshold_val, color='r', linestyle='--',
                        label=f'Threshold (1 + sqrt(N/T)) = {threshold_val:.2f}')
        if num_factors_selected > 0:
             plt.axvline(num_factors_selected, color='g', linestyle='--',
                        label=f'Selected Factors = {num_factors_selected}')

        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        plt.tight_layout()
        plot_filename = "chapter_7_scree_plot.png"
        try:
            plt.savefig(plot_filename)
            print(f"\nScree plot saved to {plot_filename}")
        except Exception as e:
            print(f"Error saving scree plot: {e}")
        plt.close()
    else:
        print("\nSkipping scree plot due to insufficient/dummy data for meaningful plot.")

    # (Optional) Demonstrate PPCA
    if N_assets > 1 :
        print("\nFitting StatisticalFactorModel using PPCA (optional demo)...")
        # For PPCA, num_factors must be < N_assets.
        num_factors_ppca = min(5, N_assets - 1)
        if num_factors_ppca <= 0 : # Ensure at least 1 factor for PPCA if N_assets > 1
             num_factors_ppca = 1

        statistical_model_ppca = StatisticalFactorModel(
            asset_universe=final_tickers,
            num_factors_to_extract=num_factors_ppca
        )
        try:
            statistical_model_ppca.fit(log_returns, estimation_method='PPCA')
            print("PPCA Model fitting complete.")
            print("\nPPCA Model - Factor Loadings (B_loadings) - First 5 assets, all factors:")
            print(statistical_model_ppca.B_loadings.head(5))
            print("\nPPCA Model - Factor Covariance Matrix:")
            print(statistical_model_ppca.factor_covariance)
            print("\nPPCA Model - Idiosyncratic Covariance (Constant Variance):")
            print(statistical_model_ppca.idiosyncratic_covariance.head(5))
        except Exception as e:
            print(f"Error fitting PPCA model or accessing attributes: {e}")
    else:
        print("\nSkipping PPCA example as N_assets <= 1 (PPCA requires num_factors < N_assets).")

    print("\nChapter 7 Example Complete.")

if __name__ == "__main__":
    run_chapter_7_example()
