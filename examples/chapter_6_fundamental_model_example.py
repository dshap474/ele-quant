import pandas as pd
import numpy as np
from ele_quant.factor_models import FundamentalFactorModel
from ele_quant.utils import winsorize_series
from ele_quant.returns_analysis import calculate_hedged_returns
# calculate_forward_premium is available but not used in this example
# as we are not generating separate forward rate data.

def run_chapter_6_examples():
    print("--- Starting Chapter 6 Example Script ---")

    # Setup
    n_periods = 100
    n_assets = 20
    n_factors = 3
    dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='M')
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    factor_names = ['Value', 'Momentum', 'Quality']
    currency_names = ['EUR', 'JPY'] # Relative to USD base (e.g. EUR/USD, JPY/USD returns)

    # Random data generation
    np.random.seed(42)
    asset_returns_data = np.random.randn(n_periods, n_assets) * 0.02 # Monthly returns
    asset_returns = pd.DataFrame(asset_returns_data, index=dates, columns=asset_names)

    # Dynamic factor exposures: MultiIndex (date, asset)
    exposures_list = []
    for date_val in dates: # Renamed date to date_val to avoid conflict with pd.Timestamp.date
        for asset_val in asset_names: # Renamed asset to asset_val
            exposures_list.append([date_val, asset_val] + list(np.random.rand(n_factors)))
    factor_exposures_multiidx = pd.DataFrame(exposures_list, columns=['date', 'asset'] + factor_names)
    factor_exposures_multiidx = factor_exposures_multiidx.set_index(['date', 'asset'])

    # FX Spot Returns (LC per USD, e.g., EUR/USD means EUR return relative to USD)
    # A positive return for EUR means EUR appreciated against USD.
    fx_spot_returns_data = np.random.randn(n_periods, len(currency_names)) * 0.01
    fx_spot_returns = pd.DataFrame(fx_spot_returns_data, index=dates, columns=currency_names)

    # Asset to currency map (example: first half EUR, second half JPY)
    asset_to_currency_map = {asset: 'EUR' for asset in asset_names[:n_assets//2]}
    asset_to_currency_map.update({asset: 'JPY' for asset in asset_names[n_assets//2:]})

    print("\n--- Sample Data Generated ---")
    print("Asset Returns (head):\n", asset_returns.head())
    print("\nFactor Exposures (head):\n", factor_exposures_multiidx.head())
    print("\nFX Spot Returns (LC/Base Currency) (head):\n", fx_spot_returns.head())
    print("\nAsset to Currency Map (sample):\n", list(asset_to_currency_map.items())[:5])
    print("\n--- End of Sample Data ---\n")

    # --- Testing FundamentalFactorModel ---
    print("\n--- Testing FundamentalFactorModel ---")
    model = FundamentalFactorModel()

    # Fit with dynamic exposures and sample covariance
    print("\nFitting model with 'sample' covariance method...")
    model.fit(asset_returns, factor_exposures_multiidx, covariance_method='sample')

    if model.factor_returns is not None:
        print("\nEstimated Factor Returns (head):\n", model.factor_returns.head())
    else:
        print("\nFactor Returns could not be estimated.")

    if model.factor_covariance_matrix is not None:
        print("\nEstimated Factor Covariance Matrix:\n", model.factor_covariance_matrix)
    else:
        print("\nFactor Covariance could not be estimated.")

    if model.idiosyncratic_variance is not None:
        print("\nEstimated Idiosyncratic Variance (head):\n", model.idiosyncratic_variance.head())
    else:
        print("\nIdiosyncratic Variance could not be estimated.")

    # Prediction example
    if model.factor_loadings is not None and model.factor_returns is not None:
        try:
            # predict() uses self.factor_loadings (latest slice from dynamic exposures if fit with dynamic)
            # and mean of self.factor_returns
            predicted_returns = model.predict()
            print("\nPredicted Asset Returns (using latest fitted loadings - head):\n", predicted_returns.head())

            # Example with future exposures (can be same as latest for demo)
            # future_exposures_example = model.factor_loadings
            # predicted_returns_future = model.predict(factor_exposures_future=future_exposures_example)
            # print("\nPredicted Asset Returns (using provided future exposures - head):\n", predicted_returns_future.head())

        except ValueError as e:
            print(f"\nError during prediction: {e}")
    else:
        print("\nSkipping prediction as model fitting might have failed or loadings/factor_returns are not set.")

    # Fit with EWMA covariance as an alternative
    print("\nRe-fitting model with 'ewma' covariance method (span=60)...")
    model_ewma = FundamentalFactorModel()
    model_ewma.fit(asset_returns, factor_exposures_multiidx, covariance_method='ewma', span=60, min_periods_ewma=n_factors*2)
    if model_ewma.factor_covariance_matrix is not None:
        print("\nEstimated Factor Covariance Matrix (EWMA):\n", model_ewma.factor_covariance_matrix)
    else:
        print("\nEWMA Factor Covariance could not be estimated.")


    # --- Testing Winsorization ---
    print("\n\n--- Testing Winsorization ---")
    # Introduce obvious outliers and NaNs for better demonstration
    sample_series_for_winsorizing = asset_returns[asset_names[0]].copy()
    sample_series_for_winsorizing.iloc[0] = 0.5  # Outlier
    sample_series_for_winsorizing.iloc[1] = -0.4 # Outlier
    sample_series_for_winsorizing.iloc[5:10] = np.nan # Add some NaNs

    print("\nOriginal Series for Winsorization (head with introduced outliers/NaNs):\n", sample_series_for_winsorizing.head(15))

    winsorized_quantile = winsorize_series(sample_series_for_winsorizing, method='quantile', lower_quantile=0.05, upper_quantile=0.95)
    print("\nWinsorized (Quantile 5%-95%) Series (head):\n", winsorized_quantile.head(15))
    print(f"Original min: {sample_series_for_winsorizing.min():.4f}, max: {sample_series_for_winsorizing.max():.4f}")
    print(f"Quantile Winsorized min: {winsorized_quantile.min():.4f}, max: {winsorized_quantile.max():.4f}")


    winsorized_zscore = winsorize_series(sample_series_for_winsorizing, method='z_score_robust', z_threshold=2.0)
    print("\nWinsorized (Robust Z-score +/-2) Series (head):\n", winsorized_zscore.head(15))
    print(f"Robust Z-score Winsorized min: {winsorized_zscore.min():.4f}, max: {winsorized_zscore.max():.4f}")


    # --- Testing Currency Hedging ---
    print("\n\n--- Testing Currency Hedging ---")
    # asset_returns are local currency returns for assets specified in asset_to_currency_map.
    # The function calculate_hedged_returns will use the corresponding fx_spot_returns.

    asset_returns_local = asset_returns.copy()

    hedged_returns = calculate_hedged_returns(
        asset_returns_local_currency=asset_returns_local,
        fx_spot_returns_local_per_base=fx_spot_returns,
        hedge_ratio=0.9,
        cost_of_hedge_per_period=0.0005, # 0.05% per period cost
        asset_to_currency_map=asset_to_currency_map
    )
    # Display for first 3 assets and their corresponding currencies
    assets_to_show = asset_names[:3]
    currencies_to_show = [asset_to_currency_map[asset] for asset in assets_to_show]

    print(f"\nOriginal Local Currency Returns ({', '.join([f'{a} ({c})' for a,c in zip(assets_to_show, currencies_to_show)])} - head):\n", asset_returns_local[assets_to_show].head())
    print(f"\nFX Spot Returns ({', '.join(currencies_to_show)} - head):\n", fx_spot_returns[list(set(currencies_to_show))].head()) # Show unique relevant currencies
    print(f"\nCalculated Hedged Returns (USD base, {', '.join(assets_to_show)} - head):\n", hedged_returns[assets_to_show].head())

    print("\n\n--- Chapter 6 Example Script Finished ---")

if __name__ == '__main__':
    run_chapter_6_examples()
