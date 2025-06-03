import pandas as pd
import numpy as np
import yfinance as yf

# Assuming quant_elements_lib is installed or in PYTHONPATH
# If running directly from a repo structure, ensure paths are set up
try:
    from quant_elements_lib.core import FactorModelBase # Use __init__.py
    from quant_elements_lib.factor_models import decompose_alpha, rotate_factor_model # Use __init__.py
    from quant_elements_lib.utils import ordinary_least_squares # Use __init__.py
except ImportError:
    # Fallback for environments where quant_elements_lib is not installed (e.g. CI)
    # This requires the script to be run from a directory where Python can find the modules
    import sys
    sys.path.append('../') # Adjust if your script is in a different location relative to the lib
    from quant_elements_lib.core.factor_model_base import FactorModelBase
    from quant_elements_lib.factor_models.utils import decompose_alpha
    from quant_elements_lib.factor_models.transformations import rotate_factor_model
    from quant_elements_lib.utils.regression import ordinary_least_squares

# 2. Define a minimal concrete FactorModelBase subclass
class MySimpleFactorModel(FactorModelBase):
    """
    A simple concrete implementation of FactorModelBase for example purposes.
    """
    def __init__(self,
                 alpha: pd.Series,
                 B_loadings: pd.DataFrame,
                 factor_covariance: pd.DataFrame,
                 idiosyncratic_covariance: Union[pd.DataFrame, pd.Series],
                 factor_returns: Optional[pd.DataFrame] = None,
                 asset_universe: Optional[List[str]] = None,
                 factor_names: Optional[List[str]] = None):
        """
        Initializes MySimpleFactorModel.

        Args:
            alpha (pd.Series): Alpha values.
            B_loadings (pd.DataFrame): Factor loadings.
            factor_covariance (pd.DataFrame): Factor covariance matrix.
            idiosyncratic_covariance (Union[pd.DataFrame, pd.Series]): Idiosyncratic covariances.
            factor_returns (Optional[pd.DataFrame]): Historical or predicted factor returns.
            asset_universe (Optional[List[str]]): List of asset identifiers.
            factor_names (Optional[List[str]]): List of factor names.
        """
        # Call super() with arguments defined in FactorModelBase's __init__
        super().__init__(
            alpha=alpha,
            B_loadings=B_loadings,
            factor_covariance=factor_covariance,
            idiosyncratic_covariance=idiosyncratic_covariance
        )
        # Store additional attributes directly
        self.factor_returns = factor_returns
        self.asset_universe = asset_universe if asset_universe else list(alpha.index)
        self.factor_names = factor_names if factor_names else list(B_loadings.columns)
        # Note: The base FactorModelBase might not store factor_returns, asset_universe, factor_names
        # This subclass explicitly stores them. Some methods in transformations.py
        # (rotate_factor_model, project_factor_model_to_subset)
        # do check for 'factor_returns' attribute and handle it.

def run_example():
    """Runs the Chapter 4 linear models example."""
    print("--- Chapter 4: Linear Models Example ---")

    # 3. Fetch Data
    print("\n1. Fetching Data...")
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    market_ticker = 'SPY'
    all_tickers = tickers + [market_ticker]
    start_date = '2020-01-01'
    end_date = '2022-12-31'

    try:
        data = yf.download(all_tickers, start=start_date, end=end_date)['Adj Close']
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Please ensure you have an internet connection and yfinance is working.")
        # Create dummy data if yfinance fails
        print("Using dummy data for demonstration.")
        idx = pd.date_range(start_date, end_date, freq='B')
        data = pd.DataFrame(
            np.random.randn(len(idx), len(all_tickers)) * 0.01 + 0.0001,
            index=idx,
            columns=all_tickers
        )
        data = data.cumsum(axis=0) + 100 # Simple random walk prices

    log_returns = np.log(data / data.shift(1)).dropna()

    stock_returns = log_returns[tickers]
    market_returns = log_returns[market_ticker].rename('Market')

    print(f"Fetched {len(log_returns)} data points for {', '.join(all_tickers)}.")
    print("Sample log returns (head):")
    print(log_returns.head(3))

    # 4. Simulate a Simple Factor Model (Market Model)
    print("\n2. Simulating a Simple Factor Model (Market Model via OLS)...")
    alpha_list = []
    beta_list = []
    residual_variances_list = []
    asset_names = []

    for stock_ticker in tickers:
        Y_stock = stock_returns[stock_ticker]
        X_market = pd.DataFrame(market_returns) # OLS function expects X as DataFrame

        try:
            beta_hat, residuals, r_squared = ordinary_least_squares(Y_stock, X_market, add_intercept=True)

            alpha_list.append(beta_hat['intercept'])
            # Ensure 'Market' factor name is used, matches X_market column name
            beta_list.append(beta_hat[market_returns.name])
            residual_variances_list.append(np.var(residuals))
            asset_names.append(stock_ticker)

            print(f"  Regression for {stock_ticker} on {market_returns.name}: Alpha={beta_hat['intercept']:.6f}, Beta={beta_hat[market_returns.name]:.4f}, R^2={r_squared:.4f}")

        except Exception as e:
            print(f"  Could not run regression for {stock_ticker}: {e}")
            # Add placeholder values if regression fails
            alpha_list.append(0.0)
            beta_list.append(1.0)
            residual_variances_list.append(0.001) # Arbitrary non-zero variance
            asset_names.append(stock_ticker)


    alpha_series = pd.Series(alpha_list, index=asset_names, name='alpha')
    B_loadings_df = pd.DataFrame(beta_list, index=asset_names, columns=[market_returns.name]) # Factor is 'Market'

    # Factor covariance: Variance of the market factor returns
    factor_covariance_df = pd.DataFrame(
        [[market_returns.var()]],
        columns=[market_returns.name],
        index=[market_returns.name],
        dtype=float # Ensure float type
    )

    idiosyncratic_covariance_series = pd.Series(residual_variances_list, index=asset_names, name='idiosyncratic_variance')

    print("\nEstimated Model Components:")
    print("Alpha Series (N x 1):\n", alpha_series)
    print("\nB Loadings (N x K):\n", B_loadings_df)
    print("\nFactor Covariance (K x K):\n", factor_covariance_df)
    print("\nIdiosyncratic Covariance (Variances, N x 1):\n", idiosyncratic_covariance_series)

    # 5. Instantiate the Custom Factor Model
    print("\n3. Instantiating MySimpleFactorModel...")
    try:
        model = MySimpleFactorModel(
            alpha=alpha_series,
            B_loadings=B_loadings_df,
            factor_covariance=factor_covariance_df,
            idiosyncratic_covariance=idiosyncratic_covariance_series, # Pass as Series (diagonal)
            factor_returns=pd.DataFrame(market_returns) # T x K
        )
        print("  Model instantiated successfully.")
    except Exception as e:
        print(f"  Error instantiating model: {e}")
        return # Cannot proceed if model instantiation fails

    # 6. Demonstrate FactorModelBase methods
    print("\n4. Demonstrating FactorModelBase methods...")
    print("  Model Alpha:\n", model.alpha)
    print("\n  Model B_loadings:\n", model.B_loadings)
    print("\n  Model Factor Covariance:\n", model.factor_covariance)
    print("\n  Model Idiosyncratic Covariance (as Series):\n", model.idiosyncratic_covariance)

    total_asset_cov = model.calculate_total_asset_covariance_matrix()
    if total_asset_cov is not None:
        print("\n  Total Asset Covariance Matrix (Σ = BΩ_fB' + Ω_ε) (Snippet N x N):")
        print(total_asset_cov.head(3))
    else:
        print("\n  Could not calculate Total Asset Covariance Matrix (some components might be missing).")

    # Example: Predict systematic returns (requires factor returns)
    # Use last 5 days of market returns as hypothetical future factor returns
    if model.factor_returns is not None and not model.factor_returns.empty:
        hypothetical_factor_returns = model.factor_returns.tail()
        systematic_returns_pred = model.predict_systematic_returns(hypothetical_factor_returns)
        if systematic_returns_pred is not None:
            print("\n  Predicted Systematic Returns for last 5 periods (T x N):")
            print(systematic_returns_pred)
        else:
            print("\n  Could not predict systematic returns.")

        # Example: Decompose total returns
        # Use last 5 days of stock_returns and corresponding market_returns
        if len(stock_returns) >= 5 and len(market_returns) >=5:
            total_returns_sample = stock_returns.tail()
            factor_returns_sample = pd.DataFrame(market_returns.tail())

            decomposed = model.decompose_total_returns(total_returns_sample, factor_returns_sample)
            if decomposed:
                systematic_comp, idiosyncratic_comp = decomposed
                print("\n  Decomposed Returns (Systematic - T x N):")
                print(systematic_comp)
                print("\n  Decomposed Returns (Idiosyncratic - T x N):")
                print(idiosyncratic_comp)
            else:
                print("\n  Could not decompose total returns.")


    # 7. Demonstrate Alpha Decomposition
    print("\n5. Demonstrating Alpha Decomposition...")
    if model.alpha is not None and model.B_loadings is not None:
        try:
            alpha_spanned, alpha_orthogonal, lambda_spanned = decompose_alpha(model.alpha, model.B_loadings)
            print("  Original Alpha:\n", model.alpha)
            print("\n  Spanned Alpha (B @ λ_spanned):\n", alpha_spanned)
            print("\n  Orthogonal Alpha (α_total - α_spanned):\n", alpha_orthogonal)
            print("\n  Lambda Spanned (Factor Exposures for Spanned Alpha, K x 1):\n", lambda_spanned)
        except Exception as e:
            print(f"  Error during alpha decomposition: {e}")
    else:
        print("  Skipping alpha decomposition as model.alpha or model.B_loadings is None.")


    # 8. Demonstrate Factor Model Rotation
    print("\n6. Demonstrating Factor Model Rotation...")
    # For a single factor model, rotation is less interesting, but let's show the mechanism.
    # C_rotation maps original factors to new (rotated) factors.
    # C_rotation columns are new factor names, index are original factor names.
    # If C is KxK, B_rot = B @ inv(C), Omega_f_rot = C @ Omega_f @ C.T
    # Factor returns f_rot = f @ C.T

    # Simple 1x1 "rotation" (scaling)
    C_rotation_values = [[2.0]]
    original_factor_name = model.B_loadings.columns[0] # Should be 'Market'
    rotated_factor_name = 'Market_Rotated'

    C_rotation = pd.DataFrame(
        C_rotation_values,
        columns=[rotated_factor_name],
        index=[original_factor_name]
    )
    print(f"  Using Rotation Matrix C ({C_rotation.index.name} to {C_rotation.columns.name}):\n", C_rotation)

    try:
        rotated_model = rotate_factor_model(model, C_rotation)
        print("\n  Original Model B_loadings:\n", model.B_loadings)
        print("  Rotated Model B_loadings (B_orig @ inv(C)):\n", rotated_model.B_loadings)
        print("\n  Original Model Factor Covariance:\n", model.factor_covariance)
        print("  Rotated Model Factor Covariance (C @ Ω_f_orig @ C.T):\n", rotated_model.factor_covariance)

        # Compare systematic risk: B @ Ω_f @ B.T (should be invariant under rotation)
        # This is the systematic component of the asset covariance matrix
        if model.B_loadings is not None and model.factor_covariance is not None:
            sys_risk_original = model.B_loadings @ model.factor_covariance @ model.B_loadings.T
            print("\n  Systematic Risk (BΩB') - Original Model (Snippet):\n", sys_risk_original.iloc[:2,:2])

        if rotated_model.B_loadings is not None and rotated_model.factor_covariance is not None:
            sys_risk_rotated = rotated_model.B_loadings @ rotated_model.factor_covariance @ rotated_model.B_loadings.T
            print("\n  Systematic Risk (BΩB') - Rotated Model (Snippet):\n", sys_risk_rotated.iloc[:2,:2])

            if 'sys_risk_original' in locals():
                 print("\n  Are systematic risk matrices close (original vs. rotated)? ",
                       np.allclose(sys_risk_original, sys_risk_rotated))

        if hasattr(model, 'factor_returns') and model.factor_returns is not None and \
           hasattr(rotated_model, 'factor_returns') and rotated_model.factor_returns is not None:
            print("\n  Original Factor Returns (Snippet T x K_orig):\n", model.factor_returns.head(3))
            print("\n  Rotated Factor Returns (f_orig @ C.T) (Snippet T x K_new):\n", rotated_model.factor_returns.head(3))


    except Exception as e:
        print(f"  Error during factor model rotation: {e}")

    print("\n--- Example Script Finished ---")

if __name__ == '__main__':
    # For type hinting, these are good to have at the top level of the module too
    from typing import Union, Optional, List
    run_example()
