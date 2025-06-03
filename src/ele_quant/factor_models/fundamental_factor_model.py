import pandas as pd
from typing import Optional, List
import statsmodels.api as sm # Added import
from ele_quant.core.factor_model_base import FactorModelBase

try:
    from sklearn.covariance import LedoitWolf
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class FundamentalFactorModel(FactorModelBase):
    """
    Fundamental Factor Model.

    This model uses fundamental factors (e.g., P/E ratio, market cap) to explain asset returns.
    The model is typically estimated using cross-sectional regression.
    """

    def __init__(self,
                 factor_loadings: Optional[pd.DataFrame] = None,
                 factor_returns: Optional[pd.DataFrame] = None,
                 factor_covariance_matrix: Optional[pd.DataFrame] = None,
                 idiosyncratic_variance: Optional[pd.DataFrame] = None,
                 assets: Optional[List[str]] = None,
                 factors: Optional[List[str]] = None):
        """
        Initialize the FundamentalFactorModel.

        Parameters
        ----------
        factor_loadings : Optional[pd.DataFrame], optional
            Factor loadings (beta), by default None. Index is assets, columns are factors.
        factor_returns : Optional[pd.DataFrame], optional
            Factor returns, by default None. Index is time, columns are factors.
        factor_covariance_matrix : Optional[pd.DataFrame], optional
            Factor covariance matrix, by default None. Index and columns are factors.
        idiosyncratic_variance : Optional[pd.DataFrame], optional
            Idiosyncratic variance of assets, by default None. Index is time or assets, columns are assets.
            If index is time, it's a time series of idiosyncratic variances.
            If index is assets, it's a static idiosyncratic variance.
        assets : Optional[List[str]], optional
            List of asset identifiers, by default None.
        factors : Optional[List[str]], optional
            List of factor identifiers, by default None.
        """
        super().__init__(factor_loadings, factor_returns, factor_covariance_matrix, idiosyncratic_variance, assets, factors)
        # Additional initializations specific to fundamental models can be added here

    def fit(self,
            returns: pd.DataFrame,
            factor_exposures: pd.DataFrame,
            **kwargs) -> None:
        """
        Fit the fundamental factor model using cross-sectional regression.

        This method will estimate the factor returns, factor covariance matrix,
        and idiosyncratic variances.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns. Index is time, columns are assets.
        factor_exposures : pd.DataFrame
            Factor exposures (characteristics) for each asset at each point in time.
            Index is a MultiIndex (time, asset), columns are factors.
            Or, index is assets and columns are factors if exposures are static over the estimation period.
        **kwargs :
            Additional arguments for the fitting process.
            For example, arguments for cross-sectional regression type,
            factor covariance estimation method, etc.
        """
        # Placeholder implementation
        # Detailed implementation will be done in subsequent tasks.
        print("Fitting FundamentalFactorModel...")
        # Example:
        # self.factor_returns = self._estimate_factor_returns(returns, factor_exposures, **kwargs)
        # self.factor_covariance_matrix = self._estimate_factor_covariance(self.factor_returns, **kwargs)
        # self.idiosyncratic_variance = self._estimate_idiosyncratic_variance(returns, factor_exposures, self.factor_returns, **kwargs)
        # self.factor_loadings = factor_exposures # Or derived from it
        # self.assets = returns.columns.tolist()
        # self.factors = factor_exposures.columns.tolist()

        # Estimate factor returns
        self.factor_returns = self._estimate_factor_returns(returns, factor_exposures, **kwargs)

        # Assign factor_exposures to factor_loadings
        # This assumes that the provided factor_exposures are the loadings.
        # If factor_exposures is (time, asset) by factors, and we need (asset) by factors,
        # we might need to decide how to aggregate or select.
        # For now, if it's multi-indexed, let's take the latest period's exposures.
        if isinstance(factor_exposures.index, pd.MultiIndex):
            latest_time = factor_exposures.index.levels[0].max()
            self.factor_loadings = factor_exposures.xs(latest_time, level=0)
        else:
            self.factor_loadings = factor_exposures # Assumes (assets x factors)

        self.assets = returns.columns.tolist()
        self.factors = self.factor_loadings.columns.tolist() # factor_exposures.columns might be more direct if no aggregation happened

        # Estimate factor covariance matrix
        if self.factor_returns is not None and not self.factor_returns.empty:
            self.factor_covariance_matrix = self._estimate_factor_covariance(self.factor_returns, **kwargs)
        else:
            self.factor_covariance_matrix = None
            print("Warning: Factor returns are None or empty. Skipping factor covariance estimation.")

        # Estimate idiosyncratic variance
        if self.factor_returns is not None and not self.factor_returns.empty and \
           returns is not None and not returns.empty and \
           factor_exposures is not None and not factor_exposures.empty:
            self.idiosyncratic_variance = self._estimate_idiosyncratic_variance(
                returns,
                factor_exposures, # Pass the original factor_exposures from fit()
                self.factor_returns,
                **kwargs
            )
        else:
            self.idiosyncratic_variance = None
            print("Warning: Missing returns, factor_exposures, or factor_returns. Skipping idiosyncratic variance estimation.")

    def predict(self,
                factor_exposures_future: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Predict asset returns using the fitted model.

        If future factor exposures are provided, they are used. Otherwise, the latest
        known factor loadings are used. Factor returns are typically assumed to be their
        historical mean or predicted by a separate model.

        Parameters
        ----------
        factor_exposures_future : Optional[pd.DataFrame], optional
            Future factor exposures for the assets, by default None.
            If None, uses the latest factor_loadings from the fitted model.
            Index is assets, columns are factors.

        Returns
        -------
        pd.DataFrame
            Predicted asset returns. Index is assets, columns are ['expected_return'].
        """
        if self.factor_returns is None or self.factor_loadings is None:
            raise ValueError("Model has not been fitted yet, or factor returns/loadings are missing.")

        # Placeholder implementation for prediction logic
        # This typically involves: E[R] = Beta * E[f]
        # where E[f] might be historical mean of factor_returns or a forecast
        print("Predicting with FundamentalFactorModel...")

        current_loadings = factor_exposures_future if factor_exposures_future is not None else self.factor_loadings

        if current_loadings is None:
            raise ValueError("Factor exposures/loadings must be available for prediction.")

        # Assuming E[f] is the historical mean of factor_returns
        # Ensure factor_returns has a DatetimeIndex for proper time-series mean
        if isinstance(self.factor_returns.index, pd.DatetimeIndex):
            expected_factor_returns = self.factor_returns.mean()
        else: # If factor_returns is not time-indexed, assume it's already the expected values
            expected_factor_returns = self.factor_returns.squeeze() # Squeeze if it's a single row DataFrame

        if not isinstance(expected_factor_returns, pd.Series):
             raise ValueError("Expected factor returns could not be determined as a pd.Series.")

        # Align factors
        common_factors = current_loadings.columns.intersection(expected_factor_returns.index)
        if len(common_factors) == 0:
            raise ValueError("No common factors between loadings and expected factor returns.")

        aligned_loadings = current_loadings[common_factors]
        aligned_expected_factor_returns = expected_factor_returns[common_factors]

        predicted_returns = aligned_loadings.dot(aligned_expected_factor_returns)

        return pd.DataFrame(predicted_returns, columns=['expected_return'])

    def _estimate_factor_returns(self, returns: pd.DataFrame, factor_exposures: pd.DataFrame, **kwargs) -> Optional[pd.DataFrame]:
        """
        Estimate factor returns using cross-sectional regression for each time period.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns. Index is time, columns are assets.
        factor_exposures : pd.DataFrame
            Factor exposures.
            If MultiIndex (time, asset) by factors: dynamic exposures.
            If Index (asset) by factors: static exposures.
        **kwargs :
            Additional arguments.

        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame of factor returns. Index is time, columns are factors.
            Returns None if estimation is not possible.
        """
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise ValueError("Returns DataFrame must have a DatetimeIndex.")

        all_factor_returns = {}
        time_periods = returns.index.unique()

        # Determine if exposures are dynamic or static
        is_static_exposures = not isinstance(factor_exposures.index, pd.MultiIndex)
        if not is_static_exposures and not all(level in factor_exposures.index.names for level in ['time', 'asset']):
             raise ValueError("Factor exposures MultiIndex must contain 'time' and 'asset' levels if not static.")


        for period in time_periods:
            period_returns_series = returns.loc[period]

            if is_static_exposures:
                period_exposures_df = factor_exposures
            else:
                if period not in factor_exposures.index.levels[0]:
                    print(f"Warning: Skipping period {period} due to missing factor exposures.")
                    continue
                period_exposures_df = factor_exposures.xs(period, level='time')

            # Align data for the current period
            common_assets = period_returns_series.index.intersection(period_exposures_df.index)

            if len(common_assets) < period_exposures_df.shape[1]: # Not enough assets for regression
                print(f"Warning: Skipping period {period} due to insufficient common assets for regression.")
                continue

            y = period_returns_series.loc[common_assets].dropna()
            X = period_exposures_df.loc[common_assets].dropna(axis=0, how='any') # Drop rows with any NaN in exposures

            # Align y and X again after dropping NaNs
            aligned_index = y.index.intersection(X.index)
            if len(aligned_index) < X.shape[1]:
                print(f"Warning: Skipping period {period} due to insufficient data after NaN removal for regression.")
                continue

            y = y.loc[aligned_index]
            X = X.loc[aligned_index]

            if X.empty or y.empty:
                print(f"Warning: Skipping period {period} due to empty data after alignment.")
                continue

            try:
                # Add constant if specified in kwargs, e.g., add_constant=True
                # For pure factor returns, typically no constant is added here.
                # The factors themselves are the explanatory variables.
                # X_reg = sm.add_constant(X) if kwargs.get('add_constant_cs', False) else X
                X_reg = X

                model = sm.OLS(y, X_reg, missing='drop').fit()
                all_factor_returns[period] = model.params # Coefficients are the factor returns
            except Exception as e:
                print(f"Warning: Could not perform OLS regression for period {period}. Error: {e}")
                continue

        if not all_factor_returns:
            print("Warning: Factor returns could not be estimated for any period.")
            return None

        factor_returns_df = pd.DataFrame.from_dict(all_factor_returns, orient='index')
        factor_returns_df.index.name = 'time' # Ensure index name is set
        return factor_returns_df

    def _estimate_factor_covariance(self, factor_returns: pd.DataFrame, **kwargs) -> Optional[pd.DataFrame]:
        """
        Estimate the factor covariance matrix.

        Parameters
        ----------
        factor_returns : pd.DataFrame
            DataFrame of factor returns. Index is time, columns are factors.
        **kwargs :
            Additional arguments, including 'covariance_method' (str)
            and parameters for specific methods (e.g., 'span' for 'ewma').

        Returns
        -------
        Optional[pd.DataFrame]
            Estimated factor covariance matrix (factors x factors).
            Returns None if estimation is not possible.
        """
        if factor_returns is None or factor_returns.empty:
            # This case should ideally be caught before calling this method, e.g., in fit()
            print("Critical: Factor returns are None or empty passed to _estimate_factor_covariance.")
            return None

        n_periods, n_factors = factor_returns.shape
        factor_names = factor_returns.columns.tolist()

        if n_periods < 2:
            print(f"Warning: Not enough data points ({n_periods}) in factor returns to estimate covariance robustly for {n_factors} factors.")
            # Return a matrix of NaNs with appropriate shape
            nan_matrix = pd.DataFrame(float('nan'), index=factor_names, columns=factor_names)
            return nan_matrix

        # Drop rows with all NaNs in factor_returns, can happen if a period had no valid regression
        factor_returns_cleaned = factor_returns.dropna(how='all')
        if factor_returns_cleaned.shape[0] < 2:
            print(f"Warning: After cleaning NaNs, not enough data points ({factor_returns_cleaned.shape[0]}) in factor returns.")
            nan_matrix = pd.DataFrame(float('nan'), index=factor_names, columns=factor_names)
            return nan_matrix

        # Update n_periods after cleaning
        n_periods = factor_returns_cleaned.shape[0]


        covariance_method = kwargs.get('covariance_method', 'sample').lower()
        cov_matrix = None

        if covariance_method == 'sample':
            cov_matrix = factor_returns_cleaned.cov()
        elif covariance_method == 'ewma':
            # Default span to n_periods; min_periods for cov is usually 2
            span = kwargs.get('span', n_periods)
            min_periods_ewma = kwargs.get('min_periods_ewma', 2)

            if span < 1: # span must be >= 1 for ewm object itself.
                 print(f"Warning: EWMA span is {span}, must be >= 1. Falling back to sample covariance.")
                 cov_matrix = factor_returns_cleaned.cov()
            elif n_periods < min_periods_ewma : # Not enough periods for EWMA
                print(f"Warning: Not enough periods ({n_periods}) for EWMA with min_periods_ewma={min_periods_ewma}. Falling back to sample covariance.")
                cov_matrix = factor_returns_cleaned.cov()
            else:
                try:
                    # .cov(pairwise=True) returns a multi-indexed DataFrame (time, factor)
                    ewm_cov_multi = factor_returns_cleaned.ewm(span=span, min_periods=min_periods_ewma, adjust=True).cov(pairwise=True)
                    if not ewm_cov_multi.empty:
                        # Get the last full covariance matrix
                        cov_matrix_values = ewm_cov_multi.iloc[-n_factors:].values
                        cov_matrix = pd.DataFrame(cov_matrix_values, index=factor_names, columns=factor_names)
                    else:
                        print("Warning: EWMA covariance calculation resulted in an empty DataFrame. Falling back to sample covariance.")
                        cov_matrix = factor_returns_cleaned.cov()
                except Exception as e:
                    print(f"Error during EWMA covariance calculation: {e}. Falling back to sample covariance.")
                    cov_matrix = factor_returns_cleaned.cov()

        elif covariance_method == 'ledoit_wolf':
            if SKLEARN_AVAILABLE:
                if n_periods < n_factors: # LedoitWolf might still work but can be unstable/warn
                     print(f"Warning: For LedoitWolf, number of observations ({n_periods}) is less than number of features ({n_factors}).")
                try:
                    # LedoitWolf().fit() expects observations as rows, features as columns.
                    # factor_returns_cleaned should have NaNs handled appropriately for sklearn
                    # Most sklearn estimators don't handle NaNs, so drop any remaining NaNs
                    # (though factor_returns_cleaned.dropna(how='all') already did this for rows)
                    # For LedoitWolf, it's better if NaNs in individual cells are imputed or rows dropped.
                    # Let's assume factor_returns_cleaned is good enough or use dropna() for safety.
                    lw_input = factor_returns_cleaned.dropna()
                    if lw_input.shape[0] < 2: # Not enough data after dropping all NaNs
                        print("Warning: Not enough data for LedoitWolf after dropping all NaNs. Falling back to sample covariance.")
                        cov_matrix = factor_returns_cleaned.cov()
                    else:
                        lw = LedoitWolf(assume_centered=kwargs.get('assume_centered', False))
                        lw.fit(lw_input)
                        cov_matrix = pd.DataFrame(lw.covariance_, index=factor_names, columns=factor_names)
                except Exception as e:
                    print(f"Warning: LedoitWolf estimation failed: {e}. Falling back to sample covariance.")
                    cov_matrix = factor_returns_cleaned.cov()
            else:
                print("Warning: Scikit-learn is not available. LedoitWolf cannot be used. Falling back to sample covariance.")
                cov_matrix = factor_returns_cleaned.cov()
        else:
            print(f"Warning: Unknown covariance_method '{covariance_method}'. Falling back to sample covariance.")
            cov_matrix = factor_returns_cleaned.cov()

        return cov_matrix

    def _estimate_idiosyncratic_variance(self,
                                     returns: pd.DataFrame,
                                     factor_exposures_input: pd.DataFrame,
                                     factor_returns: pd.DataFrame,
                                     **kwargs) -> Optional[pd.DataFrame]:
        """
        Estimate idiosyncratic variance of assets.

        Residuals are calculated for each asset over time, and then the variance
        of these residuals is taken as the idiosyncratic variance.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns (time by assets).
        factor_exposures_input : pd.DataFrame
            Factor exposures. Can be static (assets by factors) or dynamic
            (MultiIndex (time, asset) by factors). This is the original
            factor_exposures data passed to the fit method.
        factor_returns : pd.DataFrame
            Factor returns (time by factors).
        **kwargs :
            Additional arguments (currently unused).

        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame with asset identifiers as index and a single column
            'idiosyncratic_variance', or None if estimation is not possible.
        """
        if returns is None or returns.empty or \
           factor_exposures_input is None or factor_exposures_input.empty or \
           factor_returns is None or factor_returns.empty:
            print("Warning: Required data for idiosyncratic variance estimation is missing or empty.")
            return None

        if not isinstance(returns.index, pd.DatetimeIndex) or \
           not isinstance(factor_returns.index, pd.DatetimeIndex):
            print("Warning: 'returns' and 'factor_returns' must have a DatetimeIndex.")
            return None

        all_residuals_dict = {asset: [] for asset in returns.columns}

        # Determine if exposures are dynamic or static
        is_static_exposures = not isinstance(factor_exposures_input.index, pd.MultiIndex)
        if not is_static_exposures and not all(level in factor_exposures_input.index.names for level in ['time', 'asset']):
             raise ValueError("Factor exposures MultiIndex must contain 'time' and 'asset' levels if not static.")

        # Common time periods between returns and factor_returns
        common_time_periods = returns.index.intersection(factor_returns.index)
        if common_time_periods.empty:
            print("Warning: No common time periods between returns and factor_returns for idiosyncratic variance.")
            return None

        for t_idx in common_time_periods:
            period_asset_returns = returns.loc[t_idx]  # Series: index=assets, values=returns
            period_factor_returns = factor_returns.loc[t_idx] # Series: index=factors, values=factor_returns

            current_period_exposures: Optional[pd.DataFrame] = None
            if is_static_exposures:
                current_period_exposures = factor_exposures_input
            else: # Dynamic exposures
                if t_idx in factor_exposures_input.index.get_level_values('time'):
                    current_period_exposures = factor_exposures_input.xs(t_idx, level='time') # DataFrame: index=assets, columns=factors
                else:
                    # print(f"Info: Skipping period {t_idx} for idiosyncratic variance: missing dynamic factor exposures for this period.")
                    continue

            if current_period_exposures is None or current_period_exposures.empty:
                # print(f"Info: Skipping period {t_idx} due to empty exposures for this period.")
                continue

            # Align assets: common between period_asset_returns and current_period_exposures
            common_assets = period_asset_returns.index.intersection(current_period_exposures.index)

            aligned_asset_returns = period_asset_returns[common_assets].dropna()
            if aligned_asset_returns.empty:
                continue

            aligned_exposures = current_period_exposures.loc[aligned_asset_returns.index]

            # Align factors: common between aligned_exposures and period_factor_returns
            common_factors = aligned_exposures.columns.intersection(period_factor_returns.index)

            final_exposures = aligned_exposures[common_factors]
            final_factor_returns = period_factor_returns[common_factors]

            if final_exposures.empty or final_factor_returns.empty or final_exposures.shape[1] != len(final_factor_returns):
                # print(f"Info: Skipping period {t_idx} due to data misalignment after factor alignment.")
                continue

            # Calculate predicted returns and residuals for this period
            predicted_period_asset_returns = final_exposures.dot(final_factor_returns)
            residuals_for_period = aligned_asset_returns - predicted_period_asset_returns

            for asset, residual_value in residuals_for_period.items():
                if pd.notna(residual_value): # Only store non-NaN residuals
                    all_residuals_dict[asset].append(residual_value)

        # Calculate variance of residuals for each asset
        idiosyncratic_variances_dict = {}
        for asset, res_list in all_residuals_dict.items():
            if len(res_list) >= 2: # Need at least 2 data points to calculate sample variance
                idiosyncratic_variances_dict[asset] = pd.Series(res_list).var(ddof=1)
            else:
                idiosyncratic_variances_dict[asset] = float('nan') # Not enough data

        if not idiosyncratic_variances_dict:
            print("Warning: No idiosyncratic variances could be calculated.")
            return None

        idiosyncratic_variances_series = pd.Series(idiosyncratic_variances_dict)
        return pd.DataFrame(idiosyncratic_variances_series, columns=['idiosyncratic_variance'])
