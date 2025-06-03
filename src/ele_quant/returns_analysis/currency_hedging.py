import pandas as pd
from typing import Optional # Added for type hinting

def calculate_hedged_returns(
    asset_returns_local_currency: pd.DataFrame, # Time x Assets
    fx_spot_returns_local_per_base: pd.DataFrame, # Time x Currencies (or single currency Series)
    hedge_ratio: float = 1.0,
    cost_of_hedge_per_period: float = 0.0,
    asset_to_currency_map: Optional[dict] = None # Maps asset col to fx_spot_returns_local_per_base col
) -> pd.DataFrame:
    """
    Calculates currency-hedged returns.
    This formula approximates the return of assets held in foreign currency, hedged back to a base currency.
    R_hedged_base_currency approx R_asset_local_currency - hedge_ratio * R_fx_spot_local_per_base - cost.

    Parameters
    ----------
    asset_returns_local_currency : pd.DataFrame
        DataFrame of asset returns in their respective local currencies. Index is time, columns are assets.
    fx_spot_returns_local_per_base : pd.DataFrame or pd.Series
        DataFrame or Series of spot FX returns. A positive return means the local currency appreciated against the base currency.
        If DataFrame, index is time, columns are currency codes (e.g., 'EUR', 'JPY' if base is USD).
        If Series, index is time, assumes all assets are denominated in this one foreign currency.
    hedge_ratio : float, optional
        The proportion of the currency exposure to hedge (0.0 to 1.0), by default 1.0.
    cost_of_hedge_per_period : float, optional
        Cost of hedging per period (e.g., basis points per period), by default 0.0.
        Ensure this matches the frequency of returns.
    asset_to_currency_map : Optional[dict], optional
        If `fx_spot_returns_local_per_base` is a DataFrame with multiple currencies,
        this dictionary maps asset column names from `asset_returns_local_currency`
        to currency column names in `fx_spot_returns_local_per_base`.
        If None, and `fx_spot_returns_local_per_base` is a DataFrame, it's assumed that asset columns
        and currency columns correspond directly or an error is raised if ambiguous.
        If `fx_spot_returns_local_per_base` is a Series, this map is ignored.

    Returns
    -------
    pd.DataFrame
        DataFrame of hedged returns in the base currency.

    Raises
    ------
    ValueError
        If inputs are misconfigured (e.g., map needed but not provided).
    """
    if not isinstance(asset_returns_local_currency, pd.DataFrame):
        raise ValueError("asset_returns_local_currency must be a pandas DataFrame.")
    if not (isinstance(fx_spot_returns_local_per_base, pd.DataFrame) or \
            isinstance(fx_spot_returns_local_per_base, pd.Series)):
        raise ValueError("fx_spot_returns_local_per_base must be a pandas DataFrame or Series.")

    hedged_returns_df = pd.DataFrame(index=asset_returns_local_currency.index)


    for asset_col in asset_returns_local_currency.columns:
        asset_series = asset_returns_local_currency[asset_col]

        current_fx_returns = None
        if isinstance(fx_spot_returns_local_per_base, pd.Series):
            current_fx_returns = fx_spot_returns_local_per_base
        else: # DataFrame
            if asset_to_currency_map:
                currency_col = asset_to_currency_map.get(asset_col)
                if not currency_col or currency_col not in fx_spot_returns_local_per_base.columns:
                    raise ValueError(f"Currency for asset {asset_col} (mapped to {currency_col}) not found in fx_spot_returns_local_per_base columns: {fx_spot_returns_local_per_base.columns.tolist()}.")
                current_fx_returns = fx_spot_returns_local_per_base[currency_col]
            elif asset_col in fx_spot_returns_local_per_base.columns: # Direct match
                current_fx_returns = fx_spot_returns_local_per_base[asset_col]
            elif fx_spot_returns_local_per_base.shape[1] == 1: # Single currency in DataFrame
                 current_fx_returns = fx_spot_returns_local_per_base.iloc[:, 0]
            else:
                raise ValueError(f"FX returns for asset {asset_col} are ambiguous. fx_df columns: {fx_spot_returns_local_per_base.columns.tolist()}. Provide asset_to_currency_map or ensure direct column name match if fx_df has multiple columns.")

        # Align indices before operation
        aligned_asset_returns, aligned_fx_returns = asset_series.align(current_fx_returns, join='left')

        # If join='left', aligned_fx_returns might have NaNs where current_fx_returns doesn't have index from asset_series
        # We want to preserve the asset_series length and apply fx adjustment where possible
        # The formula is R_local - HR * R_fx - cost. If R_fx is NaN for a period, the result should be NaN for that period.

        hedged_asset_series = aligned_asset_returns - hedge_ratio * aligned_fx_returns - cost_of_hedge_per_period

        hedged_returns_df[asset_col] = hedged_asset_series

    return hedged_returns_df

def calculate_forward_premium(
    spot_rates: pd.Series,
    forward_rates: pd.Series
) -> Optional[pd.Series]:
    """
    Calculates the forward premium: (Forward - Spot) / Spot.

    Parameters
    ----------
    spot_rates : pd.Series
        Series of spot FX rates (time index).
    forward_rates : pd.Series
        Series of forward FX rates for a specific tenor (time index).

    Returns
    -------
    Optional[pd.Series]
        Series containing the forward premium. Returns None if alignment fails or inputs invalid.
    """
    if not isinstance(spot_rates, pd.Series) or not isinstance(forward_rates, pd.Series):
        print("Warning: Inputs must be pandas Series for calculate_forward_premium.")
        return None

    aligned_spot, aligned_forward = spot_rates.align(forward_rates, join='inner')

    if aligned_spot.empty:
        print("Warning: Spot and forward rates have no common dates for alignment in calculate_forward_premium.")
        return None

    # Avoid division by zero if spot rate is zero, though unlikely for FX rates
    # Replace 0 with pd.NA to result in pd.NA for that calculation.
    aligned_spot_no_zero = aligned_spot.replace(0, pd.NA)

    forward_premium = (aligned_forward - aligned_spot_no_zero) / aligned_spot_no_zero

    # If all results are NaN (e.g. all spot rates were zero, or inputs were misaligned to produce all NaNs)
    if forward_premium.isnull().all():
        print("Warning: Forward premium calculation resulted in all NaNs.")
        return forward_premium # Or return None, depending on desired behavior

    return forward_premium # .dropna() was removed to keep NaNs from zero division
