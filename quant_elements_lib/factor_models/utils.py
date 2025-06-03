import pandas as pd
import numpy as np
from typing import Tuple

def decompose_alpha(
    alpha_total: pd.Series,
    B_loadings: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Decomposes the total alpha into components spanned by and orthogonal to factor loadings.

    The decomposition is based on projecting alpha_total onto the space spanned by
    the columns of B_loadings.

    Args:
        alpha_total (pd.Series): A Pandas Series representing the total alpha values
                                 for N assets (N x 1). The index should represent
                                 asset identifiers.
        B_loadings (pd.DataFrame): A Pandas DataFrame representing the factor loadings
                                   (betas) for N assets and K factors (N x K).
                                   The index should represent asset identifiers and
                                   match those in alpha_total. Columns represent
                                   factor identifiers.

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: A tuple containing:
            - alpha_spanned (pd.Series): The component of alpha that is spanned by the
                                         factor loadings (N x 1). Index matches alpha_total.
            - alpha_orthogonal (pd.Series): The component of alpha that is orthogonal
                                            to the factor loadings (N x 1). Index
                                            matches alpha_total.
            - lambda_spanned (pd.Series): The factor exposures that generate alpha_spanned
                                          (K x 1). Index matches B_loadings.columns.
                                          lambda_spanned = (B'B)⁻¹B'α_total
                                          alpha_spanned = B @ lambda_spanned
                                          alpha_orthogonal = alpha_total - alpha_spanned
    Raises:
        ValueError: If the asset identifiers (index) of alpha_total and B_loadings
                    do not align.
    """
    if not alpha_total.index.equals(B_loadings.index):
        raise ValueError("Asset identifiers (index) of alpha_total and B_loadings must align.")

    B_values = B_loadings.values  # N x K
    alpha_total_values = alpha_total.values # N

    # Calculate lambda_spanned = (B'B)⁻¹B'α_total
    # (B'B) is K x K. B' is K x N. alpha_total is N (or N x 1)
    # pinv is used for pseudo-inverse, which is more stable for ill-conditioned matrices.
    try:
        # (K x N) @ (N x K) -> K x K
        BtB = B_values.T @ B_values
        # (K x N) @ (N) -> K
        Bt_alpha = B_values.T @ alpha_total_values

        lambda_spanned_values = np.linalg.pinv(BtB) @ Bt_alpha # K
    except np.linalg.LinAlgError as e:
        # Handle cases where pseudo-inverse might fail, though pinv is robust
        raise RuntimeError(f"Linear algebra error during lambda_spanned calculation: {e}")

    lambda_spanned = pd.Series(lambda_spanned_values, index=B_loadings.columns, name="lambda_spanned")

    # Calculate alpha_spanned = B @ lambda_spanned
    # (N x K) @ (K) -> N
    alpha_spanned_values = B_values @ lambda_spanned_values
    alpha_spanned = pd.Series(alpha_spanned_values, index=alpha_total.index, name="alpha_spanned")

    # Calculate alpha_orthogonal = alpha_total - alpha_spanned
    # (N) - (N) -> N
    alpha_orthogonal_values = alpha_total_values - alpha_spanned_values
    alpha_orthogonal = pd.Series(alpha_orthogonal_values, index=alpha_total.index, name="alpha_orthogonal")

    return alpha_spanned, alpha_orthogonal, lambda_spanned
