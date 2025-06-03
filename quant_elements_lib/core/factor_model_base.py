import abc
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union


class FactorModelBase(abc.ABC):
    """
    Base class for factor models.

    This class provides a foundational structure for various factor models,
    including methods for calculating asset covariance matrices, predicting
    systematic returns, and decomposing total returns.

    Attributes:
        alpha (Optional[pd.Series]): Series of alpha values for each asset.
                                      Index should match asset identifiers.
        B_loadings (Optional[pd.DataFrame]): DataFrame of factor loadings (beta values).
                                             Shape N x K (N assets, K factors).
                                             Index should match asset identifiers,
                                             columns should match factor identifiers.
        factor_covariance (Optional[pd.DataFrame]): Covariance matrix of the factors.
                                                    Shape K x K. Index and columns
                                                    should match factor identifiers.
        idiosyncratic_covariance (Optional[Union[pd.DataFrame, pd.Series]]):
            Covariance matrix of idiosyncratic returns.
            If pd.Series, it's treated as a diagonal matrix (variances).
            Shape N x N (DataFrame) or N (Series). Index (and columns if DataFrame)
            should match asset identifiers.
    """

    def __init__(self,
                 alpha: Optional[pd.Series] = None,
                 B_loadings: Optional[pd.DataFrame] = None,
                 factor_covariance: Optional[pd.DataFrame] = None,
                 idiosyncratic_covariance: Optional[Union[pd.DataFrame, pd.Series]] = None):
        """
        Initializes the FactorModelBase.

        Args:
            alpha (Optional[pd.Series]): Alpha values for each asset.
            B_loadings (Optional[pd.DataFrame]): Factor loadings (beta values).
            factor_covariance (Optional[pd.DataFrame]): Covariance matrix of the factors.
            idiosyncratic_covariance (Optional[Union[pd.DataFrame, pd.Series]]):
                Covariance matrix or variances of idiosyncratic returns.
        """
        self.alpha = alpha
        self.B_loadings = B_loadings
        self.factor_covariance = factor_covariance
        self.idiosyncratic_covariance = idiosyncratic_covariance

    def calculate_total_asset_covariance_matrix(self) -> Optional[pd.DataFrame]:
        """
        Calculates the total asset covariance matrix.

        The formula is: Σ = B @ Ω_f @ B.T + Ω_ε
        where:
            B is the factor loadings matrix (N x K)
            Ω_f is the factor covariance matrix (K x K)
            Ω_ε is the idiosyncratic covariance matrix (N x N)

        Returns:
            Optional[pd.DataFrame]: The total asset covariance matrix (N x N),
                                    or None if essential components are missing.
        """
        if self.B_loadings is None or self.factor_covariance is None or \
           self.idiosyncratic_covariance is None:
            return None

        B = self.B_loadings
        omega_f = self.factor_covariance

        systematic_component = B @ omega_f @ B.T

        if isinstance(self.idiosyncratic_covariance, pd.Series):
            # Assume Series contains variances, form a diagonal matrix
            omega_epsilon = pd.DataFrame(np.diag(self.idiosyncratic_covariance),
                                         index=self.idiosyncratic_covariance.index,
                                         columns=self.idiosyncratic_covariance.index)
        elif isinstance(self.idiosyncratic_covariance, pd.DataFrame):
            omega_epsilon = self.idiosyncratic_covariance
        else:
            # Should not happen if type hints are respected, but as a safeguard
            raise TypeError("idiosyncratic_covariance must be a pandas Series or DataFrame.")

        # Ensure alignment of indices/columns for addition
        if not systematic_component.index.equals(omega_epsilon.index) or \
           not systematic_component.columns.equals(omega_epsilon.columns):
            # Attempt to reindex if assets are the same but order differs
            common_assets = systematic_component.index.intersection(omega_epsilon.index)
            if len(common_assets) == len(systematic_component.index) and \
               len(common_assets) == len(omega_epsilon.index):
                omega_epsilon = omega_epsilon.reindex(index=common_assets, columns=common_assets)
                systematic_component = systematic_component.reindex(index=common_assets, columns=common_assets)
            else:
                raise ValueError("Asset identifiers for systematic and idiosyncratic "
                                 "covariance components do not match.")

        total_covariance = systematic_component + omega_epsilon
        return total_covariance

    def predict_systematic_returns(self, factor_returns: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Predicts systematic returns based on factor returns.

        The formula is: r_systematic_t = B @ f_t
        where:
            B is the factor loadings matrix (N x K)
            f_t is the factor returns at time t (K x T or T x K, will be transposed if T x K)

        Args:
            factor_returns (pd.DataFrame): DataFrame of factor returns.
                                           Shape T x K (T time periods, K factors).
                                           Index should be time periods, columns
                                           should match factor identifiers in B_loadings.

        Returns:
            Optional[pd.DataFrame]: DataFrame of predicted systematic returns.
                                    Shape T x N (T time periods, N assets).
                                    Returns None if B_loadings is not available.
        """
        if self.B_loadings is None:
            return None

        B = self.B_loadings  # N x K
        f = factor_returns    # T x K

        # Ensure factor_returns columns align with B_loadings columns (factors)
        if not f.columns.equals(B.columns):
            common_factors = f.columns.intersection(B.columns)
            if len(common_factors) == 0:
                raise ValueError("No common factors between factor_returns and B_loadings.")
            f_aligned = f[common_factors]
            B_aligned = B[common_factors]
        else:
            f_aligned = f
            B_aligned = B

        # Calculation: (T x K_aligned) @ (K_aligned x N) -> T x N
        # We need B.T which is K_aligned x N. Then f_aligned @ B_aligned.T
        systematic_returns = f_aligned @ B_aligned.T
        return systematic_returns

    def decompose_total_returns(self,
                                total_returns: pd.DataFrame,
                                factor_returns: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Decomposes total returns into systematic and idiosyncratic components.

        Systematic component: r_systematic_t = B @ f_t
        Idiosyncratic component: ε_it = r_it - α_i - B_i' @ f_t
                                (or r_it - B_i' @ f_t if alpha is not provided)

        Args:
            total_returns (pd.DataFrame): DataFrame of total asset returns.
                                          Shape T x N (T time periods, N assets).
                                          Index should be time periods, columns
                                          should match asset identifiers.
            factor_returns (pd.DataFrame): DataFrame of factor returns.
                                           Shape T x K (T time periods, K factors).
                                           Index should be time periods, columns
                                           should match factor identifiers.

        Returns:
            Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
                A tuple containing:
                - systematic_returns (pd.DataFrame): T x N
                - idiosyncratic_returns (pd.DataFrame): T x N
                Returns None if B_loadings is not available.
        """
        if self.B_loadings is None:
            return None

        systematic_returns = self.predict_systematic_returns(factor_returns)
        if systematic_returns is None: # Should not happen if B_loadings is not None
            return None

        # Ensure total_returns and systematic_returns are aligned
        # (same time index, same asset columns)
        common_index = total_returns.index.intersection(systematic_returns.index)
        common_columns = total_returns.columns.intersection(systematic_returns.columns)

        if len(common_index) == 0 or len(common_columns) == 0:
            raise ValueError("Total returns and predicted systematic returns have no common time periods or assets.")

        total_returns_aligned = total_returns.loc[common_index, common_columns]
        systematic_returns_aligned = systematic_returns.loc[common_index, common_columns]

        idiosyncratic_returns = total_returns_aligned - systematic_returns_aligned

        if self.alpha is not None:
            # Ensure alpha series is aligned with the columns of idiosyncratic_returns
            alpha_aligned = self.alpha.reindex(idiosyncratic_returns.columns)
            # Alpha is N, idiosyncratic_returns is T x N. Broadcasting subtracts alpha row-wise.
            idiosyncratic_returns = idiosyncratic_returns - alpha_aligned

        return systematic_returns_aligned, idiosyncratic_returns
