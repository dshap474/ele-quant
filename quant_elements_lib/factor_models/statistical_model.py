import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union

from quant_elements_lib.core.factor_model_base import FactorModelBase
# Placeholder for imports that might be needed later for fit methods
# from quant_elements_lib.utils.linalg import some_linalg_function_if_needed
# from quant_elements_lib.utils.matrix_ops import some_matrix_op_function_if_needed

class StatisticalFactorModel(FactorModelBase):
    """
    Statistical Factor Model based on PCA, PPCA, or other statistical methods.
    """
    def __init__(self,
                 asset_universe: List[str],
                 num_factors_to_extract: int,
                 model_name: str = "StatisticalFactorModel"):
        """
        Initialize the StatisticalFactorModel.

        Args:
            asset_universe (List[str]): List of asset identifiers.
            num_factors_to_extract (int): The number of factors (m) to extract.
            model_name (str): Name of the model instance.
        """
        super().__init__(asset_universe=asset_universe, factor_names=[f"Factor_{i+1}" for i in range(num_factors_to_extract)], model_name=model_name)
        self.num_factors_to_extract = num_factors_to_extract
        # Attributes to be populated by the fit method
        # self.B_loadings: Optional[pd.DataFrame] = None (N x K) inherited
        # self.factor_returns: Optional[pd.DataFrame] = None (T x K) inherited
        # self.factor_covariance: Optional[pd.DataFrame] = None (K x K) inherited
        # self.idiosyncratic_covariance: Optional[Union[pd.DataFrame, pd.Series]] = None (N x N or N x 1) inherited
        # self.alpha: Optional[pd.Series] = None (N x 1) inherited

    def fit(self,
            returns_data: pd.DataFrame,
            estimation_method: str = 'PCA',
            pca_options: Optional[Dict] = None,
            ppca_options: Optional[Dict] = None,
            twostagepca_options: Optional[Dict] = None) -> None:
        """
        Fit the statistical factor model to the returns data.

        Args:
            returns_data (pd.DataFrame): DataFrame of asset returns (T x N),
                                         with assets as columns and time as index.
            estimation_method (str): The method to use for estimation.
                                     Supported: 'PCA', 'PPCA', 'TwoStagePCA'.
            pca_options (Optional[Dict]): Dictionary of options for PCA.
                Relevant keys:
                'idio_vol_weighted' (bool): If True, apply idiosyncratic volatility re-weighting. (Not implemented yet)
                'time_weighted_hl' (float): Half-life for time-weighting returns. (Not implemented yet)
                'eigenvalue_shrinkage_func' (callable): Function to shrink eigenvalues of factor covariance. (Not implemented yet)
            ppca_options (Optional[Dict]): Dictionary of options for PPCA.
                Relevant keys: None specific yet.
            twostagepca_options (Optional[Dict]): Dictionary of options for TwoStagePCA. (Not implemented yet)

        Raises:
            ValueError: If an unsupported estimation_method is provided.
            NotImplementedError: If a method or option is not yet implemented.
        """
        if returns_data.shape[1] != len(self.asset_universe):
            raise ValueError(f"Number of assets in returns_data ({returns_data.shape[1]}) "
                             f"does not match asset_universe size ({len(self.asset_universe)}).")
        if not all(asset in returns_data.columns for asset in self.asset_universe):
            raise ValueError("Columns in returns_data must match asset_universe.")

        self.returns_data = returns_data.copy() # T x N

        if estimation_method == 'PCA':
            # 1. Preprocess returns (demean)
            # returns_data is T x N
            self.alpha = self.returns_data.mean(axis=0) # Store mean as alpha (N x 1)
            returns_data_demeaned = self.returns_data - self.alpha # T x N

            # Handle cases where T < N for SVD
            # If T (n_samples) < N (n_features), SVD on R.T is more efficient for U.
            # numpy.linalg.svd handles this fine, but good to be aware.
            # U will be T x K, s will be K, Vh will be K x N where K = min(T,N)

            try:
                U, s, Vh = np.linalg.svd(returns_data_demeaned.values, full_matrices=False)
                # U: (T x K) matrix (orthonormal columns)
                # s: (K,) vector of singular values (sorted in descending order)
                # Vh: (K x N) matrix (orthonormal rows) (V transpose)
            except np.linalg.LinAlgError as e:
                raise RuntimeError(f"SVD computation failed: {e}")

            if self.num_factors_to_extract > len(s):
                raise ValueError(
                    f"num_factors_to_extract ({self.num_factors_to_extract}) "
                    f"cannot be greater than the number of available singular values ({len(s)})."
                )

            # 4. Extract B_loadings (N x K)
            # B_loadings are the first K principal components (eigenvectors of R.T @ R)
            # Vh.T gives V. V[:, :K] are the principal axes.
            self.B_loadings = pd.DataFrame(Vh.T[:, :self.num_factors_to_extract],
                                           index=self.asset_universe,
                                           columns=self.get_factor_names())

            # 5. Estimate factor_returns (T x K)
            # factor_returns = R_demeaned @ B_loadings = U @ diag(s) @ Vh @ Vh.T[:,:K]
            # = U @ diag(s) @ I[:,:K] = U[:,:K] @ diag(s[:K])
            self.factor_returns = pd.DataFrame(U[:, :self.num_factors_to_extract] @ np.diag(s[:self.num_factors_to_extract]),
                                               index=self.returns_data.index,
                                               columns=self.get_factor_names())

            # 6. Estimate factor_covariance (K x K)
            # Covariance of F = U S. So F.T F / (T-1) = S U.T U S / (T-1) = S^2 / (T-1)
            # These are principal component variances.
            # Using (T-1) for sample covariance, or T if population. Let's use T-1.
            T = returns_data_demeaned.shape[0]
            if T <= 1:
                # Handle case with insufficient data for variance calculation
                # Or let np.diag fail if s is too short, but num_factors_to_extract check should prevent
                self.factor_covariance = pd.DataFrame(np.eye(self.num_factors_to_extract) * np.nan,
                                                      index=self.get_factor_names(),
                                                      columns=self.get_factor_names())

            else:
                factor_variances = (s[:self.num_factors_to_extract]**2) / (T - 1)
                self.factor_covariance = pd.DataFrame(np.diag(factor_variances),
                                                      index=self.get_factor_names(),
                                                      columns=self.get_factor_names())

            # 7. Estimate idiosyncratic_covariance (N x 1 Series)
            # Residuals: epsilon = R_demeaned - F @ B.T
            # Note: R_demeaned = U @ np.diag(s) @ Vh
            # F @ B.T = (U[:,:K] @ np.diag(s[:K])) @ (Vh.T[:,:K]).T
            #         = U[:,:K] @ np.diag(s[:K]) @ Vh[:K,:]
            # This is the rank-K reconstruction of R_demeaned.
            reconstructed_returns = pd.DataFrame(
                self.factor_returns.values @ self.B_loadings.values.T,
                index=returns_data_demeaned.index,
                columns=returns_data_demeaned.columns
            )
            residuals = returns_data_demeaned - reconstructed_returns # T x N

            if T <= 1:
                self.idiosyncratic_covariance = pd.Series(np.nan, index=self.asset_universe)
            else:
                self.idiosyncratic_covariance = residuals.var(axis=0, ddof=1) # N x 1 Series

            # Ensure idiosyncratic covariance is non-negative
            if self.idiosyncratic_covariance is not None:
                 self.idiosyncratic_covariance = self.idiosyncratic_covariance.clip(lower=0)

            # Alpha was set to the mean of returns earlier.
            # If a model should be strictly no-alpha from PCA, self.alpha can be set to zeros.
            # For now, using mean of returns as alpha is a common approach.

        elif estimation_method == 'PPCA':
            # 1. Preprocess returns (demean)
            self.alpha = self.returns_data.mean(axis=0) # N x 1
            returns_data_demeaned = self.returns_data - self.alpha # T x N

            T, N = returns_data_demeaned.shape

            if self.num_factors_to_extract >= N:
                raise ValueError(
                    f"num_factors_to_extract ({self.num_factors_to_extract}) "
                    f"must be less than the number of assets ({N}) for PPCA."
                )
            if T <= 1:
                 raise ValueError("Insufficient data (T <= 1) for PPCA covariance estimation.")

            # 2. Calculate sample covariance matrix Sigma_sample (N x N)
            # Using (T-1) for sample covariance
            Sigma_sample = np.cov(returns_data_demeaned.values, rowvar=False, ddof=1)

            # 3. Perform eigen-decomposition of Sigma_sample
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(Sigma_sample)
                # eigh returns eigenvalues in ascending order. We need descending.
                sorted_indices = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[sorted_indices]
                eigenvectors = eigenvectors[:, sorted_indices]
            except np.linalg.LinAlgError as e:
                raise RuntimeError(f"Eigen-decomposition of sample covariance failed: {e}")

            # 4. Estimate sigma_sq_hat
            # Mean of the smallest N-m eigenvalues
            if self.num_factors_to_extract < N:
                sigma_sq_hat = np.mean(eigenvalues[self.num_factors_to_extract:])
            else: # Should be caught by earlier check, but as a fallback
                sigma_sq_hat = 0.0

            sigma_sq_hat = max(sigma_sq_hat, 1e-9) # Ensure positivity and avoid issues

            # 5. Calculate B_loadings (N x K) using Eq. 7.15
            # B = U_m (Lambda_m - sigma_sq_hat * I_m)^(1/2)
            # eigenvectors are U. U_m is eigenvectors[:, :m]
            # Lambda_m is diag(eigenvalues[:m])
            U_m = eigenvectors[:, :self.num_factors_to_extract] # N x K
            Lambda_m_diag = eigenvalues[:self.num_factors_to_extract] # K,

            # Term inside sqrt: (Lambda_m - sigma_sq_hat * I_m)
            # Ensure this is positive before sqrt
            term_in_sqrt_diag = np.maximum(0, Lambda_m_diag - sigma_sq_hat)

            # (Lambda_m - sigma_sq_hat * I_m)^(1/2)
            sqrt_term_matrix = np.diag(np.sqrt(term_in_sqrt_diag)) # K x K

            B_values = U_m @ sqrt_term_matrix # (N x K) @ (K x K) = N x K
            self.B_loadings = pd.DataFrame(B_values,
                                           index=self.asset_universe,
                                           columns=self.get_factor_names())

            # 6. Set factor_covariance (K x K)
            # PPCA assumes factors are N(0,I)
            self.factor_covariance = pd.DataFrame(np.eye(self.num_factors_to_extract),
                                                  index=self.get_factor_names(),
                                                  columns=self.get_factor_names())

            # 7. Set idiosyncratic_covariance (N x 1 Series or N x N diag)
            # Ω_ε = σ²I
            self.idiosyncratic_covariance = pd.Series([sigma_sq_hat] * N, index=self.asset_universe)

            # 8. Estimate factor_returns (T x K)
            # F = R_demeaned @ B @ (B.T @ B + sigma_sq_hat * I_K)^-1
            # This is E[f|r] = M^-1 B.T (r - alpha) where M = B.T B + sigma_sq_hat I_K
            # M is K x K
            B_T_B = B_values.T @ B_values # K x K
            M = B_T_B + sigma_sq_hat * np.eye(self.num_factors_to_extract) # K x K

            try:
                M_inv = np.linalg.inv(M) # K x K
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if M is singular, though less likely with sigma_sq_hat > 0
                M_inv = np.linalg.pinv(M)

            # (r - alpha) is returns_data_demeaned (T x N)
            # B.T is K x N
            # M_inv @ B.T is (K x K) @ (K x N) = K x N
            # (r-alpha) @ B @ M_inv.T  -> (T x N) @ (N x K) @ (K x K) = T x K
            # factor_returns_values = returns_data_demeaned.values @ B_values @ M_inv
            # or using E[f|r].T = (r-alpha).T @ B @ M_inv (Bishop notation)
            # (r-alpha) is (N x T) matrix, B is (N x K)
            # B.T @ (r-alpha) is (K x T)
            # M_inv @ B.T @ (r-alpha) is (K x T)
            # So, factor_returns.T = M_inv @ B_values.T @ returns_data_demeaned.values.T
            # factor_returns = (M_inv @ B_values.T @ returns_data_demeaned.values.T).T

            factor_returns_values = returns_data_demeaned.values @ B_values @ M_inv

            self.factor_returns = pd.DataFrame(factor_returns_values,
                                               index=self.returns_data.index,
                                               columns=self.get_factor_names())

        elif estimation_method == 'TwoStagePCA':
            # Procedure 7.1 from the book
            raise NotImplementedError("TwoStagePCA estimation method is not yet implemented.")

        else:
            raise ValueError(f"Unsupported estimation_method: {estimation_method}. "
                             "Supported methods are 'PCA', 'PPCA', 'TwoStagePCA'.")

        # Common post-fit steps (e.g., setting fitted flag)
        self.is_fitted = True

    def get_factor_names(self) -> List[str]:
        return [f"Factor_{i+1}" for i in range(self.num_factors_to_extract)]
