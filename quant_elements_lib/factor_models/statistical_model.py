from typing import List, Optional

import numpy as np
import pandas as pd

from quant_elements_lib.core.factor_model_base import FactorModelBase


class StatisticalFactorModel(FactorModelBase):
    """
    A statistical factor model that uses methods like PCA or PPCA to extract factors.
    """

    def __init__(self, asset_universe: List[str], num_factors_to_extract: int):
        """
        Initializes the StatisticalFactorModel.

        Args:
            asset_universe: A list of asset identifiers that this model will cover.
                            These names must align with the column names in `returns_data` passed to `fit`.
            num_factors_to_extract: The number of factors to extract.
        """
        # Store asset_universe and factor_names for later use, particularly for creating
        # DataFrames with correct indices/columns after fitting.
        self.asset_universe = asset_universe # List of N asset names
        self.num_factors_to_extract = num_factors_to_extract
        self.factor_names = [f'Factor_{i+1}' for i in range(num_factors_to_extract)] # List of K factor names

        # Initialize components to None. They will be populated by the fit method.
        super().__init__(
            alpha=None,
            B_loadings=None,
            factor_covariance=None,
            idiosyncratic_covariance=None
        )

    def fit(
        self,
        returns_data: pd.DataFrame,
        estimation_method: str = 'PCA',
        pca_options: Optional[dict] = None,
        ppca_options: Optional[dict] = None
    ) -> None:
        """
        Fits the statistical factor model to the given returns data.

        Args:
            returns_data: A DataFrame of asset returns (T x N, where T is number of observations, N is number of assets).
            estimation_method: The estimation method to use ('PCA' or 'PPCA').
            pca_options: Optional dictionary of PCA-specific options.
                         Currently unused, but can include options for weighted PCA or other variations.
            ppca_options: Optional dictionary of PPCA-specific options.
                          Currently unused, but can include options for PPCA variations.
        """
        if estimation_method == 'PCA':
            # 1. Preprocess returns (demean)
            returns_demeaned = returns_data - returns_data.mean(axis=0) # T x N

            # 2. Perform SVD on demeaned returns (R = U S V^T)
            # U will be T x T, s_diag will be T, Vt will be T x N if full_matrices=True
            # U will be T x N, s_diag will be N, Vt will be N x N if full_matrices=False and T > N
            # U will be T x T, s_diag will be T, Vt will be T x N if full_matrices=False and T < N
            # We work with R (T x N), so U (T x K), S (K x K), V (N x K) where K = min(T,N)
            # np.linalg.svd returns U, s_diag, Vt
            # U: (T, K) unitary matrix having left singular vectors as columns.
            # s_diag: (K,) singular values, sorted in descending order.
            # Vt: (K, N) unitary matrix having right singular vectors as rows.

            # If returns_demeaned is T x N (observations x assets)
            # U will be T x min(T,N)
            # s_diag will be min(T,N)
            # Vt will be min(T,N) x N
            U, s_diag, Vt = np.linalg.svd(returns_demeaned.values, full_matrices=False)

            # Number of factors to extract
            m = self.num_factors_to_extract

            # 3. Extract B_loadings (Factor Exposures)
            # B_loadings are the first m columns of V (the right singular vectors, which are eigenvectors of R^T R)
            # Or, if we follow common convention R = F B^T + e, where R is N x T, F is N x K, B is T x K
            # In our case, returns_demeaned is T x N.
            # R (T x N) = U (T x m) @ S_m (m x m) @ Vt_m (m x N)
            # Factors F are U_m S_m (T x m) or U_m (T x m)
            # Loadings B are V_m (N x m)
            # Let's use the convention from the textbook (page 278, Eq 7.21, 7.22)
            # R = U S V^T. Factors F = U_m S_m or F = sqrt(T) * U_m. Loadings B = V_m.
            # Here, returns_data is T x N. `svd` on T x N data gives U (T x K), S (K), V (N x K) [where V is Vt.T]
            # B_loadings (asset factor exposures) should be N x m. These are the first m principal components of the covariance matrix of returns.
            # These are given by the first m columns of Vt.T (i.e., V).
            B_loadings_arr = Vt.T[:, :m]  # N x m
            self.B_loadings = pd.DataFrame(B_loadings_arr, index=returns_data.columns, columns=self.factor_names)

            # 4. Estimate factor_returns
            # F_hat = R V_m = U S V^T V_m = U_m S_m (T x m)
            # Or F_hat = sqrt(T) * U_m
            # Let's use F = R B = T x N @ N x m = T x m
            # factor_returns_arr = returns_demeaned.values @ B_loadings_arr # T x m
            # Alternative: F = U_m S_m
            U_m = U[:, :m] # T x m
            S_m_diag = s_diag[:m] # m
            # factor_returns_arr = U_m @ np.diag(S_m_diag) # T x m

            # Using F = U S_m V_m^T V_m = U_m S_m (T x m) from returns_demeaned = U S V^T
            # The principal components (factor realizations) are often taken as U_m * S_m_diag or simply U_m scaled.
            # Let factor_returns F = R V. Here R is T x N, V is N x m. So F is T x m.
            factor_returns_arr = returns_demeaned.values @ self.B_loadings.values # T x m
            self.factor_returns = pd.DataFrame(factor_returns_arr, index=returns_data.index, columns=self.factor_names)

            # 5. Estimate factor_covariance
            # Cov(F) = Cov(R V_m). If R is demeaned, Cov(F) = (1/T) F^T F
            # Cov(F) = (1/T) (V_m^T R^T R V_m) = V_m^T ( (1/T) R^T R ) V_m = V_m^T Sigma_R V_m
            # Since V_m are eigenvectors of Sigma_R, V_m^T Sigma_R V_m = Lambda_m (diagonal matrix of m largest eigenvalues)
            # The eigenvalues of (1/T)R^T R are s_diag^2 / T.
            factor_cov_arr = np.diag(s_diag[:m]**2) / returns_data.shape[0] # m x m
            self.factor_covariance = pd.DataFrame(factor_cov_arr, index=self.factor_names, columns=self.factor_names)

            # 6. Calculate residuals
            # R_demeaned (T x N) = F (T x m) @ B^T (m x N) + E (T x N)
            # E = R_demeaned - F @ B^T
            reconstructed_systematic_returns = self.factor_returns @ self.B_loadings.T # T x N
            residuals_arr = returns_demeaned.values - reconstructed_systematic_returns.values # T x N

            # Store residuals as a DataFrame for consistency, though not directly part of the model parameters usually exposed.
            # self.residuals = pd.DataFrame(residuals_arr, index=returns_data.index, columns=returns_data.columns)

            # 7. Estimate idiosyncratic_covariance (diagonal)
            # This is the variance of each column of E (residuals_arr)
            idio_variances = np.var(residuals_arr, axis=0) # N
            self.idiosyncratic_covariance = pd.Series(idio_variances, index=returns_data.columns, name="idiosyncratic_variance")

            # 8. Estimate alpha (intercept term)
            # Since we demeaned the returns, the model is R_i - mu_i = sum(B_ik * F_k) + e_i
            # So, R_i = mu_i + sum(B_ik * F_k) + e_i
            # Alpha here would represent the portion of mean returns not explained by factor risk premia if factors were priced.
            # For statistical models, alpha is often set to zero or is the mean of residuals.
            # If returns were not demeaned initially, alpha would be R.mean() - B @ F.mean().
            # Given we used demeaned returns, the mean of residuals_arr should be close to zero.
            # For consistency with other models, alpha can be set to the mean of residuals.
            # However, typically for PCA on demeaned returns, alpha is considered zero.
            # Let's calculate it as the mean of residuals, which should be near zero.
            alphas = residuals_arr.mean(axis=0) # N
            self.alpha = pd.Series(alphas, index=returns_data.columns, name="alpha")

        elif estimation_method == 'PPCA':
            # Number of assets
            N = returns_data.shape[1]
            # Number of factors
            m = self.num_factors_to_extract

            if m > N:
                raise ValueError("Number of factors to extract cannot exceed the number of assets.")

            # 1. Preprocess returns (demean)
            returns_demeaned = returns_data - returns_data.mean(axis=0) # T x N

            # 2. Calculate sample covariance matrix Sigma_sample (N x N)
            # ddof=0 for MLE covariance if assuming normal data, or ddof=1 for unbiased sample covariance.
            # Textbook typical derivation for PPCA often uses MLE for covariance.
            # Let's use ddof=1 for sample covariance, consistent with typical statistical approaches.
            Sigma_sample = np.cov(returns_demeaned.values, rowvar=False, ddof=1) # N x N

            # 3. Perform Eigen-decomposition on Sigma_sample
            eigenvalues, U_cov = np.linalg.eigh(Sigma_sample) # eigh for symmetric matrices

            # Sort eigenvalues and eigenvectors in descending order
            sorted_indices = np.argsort(eigenvalues)[::-1]
            Lambda_cov_diag_sorted = eigenvalues[sorted_indices] # Sorted eigenvalues (N)
            U_cov_sorted = U_cov[:, sorted_indices] # Sorted eigenvectors (N x N)

            # 4. Estimate sigma_sq_hat (noise variance)
            if m == N:
                # If num_factors equals num_assets, PPCA implies sigma_sq_hat could be zero
                # or it's underdetermined/degenerate. Set to a small epsilon.
                sigma_sq_hat = 1e-9
            else:
                sigma_sq_hat = np.mean(Lambda_cov_diag_sorted[m:])

            if sigma_sq_hat < 0: # Should not happen with eigh if Sigma_sample is psd
                sigma_sq_hat = 1e-9

            # 5. Calculate B_loadings (N x m) using Eq. 7.15 (adjusted for U_cov_sorted)
            # B = U_m (Lambda_m - sigma_sq I_m)^(1/2) R_m, where R_m is an arbitrary m x m rotation matrix.
            # We choose R_m = I_m for simplicity.
            U_m_cov = U_cov_sorted[:, :m] # N x m (eigenvectors for top m eigenvalues)
            Lambda_m_diag = Lambda_cov_diag_sorted[:m] # Top m eigenvalues

            # Ensure term inside sqrt is non-negative
            term_in_sqrt_diag = Lambda_m_diag - sigma_sq_hat
            term_in_sqrt_diag[term_in_sqrt_diag < 0] = 0 # Clamp negative values to 0

            sqrt_matrix = np.diag(np.sqrt(term_in_sqrt_diag)) # m x m

            B_loadings_arr = U_m_cov @ sqrt_matrix # N x m
            self.B_loadings = pd.DataFrame(B_loadings_arr, index=returns_data.columns, columns=self.factor_names)

            # 6. Set factor_covariance (m x m) = I_m for PPCA
            factor_cov_arr = np.eye(m)
            self.factor_covariance = pd.DataFrame(factor_cov_arr, index=self.factor_names, columns=self.factor_names)

            # 7. Set idiosyncratic_covariance (N, diagonal)
            idio_variances = np.full(N, sigma_sq_hat)
            self.idiosyncratic_covariance = pd.Series(idio_variances, index=returns_data.columns, name="idiosyncratic_variance")

            # 8. Estimate factor_returns (T x m)
            # F = R_demeaned @ B @ (B^T B + sigma_sq_hat * I_m)^-1  -- this is one form for E-step in EM
            # A simpler approach often used post-estimation of B and sigma_sq:
            # F = R_demeaned @ B_pinv, where B_pinv is (B^T B)^-1 B^T or similar projection.
            # Or, more directly using the result from Bishop (PRML, Eq. 12.44) for x_n (our returns_demeaned.T):
            # E[z_n] = M^-1 W^T (x_n - mu), where W is B_loadings, M = W^T W + sigma_sq_hat I
            # So factor_returns.T = (B^T B + sigma_sq_hat I_m)^-1 B^T returns_demeaned.T
            # factor_returns = returns_demeaned @ B @ inv(B^T B + sigma_sq_hat I_m)
            M_inv = np.linalg.inv(B_loadings_arr.T @ B_loadings_arr + sigma_sq_hat * np.eye(m)) # m x m
            factor_returns_arr = returns_demeaned.values @ B_loadings_arr @ M_inv # T x N @ N x m @ m x m = T x m
            self.factor_returns = pd.DataFrame(factor_returns_arr, index=returns_data.index, columns=self.factor_names)

            # 9. Estimate alpha (N)
            # PPCA formulation usually assumes data is centered (demeaned) for model fitting.
            # The 'alpha' in a factor model context (R = alpha + B F + e) would be the original data mean
            # if factors F are defined to have zero mean.
            alphas = returns_data.mean(axis=0).values # N
            self.alpha = pd.Series(alphas, index=returns_data.columns, name="alpha")

            # Stored components:
            # self.alpha: pd.Series (N) - original means
            # self.B_loadings: pd.DataFrame (N x m)
            # self.factor_returns: pd.DataFrame (T x m) - estimated latent factor values
            # self.factor_covariance: pd.DataFrame (m x m) - Identity for standard PPCA
            # self.idiosyncratic_covariance: pd.Series (N) - sigma_sq_hat for all assets

        elif estimation_method == 'TwoStagePCA':
            # Steps for TwoStagePCA (Procedure 7.1 - for future implementation):
            # 1. Initial PCA on R to get F^(1).
            # 2. Estimate idio vol σ_εi^(1) from residuals.
            # 3. Reweight returns R* = R / σ_εi^(1).
            # 4. Second PCA on R* to get F^(2).
            # 5. Estimate final B and σ_εi.
            # 6. Store components.
            pass # Placeholder for TwoStagePCA implementation
        else:
            raise ValueError(f"Unsupported estimation method: {estimation_method}")

        # After estimation, the following components should be populated:
        # self.alpha: pd.Series
        # self.B_loadings: pd.DataFrame
        # self.factor_returns: pd.DataFrame
        # self.factor_covariance: pd.DataFrame
        # self.idiosyncratic_covariance: pd.DataFrame (or pd.Series if diagonal)
        return None
