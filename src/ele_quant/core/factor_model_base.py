from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Optional

class FactorModelBase(ABC):
    """
    Abstract base class for factor models.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initializes the FactorModelBase.

        Args:
            model_name: Optional name for the model instance.
        """
        self.model_name = model_name

    @abstractmethod
    def fit(self, returns_data: pd.DataFrame, **kwargs: Any) -> None:
        """
        Fits the factor model to the provided returns data.

        Args:
            returns_data: A pandas DataFrame of asset returns.
            **kwargs: Additional arguments for the fitting process.
        """
        raise NotImplementedError

    @abstractmethod
    def get_loadings(self) -> pd.DataFrame:
        """
        Returns the factor loadings (betas) for each asset.

        Returns:
            A pandas DataFrame of factor loadings.
        """
        raise NotImplementedError

    @abstractmethod
    def get_factor_covariance_matrix(self) -> pd.DataFrame:
        """
        Returns the covariance matrix of the factors.

        Returns:
            A pandas DataFrame representing the factor covariance matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def get_idiosyncratic_covariance_matrix(self) -> pd.DataFrame:
        """
        Returns the covariance matrix of the idiosyncratic (specific) returns.

        Returns:
            A pandas DataFrame representing the idiosyncratic covariance matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_total_asset_covariance_matrix(self) -> pd.DataFrame:
        """
        Calculates the total covariance matrix of asset returns based on the factor model.
        Total Covariance = Factor Covariance + Idiosyncratic Covariance

        Returns:
            A pandas DataFrame representing the total asset covariance matrix.
        """
        raise NotImplementedError
