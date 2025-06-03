import pandas as pd
import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from quant_elements_lib.core.factor_model_base import FactorModelBase


def rotate_factor_model(
    model: 'FactorModelBase',
    C_rotation_matrix: pd.DataFrame
) -> 'FactorModelBase':
    """
    Rotates a factor model using a given rotation matrix C.

    The transformation rules are:
    - Rotated factor loadings: B_rotated = B @ inv(C)
    - Rotated factor covariance: Ω_f_rotated = C @ Ω_f @ C.T
    - Rotated factor returns (if applicable): f_rotated = f @ C.T

    Alpha (α) and idiosyncratic covariance (Ω_ε) remain unchanged by rotation.

    Args:
        model (FactorModelBase): The original factor model instance.
        C_rotation_matrix (pd.DataFrame): The KxK rotation matrix (C).
            The index and columns of this DataFrame should represent the
            original factor names and new (rotated) factor names, respectively.
            If columns are not new names, original names are used for rotated factors.

    Returns:
        FactorModelBase: A new FactorModelBase instance with rotated components.

    Raises:
        ValueError: If essential components (B_loadings, factor_covariance) are
                    missing in the model, or if dimensions are incompatible.
    """
    if model.B_loadings is None or model.factor_covariance is None:
        raise ValueError("Original model must have B_loadings and factor_covariance for rotation.")

    # Ensure C_rotation_matrix is KxK where K is the number of factors
    if model.B_loadings.shape[1] != C_rotation_matrix.shape[0]:
        raise ValueError(
            f"Number of factors in B_loadings ({model.B_loadings.shape[1]}) "
            f"must match rows in C_rotation_matrix ({C_rotation_matrix.shape[0]})."
        )
    if C_rotation_matrix.shape[0] != C_rotation_matrix.shape[1]:
        raise ValueError("C_rotation_matrix must be square.")

    try:
        C_inv_values = np.linalg.inv(C_rotation_matrix.values)
    except np.linalg.LinAlgError:
        raise ValueError("C_rotation_matrix is singular and cannot be inverted.")

    # Determine new factor names from C_rotation_matrix columns
    # If C_rotation_matrix.index are original factor names and C_rotation_matrix.columns are new factor names
    new_factor_names = C_rotation_matrix.columns
    if not isinstance(new_factor_names, pd.Index) or len(new_factor_names) != C_rotation_matrix.shape[1]:
        new_factor_names = [f"RotatedFactor_{i+1}" for i in range(C_rotation_matrix.shape[1])]


    # B_rotated = B @ inv(C)
    # (N x K_orig) @ (K_orig x K_new) -> N x K_new
    B_loadings_rotated_values = model.B_loadings.values @ C_inv_values
    B_loadings_rotated = pd.DataFrame(
        B_loadings_rotated_values,
        index=model.B_loadings.index,
        columns=new_factor_names
    )

    # Ω_f_rotated = C @ Ω_f @ C.T
    # (K_new x K_orig) @ (K_orig x K_orig) @ (K_orig x K_new) -> K_new x K_new
    factor_covariance_rotated_values = (
        C_rotation_matrix.values @ model.factor_covariance.values @ C_rotation_matrix.values.T
    )
    factor_covariance_rotated = pd.DataFrame(
        factor_covariance_rotated_values,
        index=new_factor_names,
        columns=new_factor_names
    )

    # Instantiate the new model - using a dynamic approach to find the class
    # This assumes FactorModelBase or its subclass can be found.
    # A more robust way might be to pass the class itself if known.
    NewModelClass = type(model)
    rotated_model = NewModelClass(
        alpha=model.alpha, # Carried over
        B_loadings=B_loadings_rotated,
        factor_covariance=factor_covariance_rotated,
        idiosyncratic_covariance=model.idiosyncratic_covariance # Carried over
    )

    # Handle factor_returns if they exist
    # f_rotated = f @ C.T
    # (T x K_orig) @ (K_orig x K_new) -> T x K_new
    if hasattr(model, 'factor_returns') and model.factor_returns is not None:
        if model.factor_returns.shape[1] != C_rotation_matrix.shape[0]:
             raise ValueError(
                f"Number of factors in model.factor_returns ({model.factor_returns.shape[1]}) "
                f"must match the dimension of C_rotation_matrix ({C_rotation_matrix.shape[0]}) "
                f"for transforming factor returns."
            )
        factor_returns_rotated_values = model.factor_returns.values @ C_rotation_matrix.values.T
        rotated_model.factor_returns = pd.DataFrame(
            factor_returns_rotated_values,
            index=model.factor_returns.index, # Time index
            columns=new_factor_names
        )
    elif hasattr(model, 'factor_returns') and model.factor_returns is None:
        rotated_model.factor_returns = None


    return rotated_model


def project_factor_model_to_subset(
    model: 'FactorModelBase',
    factors_to_keep: List[str]
) -> 'FactorModelBase':
    """
    Projects a factor model to a subset of its original factors.

    This involves selecting the relevant factor loadings, subsetting the
    factor covariance matrix, and subsetting factor returns if they exist.
    Alpha (α) and idiosyncratic covariance (Ω_ε) remain unchanged.

    Args:
        model (FactorModelBase): The original factor model instance.
        factors_to_keep (List[str]): A list of factor names (strings) to
                                     retain in the new model. These names must
                                     exist in the original model's factor names
                                     (e.g., columns of B_loadings).

    Returns:
        FactorModelBase: A new FactorModelBase instance projected to the
                         specified subset of factors.

    Raises:
        ValueError: If essential components (B_loadings, factor_covariance) are
                    missing, or if any of the specified factors_to_keep
                    are not found in the original model.
    """
    if model.B_loadings is None or model.factor_covariance is None:
        raise ValueError("Original model must have B_loadings and factor_covariance for projection.")

    original_factor_names = model.B_loadings.columns.tolist()
    for factor_name in factors_to_keep:
        if factor_name not in original_factor_names:
            raise ValueError(
                f"Factor '{factor_name}' specified in factors_to_keep "
                "not found in the model's existing factor names."
            )
        if factor_name not in model.factor_covariance.index or \
           factor_name not in model.factor_covariance.columns:
            raise ValueError(
                f"Factor '{factor_name}' specified in factors_to_keep "
                "not found in the model's factor_covariance matrix dimensions."
            )


    # Select columns for B_loadings
    B_loadings_projected = model.B_loadings[factors_to_keep]

    # Select rows and columns for factor_covariance
    factor_covariance_projected = model.factor_covariance.loc[factors_to_keep, factors_to_keep]

    NewModelClass = type(model)
    projected_model = NewModelClass(
        alpha=model.alpha, # Carried over
        B_loadings=B_loadings_projected,
        factor_covariance=factor_covariance_projected,
        idiosyncratic_covariance=model.idiosyncratic_covariance # Carried over
    )

    # Handle factor_returns if they exist
    if hasattr(model, 'factor_returns') and model.factor_returns is not None:
        # Check if all factors_to_keep are in factor_returns columns
        missing_fr_factors = [f for f in factors_to_keep if f not in model.factor_returns.columns]
        if missing_fr_factors:
            raise ValueError(
                f"Factors {missing_fr_factors} specified in factors_to_keep "
                "not found in model.factor_returns columns."
            )
        projected_model.factor_returns = model.factor_returns[factors_to_keep]
    elif hasattr(model, 'factor_returns') and model.factor_returns is None:
        projected_model.factor_returns = None

    return projected_model
