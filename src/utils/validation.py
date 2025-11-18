"""
Data validation utilities
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass


def validate_csv_file(csv_path: Path, required_columns: List[str]) -> None:
    """
    Validate that CSV file exists and has required columns

    Args:
        csv_path: Path to CSV file
        required_columns: List of required column names

    Raises:
        FileNotFoundError: If file does not exist
        DataValidationError: If required columns are missing
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if not csv_path.is_file():
        raise DataValidationError(f"Path is not a file: {csv_path}")

    # Read header only to check columns
    try:
        df_header = pd.read_csv(csv_path, nrows=0)
    except Exception as e:
        raise DataValidationError(f"Failed to read CSV file {csv_path}: {str(e)}")

    missing_columns = set(required_columns) - set(df_header.columns)
    if missing_columns:
        raise DataValidationError(
            f"Missing required columns in {csv_path.name}: {missing_columns}"
        )

    logger.info(f"[OK] CSV validation passed: {csv_path.name}")


def validate_tensor_shape(
    tensor: np.ndarray,
    expected_shape: Tuple[int, ...],
    name: str = "tensor"
) -> None:
    """
    Validate tensor shape

    Args:
        tensor: NumPy array to validate
        expected_shape: Expected shape (use -1 for any dimension)
        name: Name of tensor for error messages

    Raises:
        DataValidationError: If shape doesn't match
    """
    if tensor.ndim != len(expected_shape):
        raise DataValidationError(
            f"{name} has {tensor.ndim} dimensions, expected {len(expected_shape)}"
        )

    for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
        if expected != -1 and actual != expected:
            raise DataValidationError(
                f"{name} dimension {i}: expected {expected}, got {actual}"
            )

    logger.debug(f"[OK] Tensor shape validation passed: {name} {tensor.shape}")


def validate_no_inf_nan(
    tensor: np.ndarray,
    name: str = "tensor",
    allow_nan: bool = False
) -> None:
    """
    Validate that tensor has no inf or nan values

    Args:
        tensor: NumPy array to validate
        name: Name of tensor for error messages
        allow_nan: If True, allow NaN values

    Raises:
        DataValidationError: If invalid values found
    """
    if np.any(np.isinf(tensor)):
        inf_count = np.isinf(tensor).sum()
        raise DataValidationError(
            f"{name} contains {inf_count} inf values"
        )

    if not allow_nan and np.any(np.isnan(tensor)):
        nan_count = np.isnan(tensor).sum()
        raise DataValidationError(
            f"{name} contains {nan_count} NaN values"
        )

    logger.debug(f"[OK] Value validation passed: {name}")


def validate_value_range(
    tensor: np.ndarray,
    min_val: float = None,
    max_val: float = None,
    name: str = "tensor"
) -> None:
    """
    Validate that tensor values are within specified range

    Args:
        tensor: NumPy array to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name of tensor for error messages

    Raises:
        DataValidationError: If values out of range
    """
    if min_val is not None:
        if np.any(tensor < min_val):
            below_count = (tensor < min_val).sum()
            raise DataValidationError(
                f"{name} contains {below_count} values below {min_val}"
            )

    if max_val is not None:
        if np.any(tensor > max_val):
            above_count = (tensor > max_val).sum()
            raise DataValidationError(
                f"{name} contains {above_count} values above {max_val}"
            )

    logger.debug(f"[OK] Range validation passed: {name}")


def validate_processed_data(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    num_nodes: int,
    num_features: int
) -> None:
    """
    Validate processed data splits

    Args:
        train_data: Training data tensor (T, N, F)
        val_data: Validation data tensor (T, N, F)
        test_data: Test data tensor (T, N, F)
        num_nodes: Expected number of nodes
        num_features: Expected number of features

    Raises:
        DataValidationError: If validation fails
    """
    logger.info("Validating processed data...")

    # Check shapes
    validate_tensor_shape(train_data, (-1, num_nodes, num_features), "train_data")
    validate_tensor_shape(val_data, (-1, num_nodes, num_features), "val_data")
    validate_tensor_shape(test_data, (-1, num_nodes, num_features), "test_data")

    # Check for inf/nan
    validate_no_inf_nan(train_data, "train_data")
    validate_no_inf_nan(val_data, "val_data")
    validate_no_inf_nan(test_data, "test_data")

    # Check that we have enough samples
    if train_data.shape[0] < 10:
        raise DataValidationError(
            f"Training set too small: {train_data.shape[0]} samples"
        )

    logger.info("[OK] All data validation checks passed")
