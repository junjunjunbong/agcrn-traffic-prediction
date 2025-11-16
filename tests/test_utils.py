"""
Tests for utility functions
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from src.utils.validation import (
    validate_tensor_shape,
    validate_no_inf_nan,
    validate_value_range,
    DataValidationError
)


class TestValidation:
    """Test validation utilities"""

    def test_validate_tensor_shape_valid(self):
        """Test tensor shape validation with valid input"""
        tensor = np.random.randn(10, 20, 3)
        # Should not raise
        validate_tensor_shape(tensor, (10, 20, 3))
        validate_tensor_shape(tensor, (-1, 20, 3))
        validate_tensor_shape(tensor, (10, -1, -1))

    def test_validate_tensor_shape_invalid(self):
        """Test tensor shape validation with invalid input"""
        tensor = np.random.randn(10, 20, 3)

        with pytest.raises(DataValidationError):
            validate_tensor_shape(tensor, (10, 20, 5))

        with pytest.raises(DataValidationError):
            validate_tensor_shape(tensor, (10, 20))

    def test_validate_no_inf_nan_valid(self):
        """Test inf/nan validation with valid input"""
        tensor = np.random.randn(10, 20)
        # Should not raise
        validate_no_inf_nan(tensor)

    def test_validate_no_inf_nan_with_nan(self):
        """Test inf/nan validation with NaN values"""
        tensor = np.array([1.0, 2.0, np.nan, 4.0])

        with pytest.raises(DataValidationError):
            validate_no_inf_nan(tensor)

        # Should not raise when NaN is allowed
        validate_no_inf_nan(tensor, allow_nan=True)

    def test_validate_no_inf_nan_with_inf(self):
        """Test inf/nan validation with inf values"""
        tensor = np.array([1.0, 2.0, np.inf, 4.0])

        with pytest.raises(DataValidationError):
            validate_no_inf_nan(tensor)

    def test_validate_value_range_valid(self):
        """Test value range validation with valid input"""
        tensor = np.array([0.5, 1.0, 1.5, 2.0])

        # Should not raise
        validate_value_range(tensor, min_val=0.0, max_val=3.0)
        validate_value_range(tensor, min_val=0.0)
        validate_value_range(tensor, max_val=3.0)

    def test_validate_value_range_below_min(self):
        """Test value range validation with values below minimum"""
        tensor = np.array([0.5, 1.0, -1.0, 2.0])

        with pytest.raises(DataValidationError):
            validate_value_range(tensor, min_val=0.0)

    def test_validate_value_range_above_max(self):
        """Test value range validation with values above maximum"""
        tensor = np.array([0.5, 1.0, 5.0, 2.0])

        with pytest.raises(DataValidationError):
            validate_value_range(tensor, max_val=3.0)


class TestConfigLoader:
    """Test configuration loader"""

    def test_load_yaml_config(self):
        """Test loading YAML configuration"""
        from src.utils.config_loader import load_config

        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
model:
  hidden_dim: 64
  num_layers: 2
training:
  batch_size: 32
  learning_rate: 0.001
            """)
            temp_path = Path(f.name)

        try:
            config = load_config(temp_path)

            assert 'model' in config
            assert config['model']['hidden_dim'] == 64
            assert config['model']['num_layers'] == 2
            assert config['training']['batch_size'] == 32
        finally:
            temp_path.unlink()

    def test_config_loader_get(self):
        """Test ConfigLoader get method"""
        from src.utils.config_loader import ConfigLoader

        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
model:
  hidden_dim: 64
training:
  batch_size: 32
            """)
            temp_path = Path(f.name)

        try:
            loader = ConfigLoader(temp_path)

            assert loader.get('model.hidden_dim') == 64
            assert loader.get('training.batch_size') == 32
            assert loader.get('nonexistent', default=100) == 100
        finally:
            temp_path.unlink()

    def test_merge_configs(self):
        """Test merging two configurations"""
        from src.utils.config_loader import merge_configs

        base = {
            'model': {'hidden_dim': 64, 'num_layers': 2},
            'training': {'batch_size': 32}
        }

        override = {
            'model': {'hidden_dim': 128},
            'training': {'learning_rate': 0.001}
        }

        merged = merge_configs(base, override)

        assert merged['model']['hidden_dim'] == 128  # Overridden
        assert merged['model']['num_layers'] == 2    # From base
        assert merged['training']['batch_size'] == 32  # From base
        assert merged['training']['learning_rate'] == 0.001  # From override


class TestLogger:
    """Test logging utilities"""

    def test_get_logger(self):
        """Test getting a logger instance"""
        from src.utils.logger import get_logger

        logger = get_logger(__name__)
        assert logger is not None
        assert logger.name == __name__

    def test_setup_logging(self):
        """Test setting up logging"""
        from src.utils.logger import setup_logging
        import logging

        # Should not raise
        setup_logging(level=logging.INFO)

    def test_log_model_info(self):
        """Test logging model information"""
        from src.utils.logger import log_model_info, get_logger
        from src.model_agcrn import AGCRN

        logger = get_logger(__name__)
        model = AGCRN(num_nodes=10, input_dim=3)

        # Should not raise
        log_model_info(model, logger)
