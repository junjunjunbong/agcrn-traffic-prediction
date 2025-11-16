"""
Tests for data preprocessing module
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.preprocess import (
    validate_input_data,
    validate_tensor,
    create_node_mapping,
    convert_to_tensor_vectorized,
    interpolate_all_features,
    split_data,
    normalize_data
)
from src.config import FEATURES, NODE_MODE


class TestValidation:
    """Test validation functions"""

    def test_validate_input_data_valid(self):
        """Test with valid input data"""
        df = pd.DataFrame({
            'begin': [0.0, 5.0, 10.0],
            'end': [5.0, 10.0, 15.0],
            'raw_id': ['DLc000000_0', 'DLc000000_0', 'DLc000000_0'],
            'det_pos': [0, 0, 0],
            'flow': [10.0, 15.0, 12.0],
            'occupancy': [0.3, 0.4, 0.35],
            'harmonicMeanSpeed': [15.0, 14.0, 15.5]
        })

        # Should not raise
        validate_input_data(df)

    def test_validate_input_data_missing_columns(self):
        """Test with missing required columns"""
        df = pd.DataFrame({
            'begin': [0.0, 5.0],
            'end': [5.0, 10.0],
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_input_data(df)

    def test_validate_input_data_empty(self):
        """Test with empty DataFrame"""
        df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            validate_input_data(df)

    def test_validate_tensor_valid(self):
        """Test tensor validation with valid tensor"""
        X = np.random.randn(100, 10, 3)

        # Should not raise
        validate_tensor(X, "test_tensor", allow_nan=False)

    def test_validate_tensor_with_nan(self):
        """Test tensor validation with NaN values"""
        X = np.random.randn(100, 10, 3)
        X[0, 0, 0] = np.nan

        # Should raise when NaN not allowed
        with pytest.raises(ValueError, match="NaN"):
            validate_tensor(X, "test_tensor", allow_nan=False)

        # Should not raise when NaN allowed
        validate_tensor(X, "test_tensor", allow_nan=True)

    def test_validate_tensor_wrong_dimensions(self):
        """Test tensor validation with wrong dimensions"""
        X = np.random.randn(100, 10)  # 2D instead of 3D

        with pytest.raises(ValueError, match="must be 3D"):
            validate_tensor(X, "test_tensor")

    def test_validate_tensor_with_inf(self):
        """Test tensor validation with inf values"""
        X = np.random.randn(100, 10, 3)
        X[0, 0, 0] = np.inf

        with pytest.raises(ValueError, match="inf"):
            validate_tensor(X, "test_tensor")


class TestNodeMapping:
    """Test node mapping creation"""

    def test_create_node_mapping_raw_id(self):
        """Test node mapping for raw_id mode"""
        df = pd.DataFrame({
            'raw_id': ['DLc000000_0', 'DLc000001_0', 'DLc000002_0'],
            'det_pos': [0, 1, 2],
            'lane_idx': [0, 0, 0],
            'edge_id': [-5414, -5414, -5414]
        })

        # Temporarily set NODE_MODE
        import src.config as config
        original_mode = config.NODE_MODE
        config.NODE_MODE = "raw_id"

        try:
            sensors_df, node_to_idx = create_node_mapping(df)

            assert len(sensors_df) == 3
            assert len(node_to_idx) == 3
            assert 'DLc000000_0' in node_to_idx
            assert node_to_idx['DLc000000_0'] == 0
        finally:
            config.NODE_MODE = original_mode

    def test_create_node_mapping_det_pos(self):
        """Test node mapping for det_pos mode"""
        df = pd.DataFrame({
            'raw_id': ['DLc000000_0', 'DLc000000_1', 'DLc000001_0'],
            'det_pos': [0, 0, 1],
            'lane_idx': [0, 1, 0],
            'edge_id': [-5414, -5414, -5414]
        })

        import src.config as config
        original_mode = config.NODE_MODE
        config.NODE_MODE = "det_pos"

        try:
            sensors_df, node_to_idx = create_node_mapping(df)

            # Should have 2 unique det_pos values
            assert len(sensors_df) == 2
            assert len(node_to_idx) == 2
            assert 0 in node_to_idx
            assert 1 in node_to_idx
        finally:
            config.NODE_MODE = original_mode


class TestConvertToTensor:
    """Test tensor conversion"""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame"""
        data = []
        for t in [0.0, 5.0, 10.0]:
            for raw_id in ['DLc000000_0', 'DLc000001_0']:
                data.append({
                    'begin': t,
                    'end': t + 5.0,
                    'raw_id': raw_id,
                    'det_pos': int(raw_id.split('_')[0][-1]),
                    'flow': np.random.uniform(10, 20),
                    'occupancy': np.random.uniform(0.2, 0.5),
                    'harmonicMeanSpeed': np.random.uniform(12, 18)
                })
        return pd.DataFrame(data)

    def test_convert_to_tensor_shape(self, sample_df):
        """Test that tensor has correct shape"""
        import src.config as config
        original_mode = config.NODE_MODE
        config.NODE_MODE = "raw_id"

        try:
            sensors_df, node_to_idx = create_node_mapping(sample_df)
            time_steps = np.arange(3)  # 3 time steps

            X = convert_to_tensor_vectorized(sample_df, node_to_idx, time_steps)

            # Check shape
            assert X.shape == (3, 2, 3)  # (T, N, F)
            assert X.ndim == 3
        finally:
            config.NODE_MODE = original_mode

    def test_convert_to_tensor_det_pos_aggregation(self):
        """Test that det_pos mode properly aggregates lanes"""
        # Create data with multiple lanes at same position
        df = pd.DataFrame({
            'begin': [0.0, 0.0, 0.0],
            'end': [5.0, 5.0, 5.0],
            'raw_id': ['DLc000000_0', 'DLc000000_1', 'DLc000000_2'],
            'det_pos': [0, 0, 0],  # Same position, different lanes
            'lane_idx': [0, 1, 2],
            'flow': [10.0, 15.0, 12.0],  # Should sum to 37
            'occupancy': [0.3, 0.4, 0.35],  # Should average to ~0.35
            'harmonicMeanSpeed': [15.0, 14.0, 15.5]  # Should average to ~14.83
        })

        import src.config as config
        original_mode = config.NODE_MODE
        config.NODE_MODE = "det_pos"

        try:
            sensors_df, node_to_idx = create_node_mapping(df)
            time_steps = np.arange(1)

            X = convert_to_tensor_vectorized(df, node_to_idx, time_steps)

            # Check aggregation
            flow_idx = FEATURES.index('flow')
            occ_idx = FEATURES.index('occupancy')

            # Flow should be summed
            assert np.isclose(X[0, 0, flow_idx], 37.0)

            # Occupancy should be averaged
            assert np.isclose(X[0, 0, occ_idx], 0.35, atol=0.01)
        finally:
            config.NODE_MODE = original_mode


class TestInterpolation:
    """Test interpolation functions"""

    def test_interpolate_all_features(self):
        """Test interpolation fills NaN values"""
        # Create tensor with NaN values
        X = np.random.randn(10, 5, 3)
        X[2, 0, 0] = np.nan  # flow
        X[3, 1, 1] = np.nan  # occupancy
        X[4, 2, 2] = np.nan  # speed

        X_interp = interpolate_all_features(X)

        # Should have no NaN
        assert np.isnan(X_interp).sum() == 0

    def test_interpolate_all_nan_series(self):
        """Test interpolation with all-NaN series"""
        X = np.random.randn(10, 3, 3)
        X[:, 0, 0] = np.nan  # All NaN for one node-feature

        X_interp = interpolate_all_features(X)

        # Should fill with default value (0 for flow)
        flow_idx = 0
        assert np.all(X_interp[:, 0, flow_idx] == 0.0)


class TestDataSplit:
    """Test data splitting"""

    def test_split_data_shapes(self):
        """Test that split maintains correct proportions"""
        X = np.random.randn(100, 10, 3)

        X_train, X_val, X_test = split_data(X)

        # Check proportions (70/15/15)
        assert X_train.shape[0] == 70
        assert X_val.shape[0] == 15
        assert X_test.shape[0] == 15

        # Check other dimensions preserved
        assert X_train.shape[1:] == (10, 3)
        assert X_val.shape[1:] == (10, 3)
        assert X_test.shape[1:] == (10, 3)

    def test_split_data_no_overlap(self):
        """Test that splits don't overlap"""
        X = np.arange(100 * 10 * 3).reshape(100, 10, 3)

        X_train, X_val, X_test = split_data(X)

        # Check no overlap
        assert not np.any(np.isin(X_train, X_val))
        assert not np.any(np.isin(X_train, X_test))
        assert not np.any(np.isin(X_val, X_test))


class TestNormalization:
    """Test normalization"""

    def test_normalize_data(self):
        """Test z-score normalization"""
        # Create data with known statistics
        X_train = np.ones((100, 10, 3)) * 10.0
        X_val = np.ones((20, 10, 3)) * 10.0
        X_test = np.ones((20, 10, 3)) * 10.0

        X_train_norm, X_val_norm, X_test_norm, stats = normalize_data(
            X_train, X_val, X_test
        )

        # After normalization, train mean should be ~0
        for f_idx in range(3):
            assert np.abs(X_train_norm[:, :, f_idx].mean()) < 1e-10

    def test_normalize_data_with_nan_raises(self):
        """Test that normalization raises error with NaN"""
        X_train = np.random.randn(100, 10, 3)
        X_val = np.random.randn(20, 10, 3)
        X_test = np.random.randn(20, 10, 3)

        X_train[0, 0, 0] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            normalize_data(X_train, X_val, X_test)

    def test_normalize_stats_format(self):
        """Test that stats dictionary has correct format"""
        X_train = np.random.randn(100, 10, 3)
        X_val = np.random.randn(20, 10, 3)
        X_test = np.random.randn(20, 10, 3)

        _, _, _, stats = normalize_data(X_train, X_val, X_test)

        # Check stats format
        for feature in FEATURES:
            assert feature in stats
            assert 'mean' in stats[feature]
            assert 'std' in stats[feature]
            assert isinstance(stats[feature]['mean'], float)
            assert isinstance(stats[feature]['std'], float)


class TestEndToEnd:
    """End-to-end integration tests"""

    def test_full_pipeline_small_data(self):
        """Test full preprocessing pipeline with small synthetic data"""
        # Create small synthetic dataset
        data = []
        for t in np.arange(0, 50, 5.0):  # 10 time steps
            for node_id in range(5):  # 5 nodes
                data.append({
                    'begin': t,
                    'end': t + 5.0,
                    'raw_id': f'DLc00000{node_id}_0',
                    'det_pos': node_id,
                    'lane_idx': 0,
                    'edge_id': -5414,
                    'flow': np.random.uniform(10, 20),
                    'occupancy': np.random.uniform(0.2, 0.5),
                    'harmonicMeanSpeed': np.random.uniform(12, 18)
                })
        df = pd.DataFrame(data)

        import src.config as config
        original_mode = config.NODE_MODE
        config.NODE_MODE = "raw_id"

        try:
            # Run pipeline
            validate_input_data(df)
            sensors_df, node_to_idx = create_node_mapping(df)
            time_steps = np.arange(10)
            X = convert_to_tensor_vectorized(df, node_to_idx, time_steps)
            X = interpolate_all_features(X)
            X_train, X_val, X_test = split_data(X)
            X_train_norm, X_val_norm, X_test_norm, stats = normalize_data(
                X_train, X_val, X_test
            )

            # Validate results
            assert X_train_norm.shape[0] == 7  # 70% of 10
            assert X_val_norm.shape[0] == 1   # 15% of 10
            assert X_test_norm.shape[0] == 2  # 15% of 10
            assert np.isnan(X_train_norm).sum() == 0
            assert np.isnan(X_val_norm).sum() == 0
            assert np.isnan(X_test_norm).sum() == 0
        finally:
            config.NODE_MODE = original_mode
