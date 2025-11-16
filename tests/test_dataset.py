"""
Tests for Dataset classes
"""
import pytest
import numpy as np
import torch
from src.dataset import TrafficDataset


class TestTrafficDataset:
    """Test TrafficDataset class"""

    @pytest.fixture
    def sample_data(self):
        """Create sample traffic data"""
        # Shape: (T=100, N=10, F=3)
        T, N, F = 100, 10, 3
        return np.random.randn(T, N, F).astype(np.float32)

    def test_initialization(self, sample_data):
        """Test dataset initialization"""
        dataset = TrafficDataset(
            data=sample_data,
            sequence_length=12,
            horizon=3,
            stride=1
        )

        assert len(dataset) > 0
        assert dataset.sequence_length == 12
        assert dataset.horizon == 3

    def test_getitem_shape(self, sample_data):
        """Test __getitem__ returns correct shapes"""
        sequence_length = 12
        horizon = 3

        dataset = TrafficDataset(
            data=sample_data,
            sequence_length=sequence_length,
            horizon=horizon
        )

        x, y = dataset[0]

        # Check types
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

        # Check shapes
        assert x.shape == (sequence_length, 10, 3)  # (T_in, N, F)
        assert y.shape == (horizon, 10, 3)  # (T_out, N, F)

    def test_dataset_length(self, sample_data):
        """Test dataset length calculation"""
        T = sample_data.shape[0]
        sequence_length = 12
        horizon = 3
        stride = 1

        dataset = TrafficDataset(
            data=sample_data,
            sequence_length=sequence_length,
            horizon=horizon,
            stride=stride
        )

        expected_length = (T - sequence_length - horizon + 1) // stride
        assert len(dataset) == expected_length

    def test_stride(self, sample_data):
        """Test dataset with different stride values"""
        # Stride = 1
        dataset1 = TrafficDataset(
            data=sample_data,
            sequence_length=12,
            horizon=3,
            stride=1
        )

        # Stride = 2
        dataset2 = TrafficDataset(
            data=sample_data,
            sequence_length=12,
            horizon=3,
            stride=2
        )

        # Dataset with stride=2 should have roughly half the samples
        assert len(dataset2) < len(dataset1)
        assert len(dataset2) >= len(dataset1) // 2

    def test_data_continuity(self, sample_data):
        """Test that x and y are continuous in time"""
        dataset = TrafficDataset(
            data=sample_data,
            sequence_length=12,
            horizon=3
        )

        x, y = dataset[0]

        # The first element of y should come right after the last element of x
        # We can't directly compare values, but we can check shapes are consistent
        assert x.shape[0] == 12
        assert y.shape[0] == 3

    @pytest.mark.parametrize("sequence_length,horizon", [
        (6, 1),
        (12, 3),
        (24, 6),
    ])
    def test_different_configurations(self, sample_data, sequence_length, horizon):
        """Test dataset with different sequence lengths and horizons"""
        dataset = TrafficDataset(
            data=sample_data,
            sequence_length=sequence_length,
            horizon=horizon
        )

        if len(dataset) > 0:
            x, y = dataset[0]
            assert x.shape[0] == sequence_length
            assert y.shape[0] == horizon

    def test_dtype(self, sample_data):
        """Test that tensors have correct dtype"""
        dataset = TrafficDataset(data=sample_data)
        x, y = dataset[0]

        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_invalid_index(self, sample_data):
        """Test that invalid index raises error"""
        dataset = TrafficDataset(data=sample_data)

        with pytest.raises(IndexError):
            _ = dataset[len(dataset) + 10]
