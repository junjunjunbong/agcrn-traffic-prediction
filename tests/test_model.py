"""
Tests for AGCRN model
"""
import pytest
import torch
import numpy as np
from src.model_agcrn import AGCRN, AGCRNCell, AVWGCN


class TestAVWGCN:
    """Test Adaptive Graph Convolutional Network"""

    def test_initialization(self):
        """Test AVWGCN initialization"""
        layer = AVWGCN(cheb_k=2, embed_dim=10, num_nodes=20, dim_in=3, dim_out=64)
        assert layer.cheb_k == 2
        assert layer.embed_dim == 10
        assert layer.num_nodes == 20

    def test_forward_shape(self):
        """Test AVWGCN forward pass output shape"""
        batch_size = 4
        num_nodes = 20
        dim_in = 3
        dim_out = 64
        embed_dim = 10

        layer = AVWGCN(cheb_k=2, embed_dim=embed_dim, num_nodes=num_nodes,
                       dim_in=dim_in, dim_out=dim_out)

        x = torch.randn(batch_size, num_nodes, dim_in)
        node_embeddings = torch.randn(num_nodes, embed_dim)

        output = layer(x, node_embeddings)

        assert output.shape == (batch_size, num_nodes, dim_out)


class TestAGCRNCell:
    """Test AGCRN Cell"""

    def test_initialization(self):
        """Test AGCRNCell initialization"""
        cell = AGCRNCell(num_nodes=20, embed_dim=10, cheb_k=2, dim_in=3, dim_out=64)
        assert cell.num_nodes == 20
        assert cell.hidden_dim == 64

    def test_forward_shape(self):
        """Test AGCRNCell forward pass output shape"""
        batch_size = 4
        num_nodes = 20
        dim_in = 3
        dim_out = 64
        embed_dim = 10

        cell = AGCRNCell(num_nodes=num_nodes, embed_dim=embed_dim, cheb_k=2,
                        dim_in=dim_in, dim_out=dim_out)

        x = torch.randn(batch_size, num_nodes, dim_in)
        state = torch.zeros(batch_size, num_nodes, dim_out)
        node_embeddings = torch.randn(num_nodes, embed_dim)

        output = cell(x, state, node_embeddings)

        assert output.shape == (batch_size, num_nodes, dim_out)


class TestAGCRN:
    """Test AGCRN model"""

    def test_initialization(self):
        """Test AGCRN initialization"""
        model = AGCRN(num_nodes=20, input_dim=3, output_dim=1,
                     hidden_dim=64, num_layers=2, cheb_k=2, embed_dim=10)

        assert model.num_nodes == 20
        assert model.input_dim == 3
        assert model.output_dim == 1
        assert model.hidden_dim == 64
        assert model.num_layers == 2

    def test_forward_shape(self):
        """Test AGCRN forward pass output shape"""
        batch_size = 4
        seq_len = 12
        num_nodes = 20
        input_dim = 3
        output_dim = 1

        model = AGCRN(num_nodes=num_nodes, input_dim=input_dim,
                     output_dim=output_dim, hidden_dim=64, num_layers=2)

        x = torch.randn(batch_size, seq_len, num_nodes, input_dim)
        output = model(x)

        assert output.shape == (batch_size, num_nodes, output_dim)

    def test_predict_multi_step(self):
        """Test multi-step prediction"""
        batch_size = 4
        seq_len = 12
        num_nodes = 20
        input_dim = 3
        output_dim = 1
        horizon = 3

        model = AGCRN(num_nodes=num_nodes, input_dim=input_dim,
                     output_dim=output_dim, hidden_dim=64, num_layers=2)

        x = torch.randn(batch_size, seq_len, num_nodes, input_dim)
        predictions = model.predict_multi_step(x, horizon=horizon)

        assert predictions.shape == (batch_size, horizon, num_nodes, output_dim)

    def test_parameter_count(self):
        """Test that model has trainable parameters"""
        model = AGCRN(num_nodes=20, input_dim=3, output_dim=1)
        params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert params > 0
        assert trainable_params > 0
        assert trainable_params == params  # All params should be trainable

    @pytest.mark.parametrize("num_nodes,input_dim,output_dim", [
        (10, 3, 1),
        (50, 5, 1),
        (100, 3, 3),
    ])
    def test_different_configurations(self, num_nodes, input_dim, output_dim):
        """Test model with different configurations"""
        model = AGCRN(num_nodes=num_nodes, input_dim=input_dim,
                     output_dim=output_dim, hidden_dim=32, num_layers=1)

        batch_size = 2
        seq_len = 6
        x = torch.randn(batch_size, seq_len, num_nodes, input_dim)
        output = model(x)

        assert output.shape == (batch_size, num_nodes, output_dim)
