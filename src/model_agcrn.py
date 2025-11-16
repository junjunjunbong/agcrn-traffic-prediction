"""
AGCRN (Adaptive Graph Convolutional Recurrent Network) Model
Based on the paper: "Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from src.config import NUM_NODES, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, CHEB_K, EMBED_DIM


class AVWGCN(nn.Module):
    """
    Adaptive Graph Convolutional Network with learnable adjacency matrix
    """
    def __init__(self, cheb_k: int, embed_dim: int, num_nodes: int, dim_in: int, dim_out: int):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        
    def forward(self, x: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_nodes, dim_in)
            node_embeddings: (num_nodes, embed_dim)
            
        Returns:
            output: (batch, num_nodes, dim_out)
        """
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        
        support_set = [torch.eye(node_num).to(supports.device), supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        
        supports = torch.stack(support_set, dim=0)  # (cheb_k, num_nodes, num_nodes)
        
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # (num_nodes, cheb_k, dim_in, dim_out)
        bias = torch.matmul(node_embeddings, self.bias_pool)  # (num_nodes, dim_out)
        
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # (batch, cheb_k, num_nodes, dim_in)
        x_g = x_g.permute(0, 2, 1, 3)  # (batch, num_nodes, cheb_k, dim_in)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # (batch, num_nodes, dim_out)
        
        return x_gconv


class AGCRNCell(nn.Module):
    """
    Single AGCRN cell
    """
    def __init__(self, num_nodes: int, embed_dim: int, cheb_k: int, dim_in: int, dim_out: int):
        super(AGCRNCell, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = dim_out
        self.gate = AVWGCN(cheb_k, embed_dim, num_nodes, dim_in + dim_out, 2 * dim_out)
        self.update = AVWGCN(cheb_k, embed_dim, num_nodes, dim_in + dim_out, dim_out)
        
    def forward(self, x: torch.Tensor, state: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_nodes, dim_in)
            state: (batch, num_nodes, hidden_dim)
            node_embeddings: (num_nodes, embed_dim)
            
        Returns:
            new_state: (batch, num_nodes, hidden_dim)
        """
        state = state.to(x.device)
        input_and_state = torch.cat([x, state], dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat([x, r * state], dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = (1 - z) * state + z * hc
        return h


class AGCRN(nn.Module):
    """
    Adaptive Graph Convolutional Recurrent Network
    """
    def __init__(
        self,
        num_nodes: int = NUM_NODES,
        input_dim: int = INPUT_DIM,
        output_dim: int = 1,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        cheb_k: int = CHEB_K,
        default_graph: Optional[torch.Tensor] = None
    ):
        super(AGCRN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node embeddings (learnable)
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
        
        # AGCRN cells
        self.agcrn_cells = nn.ModuleList()
        self.agcrn_cells.append(AGCRNCell(num_nodes, embed_dim, cheb_k, input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.agcrn_cells.append(AGCRNCell(num_nodes, embed_dim, cheb_k, hidden_dim, hidden_dim))
        
        # Output projection
        self.projection_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, sequence_length, num_nodes, input_dim)
            
        Returns:
            output: (batch, num_nodes, output_dim)
        """
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        
        # Initialize hidden states
        h = [torch.zeros(batch_size, self.num_nodes, self.hidden_dim).to(x.device) 
             for _ in range(self.num_layers)]
        
        # Process sequence
        for t in range(sequence_length):
            x_t = x[:, t, :, :]  # (batch, num_nodes, input_dim)
            
            # Forward through layers
            for layer_idx, cell in enumerate(self.agcrn_cells):
                h[layer_idx] = cell(x_t, h[layer_idx], self.node_embeddings)
                x_t = h[layer_idx]
        
        # Use final hidden state for prediction
        output = self.projection_layer(h[-1])  # (batch, num_nodes, output_dim)
        
        return output
    
    def predict_multi_step(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Multi-step prediction using autoregressive approach
        
        Args:
            x: (batch, sequence_length, num_nodes, input_dim)
            horizon: Number of steps to predict ahead
            
        Returns:
            predictions: (batch, horizon, num_nodes, output_dim)
        """
        batch_size = x.shape[0]
        predictions = []
        
        # Initialize hidden states
        h = [torch.zeros(batch_size, self.num_nodes, self.hidden_dim).to(x.device) 
             for _ in range(self.num_layers)]
        
        # Process input sequence
        for t in range(x.shape[1]):
            x_t = x[:, t, :, :]
            for layer_idx, cell in enumerate(self.agcrn_cells):
                h[layer_idx] = cell(x_t, h[layer_idx], self.node_embeddings)
                x_t = h[layer_idx]
        
        # Autoregressive prediction
        current_input = x[:, -1, :, :]  # Last input
        for step in range(horizon):
            # Forward through layers
            x_t = current_input
            for layer_idx, cell in enumerate(self.agcrn_cells):
                h[layer_idx] = cell(x_t, h[layer_idx], self.node_embeddings)
                x_t = h[layer_idx]
            
            # Predict next step
            pred = self.projection_layer(h[-1])  # (batch, num_nodes, output_dim)
            predictions.append(pred)
            
            # Use prediction as next input (assuming output_dim == input_dim for autoregressive)
            if self.output_dim == self.input_dim:
                current_input = pred
            else:
                # If dimensions don't match, use last input
                current_input = x[:, -1, :, :]
        
        return torch.stack(predictions, dim=1)  # (batch, horizon, num_nodes, output_dim)

