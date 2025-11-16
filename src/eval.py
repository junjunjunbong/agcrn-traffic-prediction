"""
Evaluation module for AGCRN model
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt

from src.config import DEVICE
from src.model_agcrn import AGCRN


def evaluate_model(
    model: AGCRN,
    test_loader: DataLoader,
    criterion: torch.nn.Module = torch.nn.MSELoss(),
    device: str = DEVICE
) -> Dict[str, float]:
    """
    Evaluate model on test set
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_mae = 0.0
    total_mape = 0.0
    num_batches = 0
    num_samples = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            
            y_target = y[:, -1, :, :]  # Last time step
            
            output = model(x)
            
            # Handle different output dimensions
            if output.shape[-1] == y_target.shape[-1]:
                pred = output
            else:
                # Assume predicting speed only
                pred = output.unsqueeze(-1).expand_as(y_target[:, :, :1])
                y_target = y_target[:, :, :1]
            
            # Calculate metrics
            loss = criterion(pred, y_target)
            mae = torch.mean(torch.abs(pred - y_target))
            
            # MAPE (avoid division by zero)
            mask = y_target != 0
            if mask.sum() > 0:
                mape = torch.mean(torch.abs((pred[mask] - y_target[mask]) / y_target[mask])) * 100
            else:
                mape = torch.tensor(0.0)
            
            total_loss += loss.item()
            total_mae += mae.item()
            total_mape += mape.item()
            num_batches += 1
            num_samples += x.shape[0]
            
            all_predictions.append(pred.cpu().numpy())
            all_targets.append(y_target.cpu().numpy())
    
    metrics = {
        'mse': total_loss / num_batches,
        'rmse': np.sqrt(total_loss / num_batches),
        'mae': total_mae / num_batches,
        'mape': total_mape / num_batches,
        'num_samples': num_samples
    }
    
    return metrics, np.concatenate(all_predictions, axis=0), np.concatenate(all_targets, axis=0)


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[Path] = None,
    num_nodes_to_plot: int = 5,
    num_samples_to_plot: int = 100
):
    """
    Plot predictions vs targets for visualization
    
    Args:
        predictions: (num_samples, num_nodes, output_dim)
        targets: (num_samples, num_nodes, output_dim)
        save_path: Path to save plot
        num_nodes_to_plot: Number of nodes to visualize
        num_samples_to_plot: Number of samples to plot
    """
    num_nodes = min(num_nodes_to_plot, predictions.shape[1])
    num_samples = min(num_samples_to_plot, predictions.shape[0])
    
    fig, axes = plt.subplots(num_nodes, 1, figsize=(12, 3*num_nodes))
    if num_nodes == 1:
        axes = [axes]
    
    for i in range(num_nodes):
        ax = axes[i]
        pred_node = predictions[:num_samples, i, 0]
        target_node = targets[:num_samples, i, 0]
        
        ax.plot(target_node, label='Target', alpha=0.7)
        ax.plot(pred_node, label='Prediction', alpha=0.7)
        ax.set_title(f'Node {i}')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def load_model(model_path: Path, model_config: Dict) -> AGCRN:
    """
    Load trained model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        model_config: Model configuration dictionary
        
    Returns:
        model: Loaded AGCRN model
    """
    model = AGCRN(**model_config)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

