"""
PyTorch Dataset class for AGCRN traffic prediction
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional

from src.config import PROCESSED_DATA_DIR, SEQUENCE_LENGTH, HORIZON


class TrafficDataset(Dataset):
    """
    Dataset for traffic prediction using sliding window approach
    
    Args:
        data: Tensor of shape (T, N, F)
        sequence_length: Input sequence length
        horizon: Prediction horizon (number of steps ahead)
        stride: Stride for sliding window (default: 1)
    """
    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int = SEQUENCE_LENGTH,
        horizon: int = HORIZON,
        stride: int = 1
    ):
        self.data = data.astype(np.float32)
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.stride = stride
        
        # Create valid indices
        self.indices = []
        T = self.data.shape[0]
        for i in range(0, T - sequence_length - horizon + 1, stride):
            self.indices.append(i)
        
        print(f"Dataset created: {len(self.indices)} samples from shape {data.shape}")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: Input sequence of shape (sequence_length, N, F)
            y: Target sequence of shape (horizon, N, F)
        """
        start_idx = self.indices[idx]
        
        # Input sequence
        x = self.data[start_idx:start_idx + self.sequence_length]
        
        # Target sequence
        y = self.data[start_idx + self.sequence_length:start_idx + self.sequence_length + self.horizon]
        
        # Convert to torch tensors
        x = torch.from_numpy(x).float()  # (T_in, N, F)
        y = torch.from_numpy(y).float()  # (T_out, N, F)
        
        return x, y


def load_processed_data(data_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load processed data from npz file
    
    Args:
        data_name: Name of the data file (e.g., 'loops_035')
        
    Returns:
        train_data, val_data, test_data: Arrays of shape (T, N, F)
    """
    npz_path = PROCESSED_DATA_DIR / f"{data_name}_processed.npz"
    
    if not npz_path.exists():
        raise FileNotFoundError(f"Processed data not found: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    train_data = data['train']
    val_data = data['val']
    test_data = data['test']
    
    return train_data, val_data, test_data


def create_dataloaders(
    data_name: str,
    batch_size: int = 64,
    sequence_length: int = SEQUENCE_LENGTH,
    horizon: int = HORIZON,
    shuffle_train: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create DataLoaders for train/val/test sets
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_data, val_data, test_data = load_processed_data(data_name)
    
    train_dataset = TrafficDataset(train_data, sequence_length, horizon)
    val_dataset = TrafficDataset(val_data, sequence_length, horizon)
    test_dataset = TrafficDataset(test_data, sequence_length, horizon)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

