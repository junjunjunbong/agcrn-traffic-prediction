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
        mask: Optional observation mask of shape (T, N, F)
        sequence_length: Input sequence length
        horizon: Prediction horizon (number of steps ahead)
        stride: Stride for sliding window (default: 1)
        max_missing_gap: Maximum consecutive missing timesteps allowed (default: 60 = 5 minutes)
        filter_long_gaps: Whether to filter out sequences with long missing gaps
    """
    def __init__(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        sequence_length: int = SEQUENCE_LENGTH,
        horizon: int = HORIZON,
        stride: int = 1,
        max_missing_gap: int = 60,
        filter_long_gaps: bool = True
    ):
        self.data = data.astype(np.float32)
        self.mask = mask.astype(np.float32) if mask is not None else None
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.stride = stride
        self.max_missing_gap = max_missing_gap
        self.filter_long_gaps = filter_long_gaps

        # Create valid indices
        self.indices = []
        T = self.data.shape[0]
        total_samples = 0
        filtered_samples = 0

        for i in range(0, T - sequence_length - horizon + 1, stride):
            total_samples += 1

            # Check if this sequence should be filtered
            if self.filter_long_gaps and self.mask is not None:
                # Check for long consecutive missing gaps in the sequence
                sequence_mask = self.mask[i:i + sequence_length + horizon]
                if self._has_long_gap(sequence_mask):
                    filtered_samples += 1
                    continue

            self.indices.append(i)

        if self.mask is not None:
            observation_rate = self.mask.mean() * 100
            print(f"Dataset created: {len(self.indices)} samples from shape {data.shape}")
            print(f"  Observation rate: {observation_rate:.2f}%")
            if self.filter_long_gaps:
                print(f"  Filtered {filtered_samples}/{total_samples} samples with gaps > {max_missing_gap} timesteps")
        else:
            print(f"Dataset created: {len(self.indices)} samples from shape {data.shape}")
    
    def _has_long_gap(self, sequence_mask: np.ndarray) -> bool:
        """
        Check if sequence has consecutive missing values exceeding threshold

        Args:
            sequence_mask: Mask of shape (T, N, F)

        Returns:
            True if there's a gap longer than max_missing_gap
        """
        # Check for any node/feature combination
        T, N, F = sequence_mask.shape

        for n in range(N):
            for f in range(F):
                series = sequence_mask[:, n, f]

                # Find consecutive False values (missing observations)
                is_missing = ~series.astype(bool)

                # Find runs of consecutive missing values
                changes = np.diff(np.concatenate([[False], is_missing, [False]]).astype(int))
                starts = np.where(changes == 1)[0]
                ends = np.where(changes == -1)[0]

                for start, end in zip(starts, ends):
                    gap_length = end - start
                    if gap_length > self.max_missing_gap:
                        return True

        return False

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Returns:
            x: Input sequence of shape (sequence_length, N, F)
            y: Target sequence of shape (horizon, N, F)
            masks: Optional tuple of (mask_x, mask_y) if mask is available
        """
        start_idx = self.indices[idx]

        # Input sequence
        x = self.data[start_idx:start_idx + self.sequence_length]

        # Target sequence
        y = self.data[start_idx + self.sequence_length:start_idx + self.sequence_length + self.horizon]

        # Convert to torch tensors
        x = torch.from_numpy(x).float()  # (T_in, N, F)
        y = torch.from_numpy(y).float()  # (T_out, N, F)

        # Return masks if available
        if self.mask is not None:
            mask_x = self.mask[start_idx:start_idx + self.sequence_length]
            mask_y = self.mask[start_idx + self.sequence_length:start_idx + self.sequence_length + self.horizon]

            mask_x = torch.from_numpy(mask_x).float()
            mask_y = torch.from_numpy(mask_y).float()

            return x, y, (mask_x, mask_y)
        else:
            return x, y, None


def load_processed_data(data_name: str, load_masks: bool = True):
    """
    Load processed data from npz file

    Args:
        data_name: Name of the data file (e.g., 'loops_035')
        load_masks: Whether to load observation masks

    Returns:
        If load_masks=True:
            (train_data, val_data, test_data), (mask_train, mask_val, mask_test)
        If load_masks=False:
            train_data, val_data, test_data
    """
    npz_path = PROCESSED_DATA_DIR / f"{data_name}_processed.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"Processed data not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    train_data = data['train']
    val_data = data['val']
    test_data = data['test']

    if load_masks and 'mask_train' in data:
        mask_train = data['mask_train']
        mask_val = data['mask_val']
        mask_test = data['mask_test']
        return (train_data, val_data, test_data), (mask_train, mask_val, mask_test)
    else:
        return train_data, val_data, test_data


def create_dataloaders(
    data_name: str,
    batch_size: int = 64,
    sequence_length: int = SEQUENCE_LENGTH,
    horizon: int = HORIZON,
    shuffle_train: bool = True,
    use_masks: bool = True,
    filter_long_gaps: bool = True,
    max_missing_gap: int = 60
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create DataLoaders for train/val/test sets

    Args:
        data_name: Name of the data file
        batch_size: Batch size
        sequence_length: Input sequence length
        horizon: Prediction horizon
        shuffle_train: Whether to shuffle training data
        use_masks: Whether to use observation masks
        filter_long_gaps: Whether to filter sequences with long missing gaps
        max_missing_gap: Maximum consecutive missing timesteps allowed (default: 60 = 5 min)

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load data and masks
    result = load_processed_data(data_name, load_masks=use_masks)

    if use_masks and isinstance(result, tuple):
        (train_data, val_data, test_data), (mask_train, mask_val, mask_test) = result
        print(f"Loaded data with observation masks")
    else:
        train_data, val_data, test_data = result if isinstance(result, tuple) else result
        mask_train, mask_val, mask_test = None, None, None
        print(f"Loaded data without masks")

    # Create datasets
    train_dataset = TrafficDataset(
        train_data, mask_train, sequence_length, horizon,
        max_missing_gap=max_missing_gap, filter_long_gaps=filter_long_gaps
    )
    val_dataset = TrafficDataset(
        val_data, mask_val, sequence_length, horizon,
        max_missing_gap=max_missing_gap, filter_long_gaps=filter_long_gaps
    )
    test_dataset = TrafficDataset(
        test_data, mask_test, sequence_length, horizon,
        max_missing_gap=max_missing_gap, filter_long_gaps=filter_long_gaps
    )

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

