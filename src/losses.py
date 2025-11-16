"""
Custom loss functions for traffic prediction with missing value handling
"""
import torch
import torch.nn as nn
from typing import Optional


class MaskedMSELoss(nn.Module):
    """
    Masked Mean Squared Error Loss

    Applies different weights to observed vs imputed values:
    - Observed values (mask=1): weight = 1.0
    - Imputed values (mask=0): weight = imputed_weight (default: 0.1)

    Args:
        imputed_weight: Weight for imputed values (0.0 = ignore, 1.0 = same as observed)
        reduction: 'mean' or 'sum'
    """
    def __init__(self, imputed_weight: float = 0.1, reduction: str = 'mean'):
        super(MaskedMSELoss, self).__init__()
        self.imputed_weight = imputed_weight
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (batch, T, N, F) or (batch, N, F)
            target: Ground truth (batch, T, N, F) or (batch, N, F)
            mask: Observation mask (batch, T, N, F) or (batch, N, F)
                  1.0 = observed, 0.0 = imputed

        Returns:
            Weighted MSE loss
        """
        # Compute squared error
        squared_error = (pred - target) ** 2

        if mask is not None:
            # Create weight tensor: 1.0 for observed, imputed_weight for imputed
            weights = mask + (1 - mask) * self.imputed_weight

            # Apply weights
            weighted_error = squared_error * weights

            # Normalize by total weight
            if self.reduction == 'mean':
                total_weight = weights.sum()
                if total_weight > 0:
                    return weighted_error.sum() / total_weight
                else:
                    return weighted_error.mean()
            else:  # sum
                return weighted_error.sum()
        else:
            # No mask: standard MSE
            if self.reduction == 'mean':
                return squared_error.mean()
            else:
                return squared_error.sum()


class MaskedMAELoss(nn.Module):
    """
    Masked Mean Absolute Error Loss

    Similar to MaskedMSELoss but using absolute error instead of squared error.

    Args:
        imputed_weight: Weight for imputed values (0.0 = ignore, 1.0 = same as observed)
        reduction: 'mean' or 'sum'
    """
    def __init__(self, imputed_weight: float = 0.1, reduction: str = 'mean'):
        super(MaskedMAELoss, self).__init__()
        self.imputed_weight = imputed_weight
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (batch, T, N, F) or (batch, N, F)
            target: Ground truth (batch, T, N, F) or (batch, N, F)
            mask: Observation mask (batch, T, N, F) or (batch, N, F)

        Returns:
            Weighted MAE loss
        """
        # Compute absolute error
        absolute_error = torch.abs(pred - target)

        if mask is not None:
            # Create weight tensor
            weights = mask + (1 - mask) * self.imputed_weight

            # Apply weights
            weighted_error = absolute_error * weights

            # Normalize by total weight
            if self.reduction == 'mean':
                total_weight = weights.sum()
                if total_weight > 0:
                    return weighted_error.sum() / total_weight
                else:
                    return weighted_error.mean()
            else:  # sum
                return weighted_error.sum()
        else:
            # No mask: standard MAE
            if self.reduction == 'mean':
                return absolute_error.mean()
            else:
                return absolute_error.sum()


class ObservedOnlyLoss(nn.Module):
    """
    Loss computed ONLY on observed values (ignore imputed values completely)

    This is the most conservative approach - only trust real observations.

    Args:
        loss_fn: Base loss function ('mse' or 'mae')
        reduction: 'mean' or 'sum'
    """
    def __init__(self, loss_fn: str = 'mse', reduction: str = 'mean'):
        super(ObservedOnlyLoss, self).__init__()
        self.loss_fn = loss_fn
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions
            target: Ground truth
            mask: Observation mask (1.0 = observed, 0.0 = imputed)

        Returns:
            Loss computed only on observed values
        """
        if mask is not None:
            # Only compute loss on observed values
            if self.loss_fn == 'mse':
                error = (pred - target) ** 2
            else:  # mae
                error = torch.abs(pred - target)

            # Mask out imputed values
            masked_error = error * mask

            # Compute mean over observed values only
            num_observed = mask.sum()
            if num_observed > 0:
                if self.reduction == 'mean':
                    return masked_error.sum() / num_observed
                else:
                    return masked_error.sum()
            else:
                # No observed values - return zero loss
                return torch.tensor(0.0, device=pred.device)
        else:
            # No mask: standard loss
            if self.loss_fn == 'mse':
                error = (pred - target) ** 2
            else:
                error = torch.abs(pred - target)

            if self.reduction == 'mean':
                return error.mean()
            else:
                return error.sum()


def get_loss_function(loss_type: str = 'masked_mse', **kwargs):
    """
    Factory function to get loss function

    Args:
        loss_type: One of 'masked_mse', 'masked_mae', 'observed_only_mse', 'observed_only_mae', 'mse', 'mae'
        **kwargs: Additional arguments for the loss function

    Returns:
        Loss function instance
    """
    if loss_type == 'masked_mse':
        return MaskedMSELoss(**kwargs)
    elif loss_type == 'masked_mae':
        return MaskedMAELoss(**kwargs)
    elif loss_type == 'observed_only_mse':
        return ObservedOnlyLoss(loss_fn='mse', **kwargs)
    elif loss_type == 'observed_only_mae':
        return ObservedOnlyLoss(loss_fn='mae', **kwargs)
    elif loss_type == 'mse':
        return nn.MSELoss(**kwargs)
    elif loss_type == 'mae':
        return nn.L1Loss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
