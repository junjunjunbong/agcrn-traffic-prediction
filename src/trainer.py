"""
Training module for AGCRN model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm
import json
import copy
import matplotlib.pyplot as plt

from src.config import SAVED_MODELS_DIR, LOGS_DIR, DEVICE, PATIENCE
from src.model_agcrn import AGCRN


class Trainer:
    """
    Trainer class for AGCRN model
    """
    def __init__(
        self,
        model: AGCRN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module = nn.MSELoss(),
        optimizer: Optional[optim.Optimizer] = None,
        lr: float = 0.001,
        device: str = DEVICE
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
        
        self.train_losses = []
        self.val_losses = []
        self.val_maes = []
        self.val_rmses = []
        self.val_mapes = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_data in pbar:
            # Unpack batch (x, y, masks)
            x, y, masks = batch_data
            x = x.to(self.device)  # (batch, T_in, N, F)
            y = y.to(self.device)  # (batch, T_out, N, F)

            # For now, predict only the last time step
            # TODO: Support multi-step prediction
            y_target = y[:, -1, :, :]  # (batch, N, F) - last time step

            # Extract corresponding mask if available
            mask_target = None
            if masks is not None:
                _, mask_y = masks
                mask_y = mask_y.to(self.device)
                mask_target = mask_y[:, -1, :, :]  # (batch, N, F) - last time step

            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(x)  # (batch, N, output_dim)

            # Compute loss with mask support
            if output.shape[-1] == y_target.shape[-1]:
                # Full feature prediction
                if hasattr(self.criterion, 'forward') and 'mask' in self.criterion.forward.__code__.co_varnames:
                    loss = self.criterion(output, y_target, mask_target)
                else:
                    loss = self.criterion(output, y_target)
            else:
                # Assume predicting only speed (first feature)
                pred = output.squeeze(-1)
                target = y_target[:, :, 0]
                mask = mask_target[:, :, 0] if mask_target is not None else None

                if hasattr(self.criterion, 'forward') and 'mask' in self.criterion.forward.__code__.co_varnames:
                    loss = self.criterion(pred, target, mask)
                else:
                    loss = self.criterion(pred, target)

            # Check for NaN/Inf loss BEFORE backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️  NaN/Inf loss detected at batch {num_batches}!")
                print(f"  Loss value: {loss.item()}")
                print(f"  Output stats: min={output.min().item():.6f}, max={output.max().item():.6f}, mean={output.mean().item():.6f}")
                print(f"  Target stats: min={y_target.min().item():.6f}, max={y_target.max().item():.6f}, mean={y_target.mean().item():.6f}")
                if mask_target is not None:
                    print(f"  Mask sum: {mask_target.sum().item()}")
                print(f"  Skipping this batch and continuing...")
                continue

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model and compute comprehensive metrics

        Returns:
            metrics: Dictionary with loss, mae, rmse, mape
        """
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_mse = 0.0
        total_mape = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Validation"):
                # Unpack batch (x, y, masks)
                x, y, masks = batch_data
                x = x.to(self.device)
                y = y.to(self.device)

                y_target = y[:, -1, :, :]

                # Extract corresponding mask if available
                mask_target = None
                if masks is not None:
                    _, mask_y = masks
                    mask_y = mask_y.to(self.device)
                    mask_target = mask_y[:, -1, :, :]

                output = self.model(x)

                # Compute loss with mask support
                if output.shape[-1] == y_target.shape[-1]:
                    pred = output
                    target = y_target
                    if hasattr(self.criterion, 'forward') and 'mask' in self.criterion.forward.__code__.co_varnames:
                        loss = self.criterion(output, y_target, mask_target)
                    else:
                        loss = self.criterion(output, y_target)
                else:
                    pred = output.squeeze(-1)
                    target = y_target[:, :, 0]
                    mask = mask_target[:, :, 0] if mask_target is not None else None

                    if hasattr(self.criterion, 'forward') and 'mask' in self.criterion.forward.__code__.co_varnames:
                        loss = self.criterion(pred, target, mask)
                    else:
                        loss = self.criterion(pred, target)

                # Check for NaN/Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n⚠️  NaN/Inf loss detected in validation at batch {num_batches}!")
                    print(f"  Skipping this batch...")
                    continue

                # Calculate additional metrics (MAE, MSE for RMSE, MAPE)
                mae = torch.mean(torch.abs(pred - target))
                mse = torch.mean((pred - target) ** 2)

                # MAPE (avoid division by zero)
                mask_nonzero = target != 0
                if mask_nonzero.sum() > 0:
                    mape = torch.mean(torch.abs((pred[mask_nonzero] - target[mask_nonzero]) / target[mask_nonzero])) * 100
                else:
                    mape = torch.tensor(0.0)

                total_loss += loss.item()
                total_mae += mae.item()
                total_mse += mse.item()
                total_mape += mape.item()
                num_batches += 1

        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        avg_rmse = np.sqrt(total_mse / num_batches)
        avg_mape = total_mape / num_batches

        # Store in history
        self.val_losses.append(avg_loss)

        metrics = {
            'loss': avg_loss,
            'mae': avg_mae,
            'rmse': avg_rmse,
            'mape': avg_mape
        }

        return metrics
    
    def train(self, num_epochs: int, save_path: Optional[Path] = None) -> Dict:
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
            save_path: Path to save best model
            
        Returns:
            training_history: Dictionary with training metrics
        """
        if save_path is None:
            save_path = SAVED_MODELS_DIR / "best_agcrn.pt"
        
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 70)

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_metrics = self.validate()
            val_loss = val_metrics['loss']

            # Store metrics in history
            self.val_maes.append(val_metrics['mae'])
            self.val_rmses.append(val_metrics['rmse'])
            self.val_mapes.append(val_metrics['mape'])

            # Print comprehensive metrics
            print(f"\n{'Metric':<15} {'Train':<15} {'Validation':<15}")
            print("-" * 45)
            print(f"{'Loss':<15} {train_loss:<15.6f} {val_loss:<15.6f}")
            print(f"{'MAE':<15} {'-':<15} {val_metrics['mae']:<15.6f}")
            print(f"{'RMSE':<15} {'-':<15} {val_metrics['rmse']:<15.6f}")
            print(f"{'MAPE (%)':<15} {'-':<15} {val_metrics['mape']:<15.2f}")
            print("-" * 70)

            # Early stopping and model saving
            if val_loss < self.best_val_loss:
                improvement = ((self.best_val_loss - val_loss) / self.best_val_loss) * 100
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_mae': val_metrics['mae'],
                    'val_rmse': val_metrics['rmse'],
                    'val_mape': val_metrics['mape'],
                    'train_loss': train_loss,
                }, save_path)
                print(f"[OK] Model improved by {improvement:.2f}% - Saved to {save_path}")
            else:
                self.patience_counter += 1
                print(f"[INFO] No improvement for {self.patience_counter}/{PATIENCE} epochs")
                if self.patience_counter >= PATIENCE:
                    print(f"\n[EARLY STOP] Stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                    break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_maes': self.val_maes,
            'val_rmses': self.val_rmses,
            'val_mapes': self.val_mapes,
            'best_val_loss': self.best_val_loss,
            'best_val_mae': min(self.val_maes) if self.val_maes else float('inf'),
            'best_val_rmse': min(self.val_rmses) if self.val_rmses else float('inf'),
            'best_val_mape': min(self.val_mapes) if self.val_mapes else float('inf')
        }

        history_path = LOGS_DIR / "training_history.json"
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"\n[INFO] Training history saved to {history_path}")

        # Plot training curves
        self._plot_training_curves(history, LOGS_DIR / "training_curves.png")

        return history

    def _plot_training_curves(self, history: Dict, save_path: Path):
        """
        Plot training curves for loss and metrics

        Args:
            history: Training history dictionary
            save_path: Path to save the plot
        """
        epochs = range(1, len(history['train_losses']) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Loss
        axes[0, 0].plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_losses'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: MAE
        axes[0, 1].plot(epochs, history['val_maes'], 'g-', label='Val MAE', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('MAE', fontsize=12)
        axes[0, 1].set_title('Validation MAE', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: RMSE
        axes[1, 0].plot(epochs, history['val_rmses'], 'm-', label='Val RMSE', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('RMSE', fontsize=12)
        axes[1, 0].set_title('Validation RMSE', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: MAPE
        axes[1, 1].plot(epochs, history['val_mapes'], 'c-', label='Val MAPE', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('MAPE (%)', fontsize=12)
        axes[1, 1].set_title('Validation MAPE', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[INFO] Training curves saved to {save_path}")

        return history

