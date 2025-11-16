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
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for x, y in pbar:
            x = x.to(self.device)  # (batch, T_in, N, F)
            y = y.to(self.device)  # (batch, T_out, N, F)
            
            # For now, predict only the last time step
            # TODO: Support multi-step prediction
            y_target = y[:, -1, :, :]  # (batch, N, F) - last time step
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(x)  # (batch, N, output_dim)
            
            # If output_dim != F, we need to select features
            if output.shape[-1] == y_target.shape[-1]:
                loss = self.criterion(output, y_target)
            else:
                # Assume predicting only speed (first feature)
                loss = self.criterion(output.squeeze(-1), y_target[:, :, 0])
            
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
    
    def validate(self) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc="Validation"):
                x = x.to(self.device)
                y = y.to(self.device)
                
                y_target = y[:, -1, :, :]
                
                output = self.model(x)
                
                if output.shape[-1] == y_target.shape[-1]:
                    loss = self.criterion(output, y_target)
                else:
                    loss = self.criterion(output.squeeze(-1), y_target[:, :, 0])
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
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
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping and model saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }, save_path)
                print(f"✓ Saved best model (val_loss: {val_loss:.6f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        history_path = LOGS_DIR / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        return history

