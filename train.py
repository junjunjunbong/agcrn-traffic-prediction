"""
Main training script for AGCRN traffic prediction
"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn

from src.config import (
    NUM_NODES, INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, NUM_LAYERS,
    CHEB_K, EMBED_DIM, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, DEVICE
)
from src.model_agcrn import AGCRN
from src.dataset import create_dataloaders
from src.trainer import Trainer
from src.losses import MaskedMSELoss, MaskedMAELoss, ObservedOnlyLoss


def main():
    parser = argparse.ArgumentParser(description='Train AGCRN model')
    parser.add_argument('--data', type=str, default='loops_035', help='Data name (e.g., loops_035)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--hidden_dim', type=int, default=HIDDEN_DIM, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS, help='Number of layers')
    parser.add_argument('--device', type=str, default=DEVICE, help='Device to use')
    parser.add_argument('--loss', type=str, default='masked_mse',
                        choices=['mse', 'masked_mse', 'masked_mae', 'observed_only'],
                        help='Loss function: mse (standard), masked_mse (weighted), masked_mae (MAE), observed_only (ignore imputed)')
    parser.add_argument('--imputed_weight', type=float, default=0.1,
                        help='Weight for imputed values in masked losses (0.0-1.0)')

    args = parser.parse_args()
    
    print("="*60)
    print("AGCRN Traffic Prediction - Training")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Loss function: {args.loss}")
    if args.loss in ['masked_mse', 'masked_mae']:
        print(f"Imputed weight: {args.imputed_weight}")
    print("="*60)

    # Select loss function
    if args.loss == 'mse':
        criterion = nn.MSELoss()
        print("\nUsing standard MSE loss (treats all values equally)")
    elif args.loss == 'masked_mse':
        criterion = MaskedMSELoss(imputed_weight=args.imputed_weight)
        print(f"\nUsing Masked MSE loss (imputed values weighted at {args.imputed_weight:.1%})")
    elif args.loss == 'masked_mae':
        criterion = MaskedMAELoss(imputed_weight=args.imputed_weight)
        print(f"\nUsing Masked MAE loss (imputed values weighted at {args.imputed_weight:.1%})")
    elif args.loss == 'observed_only':
        criterion = ObservedOnlyLoss()
        print("\nUsing Observed-Only loss (completely ignores imputed values)")
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")

    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_name=args.data,
        batch_size=args.batch_size
    )
    
    # Create model
    print("\nCreating model...")
    model = AGCRN(
        num_nodes=NUM_NODES,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        cheb_k=CHEB_K,
        embed_dim=EMBED_DIM
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        lr=args.lr,
        device=args.device
    )
    
    # Train
    print("\nStarting training...")
    history = trainer.train(num_epochs=args.epochs)
    
    print("\n" + "="*70)
    print("Training Completed!")
    print("="*70)
    print(f"\nBest Validation Metrics:")
    print(f"  Loss:       {history['best_val_loss']:.6f}")
    print(f"  MAE:        {history['best_val_mae']:.6f}")
    print(f"  RMSE:       {history['best_val_rmse']:.6f}")
    print(f"  MAPE:       {history['best_val_mape']:.2f}%")
    print("\n" + "="*70)
    print(f"Total epochs trained: {len(history['train_losses'])}")
    print("="*70)


if __name__ == "__main__":
    main()

