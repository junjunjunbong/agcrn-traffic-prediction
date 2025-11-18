"""
Debug script to identify NaN/Inf issues in training

This script checks:
1. NaN/Inf in input data (x, y, mask)
2. NaN/Inf in model outputs
3. NaN/Inf in loss computation
4. Data statistics (mean, std, min, max)
"""
import torch
import numpy as np
from pathlib import Path

from src.config import (
    NUM_NODES, INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, NUM_LAYERS,
    CHEB_K, EMBED_DIM, BATCH_SIZE, LEARNING_RATE, DEVICE
)
from src.model_agcrn import AGCRN
from src.dataset import create_dataloaders
from src.losses import MaskedMSELoss


def check_tensor(tensor, name):
    """Check tensor for NaN/Inf and print statistics"""
    print(f"\n{'='*60}")
    print(f"Checking: {name}")
    print(f"{'='*60}")
    print(f"  Shape: {tensor.shape}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  device: {tensor.device}")

    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")

    if has_nan:
        nan_count = torch.isnan(tensor).sum().item()
        nan_pct = 100 * nan_count / tensor.numel()
        print(f"  NaN count: {nan_count:,} / {tensor.numel():,} ({nan_pct:.2f}%)")

    if has_inf:
        inf_count = torch.isinf(tensor).sum().item()
        print(f"  Inf count: {inf_count:,}")

    # Statistics (only for finite values)
    finite_mask = torch.isfinite(tensor)
    if finite_mask.any():
        finite_vals = tensor[finite_mask]
        print(f"  Mean (finite): {finite_vals.mean().item():.6f}")
        print(f"  Std (finite): {finite_vals.std().item():.6f}")
        print(f"  Min (finite): {finite_vals.min().item():.6f}")
        print(f"  Max (finite): {finite_vals.max().item():.6f}")
    else:
        print("  WARNING: No finite values!")

    return has_nan, has_inf


def check_processed_data(data_name='loops_035'):
    """Check if processed data files contain NaN/Inf"""
    from src.config import PROCESSED_DATA_DIR

    print(f"\n{'='*60}")
    print(f"Step 0: Checking processed data file")
    print(f"{'='*60}")

    npz_path = PROCESSED_DATA_DIR / f"{data_name}_processed.npz"
    if not npz_path.exists():
        print(f"ERROR: Processed data not found at {npz_path}")
        return False

    data = np.load(npz_path, allow_pickle=True)

    issues_found = False
    for key in ['train', 'val', 'test', 'mask_train', 'mask_val', 'mask_test']:
        if key in data:
            arr = data[key]
            has_nan = np.isnan(arr).any()
            has_inf = np.isinf(arr).any()

            print(f"\n  {key}:")
            print(f"    Shape: {arr.shape}")
            print(f"    Has NaN: {has_nan}")
            print(f"    Has Inf: {has_inf}")

            if has_nan:
                nan_count = np.isnan(arr).sum()
                nan_pct = 100 * nan_count / arr.size
                print(f"    NaN count: {nan_count:,} / {arr.size:,} ({nan_pct:.2f}%)")
                issues_found = True

            if has_inf:
                inf_count = np.isinf(arr).sum()
                print(f"    Inf count: {inf_count:,}")
                issues_found = True

            # Show statistics
            if not (has_nan or has_inf):
                print(f"    Mean: {np.mean(arr):.6f}")
                print(f"    Std: {np.std(arr):.6f}")
                print(f"    Min: {np.min(arr):.6f}")
                print(f"    Max: {np.max(arr):.6f}")

    if issues_found:
        print("\n  ⚠️  WARNING: Found NaN/Inf in processed data files!")
        print("  This is the root cause. Need to re-run preprocessing.")
        return False
    else:
        print("\n  ✓ Processed data files are clean (no NaN/Inf)")
        return True


def main():
    print("="*60)
    print("NaN/Inf Debug Script for AGCRN Training")
    print("="*60)

    data_name = 'loops_035'

    # Step 0: Check processed data files
    data_clean = check_processed_data(data_name)
    if not data_clean:
        print("\n" + "="*60)
        print("DIAGNOSIS: Processed data contains NaN/Inf")
        print("ACTION: Re-run preprocessing with proper validation")
        print("="*60)
        return

    # Step 1: Load first batch
    print(f"\n{'='*60}")
    print(f"Step 1: Loading first batch from DataLoader")
    print(f"{'='*60}")

    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_name=data_name,
            batch_size=BATCH_SIZE,
            use_masks=True
        )
    except Exception as e:
        print(f"ERROR loading dataloaders: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\nDataLoaders created successfully")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Get first batch
    try:
        x, y, masks = next(iter(train_loader))
    except Exception as e:
        print(f"ERROR getting first batch: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\nFirst batch loaded successfully")

    # Check input data
    issues_found = False

    x_has_nan, x_has_inf = check_tensor(x, "Input x")
    y_has_nan, y_has_inf = check_tensor(y, "Target y")

    if x_has_nan or x_has_inf or y_has_nan or y_has_inf:
        issues_found = True

    # Check masks
    if masks is not None:
        mask_x, mask_y = masks
        mx_has_nan, mx_has_inf = check_tensor(mask_x, "Mask x")
        my_has_nan, my_has_inf = check_tensor(mask_y, "Mask y")

        if mx_has_nan or mx_has_inf or my_has_nan or my_has_inf:
            issues_found = True

        # Check mask statistics
        print(f"\nMask statistics:")
        print(f"  mask_x sum: {mask_x.sum().item()}")
        print(f"  mask_y sum: {mask_y.sum().item()}")
        print(f"  mask_x observation rate: {mask_x.mean().item()*100:.2f}%")
        print(f"  mask_y observation rate: {mask_y.mean().item()*100:.2f}%")

        # WARNING: Check if mask is all zeros
        if mask_y.sum() == 0:
            print(f"  ⚠️  WARNING: mask_y is all zeros! This will cause division by zero!")
            issues_found = True

    if issues_found:
        print("\n" + "="*60)
        print("DIAGNOSIS: Input data contains NaN/Inf or invalid masks")
        print("ACTION: Check data preprocessing and dataset creation")
        print("="*60)
        return

    # Step 2: Test model forward pass
    print(f"\n{'='*60}")
    print(f"Step 2: Testing model forward pass")
    print(f"{'='*60}")

    try:
        model = AGCRN(
            num_nodes=NUM_NODES,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            cheb_k=CHEB_K,
            embed_dim=EMBED_DIM
        )
        model.to(DEVICE)
        model.eval()

        print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

        # Check if model parameters have NaN/Inf
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"  ⚠️  WARNING: Model parameter '{name}' has NaN/Inf!")
                issues_found = True

        if not issues_found:
            print(f"  ✓ Model parameters are clean")

    except Exception as e:
        print(f"ERROR creating model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Forward pass
    try:
        with torch.no_grad():
            x_device = x.to(DEVICE)
            output = model(x_device)

        output_has_nan, output_has_inf = check_tensor(output, "Model output")

        if output_has_nan or output_has_inf:
            issues_found = True
            print("\n  ⚠️  Model produces NaN/Inf output!")
            print("  This could be due to:")
            print("    - Unstable operations in the model")
            print("    - Extreme input values")
            print("    - Gradient explosion (but we're in eval mode)")

    except Exception as e:
        print(f"ERROR during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Test loss computation
    print(f"\n{'='*60}")
    print(f"Step 3: Testing loss computation")
    print(f"{'='*60}")

    try:
        criterion = MaskedMSELoss(imputed_weight=0.1)

        # Prepare targets
        y_target = y[:, -1, :, :]  # Last timestep
        y_target_device = y_target.to(DEVICE)

        # Prepare mask
        mask_target = None
        if masks is not None:
            _, mask_y = masks
            mask_target = mask_y[:, -1, :, :].to(DEVICE)

            print(f"\nMask for loss computation:")
            print(f"  Shape: {mask_target.shape}")
            print(f"  Sum: {mask_target.sum().item()}")
            print(f"  Mean: {mask_target.mean().item():.4f}")

            if mask_target.sum() == 0:
                print(f"  ⚠️  CRITICAL: Mask is all zeros! Loss computation will fail!")

        # Compute loss
        if output.shape[-1] == y_target.shape[-1]:
            loss = criterion(output, y_target_device, mask_target)
        else:
            # Assume predicting only first feature
            pred = output.squeeze(-1)
            target = y_target_device[:, :, 0]
            mask = mask_target[:, :, 0] if mask_target is not None else None
            loss = criterion(pred, target, mask)

        print(f"\nLoss value: {loss.item()}")

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  ⚠️  CRITICAL: Loss is NaN/Inf!")
            issues_found = True

            # Debug loss computation
            print("\n  Debugging loss computation:")
            if output.shape[-1] == y_target.shape[-1]:
                diff = output - y_target_device
                squared_error = diff ** 2
            else:
                pred = output.squeeze(-1)
                target = y_target_device[:, :, 0]
                diff = pred - target
                squared_error = diff ** 2

            print(f"    diff - has NaN: {torch.isnan(diff).any()}, has Inf: {torch.isinf(diff).any()}")
            print(f"    squared_error - has NaN: {torch.isnan(squared_error).any()}, has Inf: {torch.isinf(squared_error).any()}")

            if mask_target is not None:
                weights = mask_target + (1 - mask_target) * 0.1
                weighted_error = squared_error * weights if output.shape[-1] == y_target.shape[-1] else squared_error * weights[:, :, 0]
                total_weight = weights.sum() if output.shape[-1] == y_target.shape[-1] else weights[:, :, 0].sum()

                print(f"    weights - sum: {weights.sum().item():.4f}")
                print(f"    weighted_error - has NaN: {torch.isnan(weighted_error).any()}, has Inf: {torch.isinf(weighted_error).any()}")
                print(f"    total_weight: {total_weight.item():.4f}")
        else:
            print(f"  ✓ Loss is finite: {loss.item():.6f}")

    except Exception as e:
        print(f"ERROR computing loss: {e}")
        import traceback
        traceback.print_exc()
        return

    # Summary
    print("\n" + "="*60)
    if issues_found:
        print("❌ DIAGNOSIS: Issues found!")
        print("="*60)
        print("\nPossible causes:")
        print("  1. Input data contains NaN/Inf")
        print("  2. Mask is all zeros (division by zero)")
        print("  3. Model produces NaN/Inf output")
        print("  4. Loss computation has numerical instability")
        print("\nNext steps:")
        print("  1. Check preprocessing pipeline")
        print("  2. Verify mask creation logic")
        print("  3. Add gradient clipping and check model stability")
    else:
        print("✓ All checks passed!")
        print("="*60)
        print("\nThe first batch looks clean:")
        print("  - No NaN/Inf in input data")
        print("  - No NaN/Inf in model output")
        print("  - No NaN/Inf in loss computation")
        print("\nIf training still fails, possible causes:")
        print("  1. Later batches may have issues")
        print("  2. Gradient explosion during backprop")
        print("  3. Learning rate too high")
        print("\nRecommended actions:")
        print("  1. Add more aggressive gradient clipping")
        print("  2. Lower learning rate (try 1e-4 or 1e-5)")
        print("  3. Add loss.isnan() check in training loop")


if __name__ == "__main__":
    main()
