"""
Simple script to check processed data for NaN/Inf without requiring torch
"""
import numpy as np
from pathlib import Path

def main():
    print("="*60)
    print("Simple Data NaN/Inf Checker (NumPy only)")
    print("="*60)

    # Check if processed data exists
    data_dir = Path("data/processed")
    data_file = data_dir / "loops_035_processed.npz"

    if not data_file.exists():
        print(f"\nERROR: Data file not found: {data_file}")
        print("Please run preprocessing first: python preprocess.py")
        return

    print(f"\nLoading data from: {data_file}")

    try:
        data = np.load(data_file, allow_pickle=True)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return

    print(f"Data file loaded successfully")
    print(f"\nAvailable keys: {list(data.keys())}")

    # Check each array
    issues_found = False

    for key in ['train', 'val', 'test', 'mask_train', 'mask_val', 'mask_test']:
        if key not in data:
            print(f"\nWARNING: Key '{key}' not found in data")
            continue

        arr = data[key]

        print(f"\n{'='*60}")
        print(f"Checking: {key}")
        print(f"{'='*60}")
        print(f"  Shape: {arr.shape}")
        print(f"  dtype: {arr.dtype}")

        # Check for NaN
        has_nan = np.isnan(arr).any()
        print(f"  Has NaN: {has_nan}")

        if has_nan:
            nan_count = np.isnan(arr).sum()
            nan_pct = 100 * nan_count / arr.size
            print(f"  NaN count: {nan_count:,} / {arr.size:,} ({nan_pct:.2f}%)")
            issues_found = True

            # Show where NaN values are
            nan_positions = np.where(np.isnan(arr))
            print(f"  First few NaN positions (T, N, F):")
            for i in range(min(5, len(nan_positions[0]))):
                t_idx = nan_positions[0][i]
                n_idx = nan_positions[1][i]
                f_idx = nan_positions[2][i]
                print(f"    [{t_idx}, {n_idx}, {f_idx}]")

        # Check for Inf
        has_inf = np.isinf(arr).any()
        print(f"  Has Inf: {has_inf}")

        if has_inf:
            inf_count = np.isinf(arr).sum()
            print(f"  Inf count: {inf_count:,}")
            issues_found = True

            # Show where Inf values are
            inf_positions = np.where(np.isinf(arr))
            print(f"  First few Inf positions (T, N, F):")
            for i in range(min(5, len(inf_positions[0]))):
                t_idx = inf_positions[0][i]
                n_idx = inf_positions[1][i]
                f_idx = inf_positions[2][i]
                print(f"    [{t_idx}, {n_idx}, {f_idx}]")

        # Statistics (only if no NaN/Inf)
        if not (has_nan or has_inf):
            print(f"  Mean: {np.mean(arr):.6f}")
            print(f"  Std: {np.std(arr):.6f}")
            print(f"  Min: {np.min(arr):.6f}")
            print(f"  Max: {np.max(arr):.6f}")
        else:
            # Show stats for finite values only
            finite_mask = np.isfinite(arr)
            if finite_mask.any():
                finite_vals = arr[finite_mask]
                print(f"  Mean (finite only): {np.mean(finite_vals):.6f}")
                print(f"  Std (finite only): {np.std(finite_vals):.6f}")
                print(f"  Min (finite only): {np.min(finite_vals):.6f}")
                print(f"  Max (finite only): {np.max(finite_vals):.6f}")

    # Check stats if available
    if 'stats' in data:
        print(f"\n{'='*60}")
        print(f"Normalization Statistics:")
        print(f"{'='*60}")
        stats = data['stats'].item() if isinstance(data['stats'], np.ndarray) else data['stats']
        for feat_name, feat_stats in stats.items():
            print(f"\n  {feat_name}:")
            print(f"    mean: {feat_stats['mean']:.6f}")
            print(f"    std: {feat_stats['std']:.6f}")

            if np.isnan(feat_stats['mean']) or np.isnan(feat_stats['std']):
                print(f"    ⚠️  WARNING: Stats contain NaN!")
                issues_found = True
            if np.isinf(feat_stats['mean']) or np.isinf(feat_stats['std']):
                print(f"    ⚠️  WARNING: Stats contain Inf!")
                issues_found = True
            if feat_stats['std'] < 1e-10:
                print(f"    ⚠️  WARNING: Std is too small (near zero)!")
                issues_found = True

    # Summary
    print(f"\n{'='*60}")
    if issues_found:
        print("❌ DIAGNOSIS: Issues found in processed data!")
        print("="*60)
        print("\nThe processed data contains NaN/Inf values.")
        print("This explains why training produces NaN loss.")
        print("\nROOT CAUSE:")
        print("  Preprocessing pipeline did not properly handle missing values")
        print("\nSOLUTION:")
        print("  1. Re-run preprocessing: python preprocess.py")
        print("  2. Check that interpolation is working correctly")
        print("  3. Verify that std > 0 for all features")
        print("  4. Ensure no NaN values remain after interpolation")
    else:
        print("✓ All checks passed! Data is clean.")
        print("="*60)
        print("\nThe processed data does NOT contain NaN/Inf.")
        print("The issue must be elsewhere:")
        print("  - Model initialization")
        print("  - Forward pass computation")
        print("  - Loss computation")
        print("  - Gradient explosion during backprop")


if __name__ == "__main__":
    main()
