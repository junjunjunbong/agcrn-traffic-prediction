"""
Data preprocessing module for traffic loop detector data
Converts CSV files to (T, N, F) tensors suitable for AGCRN
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, META_DATA_DIR,
    NODE_MODE, FEATURES, TIME_STEP_SIZE,
    MISSING_SPEED_VALUE, FREE_FLOW_SPEED, CONGESTED_SPEED,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)


def load_csv_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV file and return DataFrame"""
    print(f"Loading {csv_path.name}...")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape}")
    return df


def create_node_mapping(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Create node mapping from raw_id or det_pos
    
    Args:
        df: Input DataFrame
        
    Returns:
        sensors_df: DataFrame with node metadata
        node_to_idx: Dictionary mapping node_id to index
    """
    if NODE_MODE == "raw_id":
        # Use raw_id as nodes (480 nodes)
        unique_nodes = df['raw_id'].unique()
        sensors_df = df[['raw_id', 'det_pos', 'lane_idx', 'edge_id']].drop_duplicates('raw_id')
        sensors_df = sensors_df.sort_values('raw_id').reset_index(drop=True)
        sensors_df['node_idx'] = range(len(sensors_df))
        node_to_idx = {node_id: idx for idx, node_id in enumerate(sensors_df['raw_id'])}
        
    elif NODE_MODE == "det_pos":
        # Aggregate by det_pos (160 nodes)
        sensors_df = df[['det_pos', 'edge_id']].drop_duplicates('det_pos')
        sensors_df = sensors_df.sort_values('det_pos').reset_index(drop=True)
        sensors_df['node_idx'] = range(len(sensors_df))
        node_to_idx = {f"det_pos_{pos}": idx for idx, pos in enumerate(sensors_df['det_pos'])}
        
    else:
        raise ValueError(f"Unknown NODE_MODE: {NODE_MODE}")
    
    print(f"  Created {len(sensors_df)} nodes using {NODE_MODE} mode")
    return sensors_df, node_to_idx


def create_time_index(df: pd.DataFrame) -> np.ndarray:
    """
    Create time index from begin/end columns
    
    Returns:
        time_steps: Array of time step indices (0, 1, 2, ...)
    """
    unique_times = sorted(df['begin'].unique())
    num_steps = len(unique_times)
    print(f"  Found {num_steps} time steps")
    return np.arange(num_steps)


def convert_to_tensor(
    df: pd.DataFrame,
    node_to_idx: Dict[str, int],
    time_steps: np.ndarray
) -> np.ndarray:
    """
    Convert DataFrame to (T, N, F) tensor
    
    Args:
        df: Input DataFrame
        node_to_idx: Node ID to index mapping
        time_steps: Time step indices
        
    Returns:
        X: Tensor of shape (T, N, F)
    """
    T = len(time_steps)
    N = len(node_to_idx)
    F = len(FEATURES)
    
    X = np.full((T, N, F), np.nan, dtype=np.float32)
    
    # Map time to index
    time_to_idx = {t: idx for idx, t in enumerate(sorted(df['begin'].unique()))}
    
    print(f"  Converting to tensor shape ({T}, {N}, {F})...")
    
    for _, row in df.iterrows():
        if NODE_MODE == "raw_id":
            node_id = row['raw_id']
        else:
            node_id = f"det_pos_{row['det_pos']}"
        
        if node_id not in node_to_idx:
            continue
            
        n_idx = node_to_idx[node_id]
        t_idx = time_to_idx[row['begin']]
        
        for f_idx, feat_name in enumerate(FEATURES):
            value = row[feat_name]
            if pd.notna(value) and value != '':
                try:
                    val = float(value)
                    if feat_name == 'harmonicMeanSpeed' and val == MISSING_SPEED_VALUE:
                        X[t_idx, n_idx, f_idx] = np.nan
                    else:
                        X[t_idx, n_idx, f_idx] = val
                except (ValueError, TypeError):
                    X[t_idx, n_idx, f_idx] = np.nan
    
    print(f"  Tensor created. Missing values: {np.isnan(X).sum()} / {X.size}")
    return X


def interpolate_missing_speed(X: np.ndarray) -> np.ndarray:
    """
    Interpolate missing speed values (-1 or NaN)
    
    Strategy:
    1. Time-direction linear interpolation for each node
    2. If still missing, use flow/occupancy to infer:
       - Low flow & occupancy -> free flow speed
       - High occupancy & low flow -> congested speed
    """
    X_interp = X.copy()
    speed_idx = FEATURES.index('harmonicMeanSpeed')
    flow_idx = FEATURES.index('flow')
    occ_idx = FEATURES.index('occupancy')
    
    print("  Interpolating missing speed values...")
    
    for n in range(X.shape[1]):
        speed_series = X[:, n, speed_idx]
        flow_series = X[:, n, flow_idx]
        occ_series = X[:, n, occ_idx]
        
        # Step 1: Time-direction linear interpolation
        if np.any(~np.isnan(speed_series)):
            # Use pandas interpolate for time-series interpolation
            speed_df = pd.Series(speed_series)
            speed_interp = speed_df.interpolate(method='linear', limit_direction='both')
            X_interp[:, n, speed_idx] = speed_interp.values
        
        # Step 2: Fill remaining NaN using flow/occupancy heuristics
        nan_mask = np.isnan(X_interp[:, n, speed_idx])
        if np.any(nan_mask):
            for t in np.where(nan_mask)[0]:
                flow_val = X[t, n, flow_idx] if not np.isnan(X[t, n, flow_idx]) else 0
                occ_val = X[t, n, occ_idx] if not np.isnan(X[t, n, occ_idx]) else 0
                
                if flow_val < 0.1 and occ_val < 0.1:
                    # No vehicles -> free flow
                    X_interp[t, n, speed_idx] = FREE_FLOW_SPEED
                elif occ_val > 0.3 and flow_val < 0.5:
                    # Congested
                    X_interp[t, n, speed_idx] = CONGESTED_SPEED
                else:
                    # Use mean of non-NaN values or default
                    valid_speeds = speed_series[~np.isnan(speed_series)]
                    if len(valid_speeds) > 0:
                        X_interp[t, n, speed_idx] = np.mean(valid_speeds)
                    else:
                        X_interp[t, n, speed_idx] = FREE_FLOW_SPEED
    
    remaining_nan = np.isnan(X_interp[:, :, speed_idx]).sum()
    print(f"  Remaining NaN after interpolation: {remaining_nan}")
    
    return X_interp


def normalize_data(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Normalize data using z-score normalization based on training statistics
    
    Returns:
        X_train_norm, X_val_norm, X_test_norm: Normalized tensors
        stats: Dictionary with mean and std for each feature
    """
    print("  Normalizing data...")
    
    stats = {}
    X_train_norm = X_train.copy()
    X_val_norm = X_val.copy()
    X_test_norm = X_test.copy()
    
    for f_idx, feat_name in enumerate(FEATURES):
        train_feat = X_train[:, :, f_idx]
        mean = np.nanmean(train_feat)
        std = np.nanstd(train_feat)
        
        if std < 1e-6:
            std = 1.0
        
        stats[feat_name] = {'mean': mean, 'std': std}
        
        # Normalize
        X_train_norm[:, :, f_idx] = (X_train[:, :, f_idx] - mean) / std
        X_val_norm[:, :, f_idx] = (X_val[:, :, f_idx] - mean) / std
        X_test_norm[:, :, f_idx] = (X_test[:, :, f_idx] - mean) / std
        
        print(f"    {feat_name}: mean={mean:.4f}, std={std:.4f}")
    
    return X_train_norm, X_val_norm, X_test_norm, stats


def split_data(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/val/test sets along time axis
    
    Returns:
        X_train, X_val, X_test: Split tensors
    """
    T = X.shape[0]
    train_end = int(T * TRAIN_RATIO)
    val_end = int(T * (TRAIN_RATIO + VAL_RATIO))
    
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]
    
    print(f"  Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    return X_train, X_val, X_test


def process_single_file(csv_path: Path) -> Tuple[str, Dict]:
    """
    Process a single CSV file
    
    Returns:
        output_name: Name for output files
        metadata: Dictionary with processing metadata
    """
    print(f"\n{'='*60}")
    print(f"Processing {csv_path.name}")
    print(f"{'='*60}")
    
    # Load data
    df = load_csv_data(csv_path)
    
    # Create node mapping
    sensors_df, node_to_idx = create_node_mapping(df)
    
    # Create time index
    time_steps = create_time_index(df)
    
    # Convert to tensor
    X = convert_to_tensor(df, node_to_idx, time_steps)
    
    # Handle missing speed values
    X = interpolate_missing_speed(X)
    
    # Split data
    X_train, X_val, X_test = split_data(X)
    
    # Normalize
    X_train_norm, X_val_norm, X_test_norm, stats = normalize_data(X_train, X_val, X_test)
    
    # Save processed data
    output_name = csv_path.stem.replace('loops', 'loops_')
    np.savez(
        PROCESSED_DATA_DIR / f"{output_name}_processed.npz",
        train=X_train_norm,
        val=X_val_norm,
        test=X_test_norm,
        stats=stats
    )
    
    # Save metadata
    sensors_df.to_csv(META_DATA_DIR / f"{output_name}_sensors.csv", index=False)
    
    metadata = {
        'num_nodes': len(node_to_idx),
        'num_time_steps': len(time_steps),
        'num_features': len(FEATURES),
        'train_shape': X_train_norm.shape,
        'val_shape': X_val_norm.shape,
        'test_shape': X_test_norm.shape,
        'stats': stats
    }
    
    print(f"\n✓ Processing complete: {output_name}")
    print(f"  Final shapes - Train: {X_train_norm.shape}, Val: {X_val_norm.shape}, Test: {X_test_norm.shape}")
    
    return output_name, metadata


def main():
    """Main preprocessing function"""
    print("="*60)
    print("AGCRN Traffic Data Preprocessing")
    print("="*60)
    
    csv_files = list(RAW_DATA_DIR.glob("loops*.csv"))
    if not csv_files:
        print(f"No CSV files found in {RAW_DATA_DIR}")
        return
    
    all_metadata = {}
    
    for csv_path in csv_files:
        output_name, metadata = process_single_file(csv_path)
        all_metadata[output_name] = metadata
    
    # Save summary
    import json
    with open(PROCESSED_DATA_DIR / "preprocessing_summary.json", "w") as f:
        json.dump(all_metadata, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("All files processed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

