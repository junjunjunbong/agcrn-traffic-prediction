"""
Validation Analysis Module for AGCRN Traffic Prediction

This module provides:
1. Heatmap visualization (prediction vs actual)
2. Node-wise error distribution heatmap
3. Loss convergence analysis
4. Comprehensive validation report generation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class ConvergenceResult:
    """Result of convergence analysis"""
    is_converged: bool
    convergence_epoch: Optional[int]
    final_loss: float
    best_loss: float
    loss_reduction_percent: float
    is_stable: bool
    stability_score: float  # 0-1, higher is more stable
    trend: str  # 'decreasing', 'stable', 'oscillating', 'diverging'
    recommendations: List[str]


def analyze_convergence(
    train_losses: List[float],
    val_losses: List[float],
    window_size: int = 5,
    stability_threshold: float = 0.01,
    min_improvement: float = 0.001
) -> ConvergenceResult:
    """
    Analyze whether training has converged properly.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        window_size: Window size for moving average (default: 5)
        stability_threshold: Max relative change to consider stable (default: 1%)
        min_improvement: Minimum improvement to consider meaningful (default: 0.1%)

    Returns:
        ConvergenceResult with detailed analysis
    """
    if len(val_losses) < 3:
        return ConvergenceResult(
            is_converged=False,
            convergence_epoch=None,
            final_loss=val_losses[-1] if val_losses else float('inf'),
            best_loss=min(val_losses) if val_losses else float('inf'),
            loss_reduction_percent=0.0,
            is_stable=False,
            stability_score=0.0,
            trend='insufficient_data',
            recommendations=['Need more epochs to analyze convergence']
        )

    val_losses = np.array(val_losses)
    train_losses = np.array(train_losses)

    # Basic metrics
    initial_loss = val_losses[0]
    final_loss = val_losses[-1]
    best_loss = np.min(val_losses)
    best_epoch = int(np.argmin(val_losses)) + 1
    loss_reduction = (initial_loss - best_loss) / initial_loss * 100 if initial_loss > 0 else 0

    # Calculate moving average for stability analysis
    def moving_average(arr, window):
        if len(arr) < window:
            return arr
        return np.convolve(arr, np.ones(window)/window, mode='valid')

    val_ma = moving_average(val_losses, window_size)

    # Analyze trend
    if len(val_ma) >= 2:
        recent_changes = np.diff(val_ma[-min(5, len(val_ma)):])
        avg_change = np.mean(recent_changes)
        change_std = np.std(recent_changes)

        if avg_change < -min_improvement:
            trend = 'decreasing'
        elif abs(avg_change) < stability_threshold * np.mean(val_ma[-5:]):
            trend = 'stable'
        elif change_std > abs(avg_change) * 2:
            trend = 'oscillating'
        else:
            trend = 'diverging'
    else:
        trend = 'insufficient_data'

    # Calculate stability score
    if len(val_losses) >= window_size:
        recent_losses = val_losses[-window_size:]
        relative_std = np.std(recent_losses) / np.mean(recent_losses) if np.mean(recent_losses) > 0 else 1
        stability_score = max(0, 1 - relative_std * 10)  # Scale to 0-1
    else:
        stability_score = 0.0

    is_stable = bool(stability_score > 0.7)

    # Check for convergence
    convergence_epoch = None
    is_converged = False

    # Method 1: Check if recent losses are stable around the minimum
    if len(val_losses) >= window_size:
        recent_mean = np.mean(val_losses[-window_size:])
        if abs(recent_mean - best_loss) / best_loss < stability_threshold * 2:
            is_converged = True
            # Find first epoch where loss got close to best
            for i, loss in enumerate(val_losses):
                if abs(loss - best_loss) / best_loss < stability_threshold * 3:
                    convergence_epoch = i + 1
                    break

    # Method 2: Check if improvement has stalled
    if not is_converged and len(val_losses) >= 10:
        last_10_min = np.min(val_losses[-10:])
        prev_10_min = np.min(val_losses[:-10]) if len(val_losses) > 10 else float('inf')
        if (prev_10_min - last_10_min) / prev_10_min < min_improvement:
            is_converged = True
            convergence_epoch = len(val_losses) - 10

    # Generate recommendations
    recommendations = []

    if not is_converged:
        if trend == 'decreasing':
            recommendations.append('Loss is still decreasing - continue training')
        elif trend == 'oscillating':
            recommendations.append('Loss is oscillating - try reducing learning rate')
            recommendations.append('Consider using learning rate scheduler')
        elif trend == 'diverging':
            recommendations.append('Loss is increasing - possible overfitting or learning rate too high')
            recommendations.append('Try regularization or reduce learning rate')
    else:
        if stability_score < 0.5:
            recommendations.append('Converged but unstable - consider longer training or lower learning rate')
        else:
            recommendations.append('Training converged successfully')

    # Check for overfitting
    if len(train_losses) >= 5 and len(val_losses) >= 5:
        train_recent = np.mean(train_losses[-5:])
        val_recent = np.mean(val_losses[-5:])
        if val_recent > train_recent * 1.5:
            recommendations.append('Possible overfitting detected (val_loss >> train_loss)')
            recommendations.append('Consider: dropout, early stopping, or more training data')

    return ConvergenceResult(
        is_converged=is_converged,
        convergence_epoch=convergence_epoch,
        final_loss=float(final_loss),
        best_loss=float(best_loss),
        loss_reduction_percent=float(loss_reduction),
        is_stable=is_stable,
        stability_score=float(stability_score),
        trend=trend,
        recommendations=recommendations
    )


def plot_prediction_heatmap(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[Path] = None,
    time_steps: int = 50,
    title_prefix: str = ""
) -> plt.Figure:
    """
    Plot prediction vs actual heatmaps for visual comparison.

    Args:
        predictions: (num_samples, num_nodes, features) or (num_samples, num_nodes)
        targets: Same shape as predictions
        save_path: Path to save the figure
        time_steps: Number of time steps to display
        title_prefix: Prefix for titles

    Returns:
        matplotlib Figure
    """
    # Handle different input shapes
    if predictions.ndim == 4:
        # (num_samples, num_nodes, features, extra_dim)
        pred = predictions[:time_steps, :, 0, 0]
        target = targets[:time_steps, :, 0] if targets.ndim == 3 else targets[:time_steps, :, 0, 0]
    elif predictions.ndim == 3:
        # (num_samples, num_nodes, features)
        pred = predictions[:time_steps, :, 0]
        target = targets[:time_steps, :, 0] if targets.ndim == 3 else targets[:time_steps, :]
    else:
        # (num_samples, num_nodes)
        pred = predictions[:time_steps, :]
        target = targets[:time_steps, :]

    # Calculate error
    error = np.abs(pred - target)

    # Determine common scale for pred and target
    vmin = min(pred.min(), target.min())
    vmax = max(pred.max(), target.max())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Custom colormap for traffic (green = low, yellow = medium, red = high)
    traffic_cmap = LinearSegmentedColormap.from_list(
        'traffic', ['#2ecc71', '#f1c40f', '#e74c3c']
    )

    # Plot 1: Target (Actual)
    im1 = axes[0, 0].imshow(target.T, aspect='auto', cmap=traffic_cmap, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'{title_prefix}Actual Traffic (Ground Truth)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Time Step', fontsize=12)
    axes[0, 0].set_ylabel('Node (Sensor)', fontsize=12)
    plt.colorbar(im1, ax=axes[0, 0], label='Traffic Value')

    # Plot 2: Prediction
    im2 = axes[0, 1].imshow(pred.T, aspect='auto', cmap=traffic_cmap, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'{title_prefix}Predicted Traffic', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Time Step', fontsize=12)
    axes[0, 1].set_ylabel('Node (Sensor)', fontsize=12)
    plt.colorbar(im2, ax=axes[0, 1], label='Traffic Value')

    # Plot 3: Absolute Error
    im3 = axes[1, 0].imshow(error.T, aspect='auto', cmap='Reds')
    axes[1, 0].set_title(f'{title_prefix}Absolute Error |Pred - Actual|', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Time Step', fontsize=12)
    axes[1, 0].set_ylabel('Node (Sensor)', fontsize=12)
    plt.colorbar(im3, ax=axes[1, 0], label='Error')

    # Plot 4: Error Distribution per Node
    node_mae = np.mean(error, axis=0)  # Average error per node
    node_std = np.std(error, axis=0)

    x_nodes = np.arange(len(node_mae))
    axes[1, 1].bar(x_nodes, node_mae, yerr=node_std, capsize=2, alpha=0.7, color='steelblue')
    axes[1, 1].axhline(y=np.mean(node_mae), color='red', linestyle='--', label=f'Mean MAE: {np.mean(node_mae):.4f}')
    axes[1, 1].set_title(f'{title_prefix}Error Distribution by Node', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Node ID', fontsize=12)
    axes[1, 1].set_ylabel('MAE', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Prediction heatmap saved to {save_path}")

    return fig


def plot_error_distribution_heatmap(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[Path] = None,
    title_prefix: str = ""
) -> plt.Figure:
    """
    Plot detailed error distribution analysis as heatmaps.

    Args:
        predictions: (num_samples, num_nodes, features) or (num_samples, num_nodes)
        targets: Same shape as predictions
        save_path: Path to save the figure
        title_prefix: Prefix for titles

    Returns:
        matplotlib Figure
    """
    # Handle different input shapes
    if predictions.ndim == 4:
        # (num_samples, num_nodes, features, extra_dim)
        pred = predictions[:, :, 0, 0]
        target = targets[:, :, 0] if targets.ndim == 3 else targets[:, :, 0, 0]
    elif predictions.ndim == 3:
        # (num_samples, num_nodes, features)
        pred = predictions[:, :, 0]
        target = targets[:, :, 0] if targets.ndim == 3 else targets[:, :]
    else:
        # (num_samples, num_nodes)
        pred = predictions
        target = targets

    error = pred - target
    abs_error = np.abs(error)

    num_nodes = pred.shape[1]
    num_samples = pred.shape[0]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Error over time (heatmap)
    im1 = axes[0, 0].imshow(error.T, aspect='auto', cmap='RdBu_r',
                            vmin=-np.percentile(np.abs(error), 95),
                            vmax=np.percentile(np.abs(error), 95))
    axes[0, 0].set_title(f'{title_prefix}Signed Error (Pred - Actual)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Node')
    plt.colorbar(im1, ax=axes[0, 0], label='Error')

    # Plot 2: Node-wise MAE
    node_mae = np.mean(abs_error, axis=0)
    colors = plt.cm.RdYlGn_r(node_mae / node_mae.max())
    axes[0, 1].bar(range(num_nodes), node_mae, color=colors)
    axes[0, 1].axhline(y=np.mean(node_mae), color='black', linestyle='--',
                       label=f'Overall MAE: {np.mean(node_mae):.4f}')
    axes[0, 1].set_title(f'{title_prefix}MAE by Node', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Node ID')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()

    # Plot 3: Error histogram
    axes[0, 2].hist(error.flatten(), bins=50, density=True, alpha=0.7, color='steelblue')
    axes[0, 2].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 2].axvline(x=np.mean(error), color='orange', linestyle='-',
                       label=f'Mean: {np.mean(error):.4f}')
    axes[0, 2].set_title(f'{title_prefix}Error Distribution', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Error (Pred - Actual)')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].legend()

    # Plot 4: Node-wise RMSE
    node_rmse = np.sqrt(np.mean(error**2, axis=0))
    axes[1, 0].bar(range(num_nodes), node_rmse, color='coral', alpha=0.7)
    axes[1, 0].axhline(y=np.mean(node_rmse), color='black', linestyle='--',
                       label=f'Overall RMSE: {np.mean(node_rmse):.4f}')
    axes[1, 0].set_title(f'{title_prefix}RMSE by Node', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Node ID')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].legend()

    # Plot 5: Correlation scatter (sampled for visibility)
    sample_size = min(5000, pred.size)
    idx = np.random.choice(pred.size, sample_size, replace=False)
    pred_flat = pred.flatten()[idx]
    target_flat = target.flatten()[idx]

    axes[1, 1].scatter(target_flat, pred_flat, alpha=0.3, s=1)
    lims = [min(target_flat.min(), pred_flat.min()), max(target_flat.max(), pred_flat.max())]
    axes[1, 1].plot(lims, lims, 'r--', label='Perfect Prediction')

    # Calculate R2
    correlation = np.corrcoef(target_flat, pred_flat)[0, 1]
    r2 = correlation ** 2

    axes[1, 1].set_title(f'{title_prefix}Prediction vs Actual (R2={r2:.4f})', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Actual')
    axes[1, 1].set_ylabel('Predicted')
    axes[1, 1].legend()
    axes[1, 1].set_aspect('equal')

    # Plot 6: MAPE by Node (percentage error)
    mask = target != 0
    mape_per_node = []
    for i in range(num_nodes):
        node_mask = target[:, i] != 0
        if node_mask.sum() > 0:
            node_mape = np.mean(np.abs(error[:, i][node_mask] / target[:, i][node_mask])) * 100
        else:
            node_mape = 0
        mape_per_node.append(node_mape)

    colors = plt.cm.RdYlGn_r(np.array(mape_per_node) / max(mape_per_node) if max(mape_per_node) > 0 else 0)
    axes[1, 2].bar(range(num_nodes), mape_per_node, color=colors)
    axes[1, 2].axhline(y=np.mean(mape_per_node), color='black', linestyle='--',
                       label=f'Overall MAPE: {np.mean(mape_per_node):.2f}%')
    axes[1, 2].set_title(f'{title_prefix}MAPE by Node (%)', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Node ID')
    axes[1, 2].set_ylabel('MAPE (%)')
    axes[1, 2].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Error distribution heatmap saved to {save_path}")

    return fig


def plot_convergence_analysis(
    train_losses: List[float],
    val_losses: List[float],
    convergence_result: ConvergenceResult,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot detailed convergence analysis.

    Args:
        train_losses: Training losses
        val_losses: Validation losses
        convergence_result: Result from analyze_convergence()
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Loss curves with annotations
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)

    # Mark best epoch
    best_epoch = val_losses.index(min(val_losses)) + 1
    axes[0, 0].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7,
                        label=f'Best Epoch: {best_epoch}')
    axes[0, 0].scatter([best_epoch], [min(val_losses)], color='green', s=100, zorder=5)

    # Mark convergence epoch if detected
    if convergence_result.convergence_epoch:
        axes[0, 0].axvline(x=convergence_result.convergence_epoch, color='purple',
                           linestyle=':', alpha=0.7, label=f'Converged: Epoch {convergence_result.convergence_epoch}')

    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Progress', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Loss reduction analysis
    val_losses_arr = np.array(val_losses)
    initial = val_losses_arr[0]
    reduction_pct = (initial - val_losses_arr) / initial * 100

    axes[0, 1].fill_between(epochs, 0, reduction_pct, alpha=0.3, color='green')
    axes[0, 1].plot(epochs, reduction_pct, 'g-', linewidth=2)
    axes[0, 1].axhline(y=convergence_result.loss_reduction_percent, color='red', linestyle='--',
                        label=f'Max Reduction: {convergence_result.loss_reduction_percent:.1f}%')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss Reduction (%)', fontsize=12)
    axes[0, 1].set_title('Loss Reduction Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Stability analysis (rolling std)
    window = 5
    if len(val_losses) >= window:
        rolling_std = []
        rolling_mean = []
        for i in range(len(val_losses) - window + 1):
            window_data = val_losses_arr[i:i+window]
            rolling_std.append(np.std(window_data))
            rolling_mean.append(np.mean(window_data))

        rolling_cv = np.array(rolling_std) / np.array(rolling_mean)  # Coefficient of variation

        axes[1, 0].plot(range(window, len(val_losses) + 1), rolling_cv, 'purple', linewidth=2)
        axes[1, 0].axhline(y=0.01, color='green', linestyle='--', alpha=0.7, label='Stable threshold (1%)')
        axes[1, 0].fill_between(range(window, len(val_losses) + 1), 0, rolling_cv,
                                 where=(np.array(rolling_cv) < 0.01), alpha=0.3, color='green')

    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Coefficient of Variation', fontsize=12)
    axes[1, 0].set_title(f'Stability Analysis (Score: {convergence_result.stability_score:.2f})',
                         fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Summary text
    axes[1, 1].axis('off')

    status_color = 'green' if convergence_result.is_converged else 'orange'
    stability_color = 'green' if convergence_result.is_stable else 'orange'

    summary_text = f"""
    CONVERGENCE ANALYSIS SUMMARY
    {'='*40}

    Status: {'CONVERGED' if convergence_result.is_converged else 'NOT CONVERGED'}
    Trend: {convergence_result.trend.upper()}

    METRICS
    {'-'*40}
    Initial Loss:     {val_losses[0]:.6f}
    Final Loss:       {convergence_result.final_loss:.6f}
    Best Loss:        {convergence_result.best_loss:.6f}
    Loss Reduction:   {convergence_result.loss_reduction_percent:.2f}%

    STABILITY
    {'-'*40}
    Stability Score:  {convergence_result.stability_score:.2f}/1.00
    Is Stable:        {'Yes' if convergence_result.is_stable else 'No'}

    RECOMMENDATIONS
    {'-'*40}
    """

    for rec in convergence_result.recommendations:
        summary_text += f"\n    * {rec}"

    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Convergence analysis saved to {save_path}")

    return fig


def generate_validation_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    train_losses: List[float],
    val_losses: List[float],
    output_dir: Path,
    additional_metrics: Optional[Dict] = None
) -> Dict:
    """
    Generate comprehensive validation report with all visualizations.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        train_losses: Training loss history
        val_losses: Validation loss history
        output_dir: Directory to save outputs
        additional_metrics: Optional dict with MAE, RMSE, MAPE lists

    Returns:
        Dictionary with analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  GENERATING VALIDATION REPORT")
    print("="*60)

    # 1. Analyze convergence
    print("\n[1/4] Analyzing convergence...")
    convergence = analyze_convergence(train_losses, val_losses)

    # 2. Generate prediction heatmap
    print("[2/4] Generating prediction heatmaps...")
    plot_prediction_heatmap(
        predictions, targets,
        save_path=output_dir / "prediction_heatmap.png"
    )
    plt.close()

    # 3. Generate error distribution
    print("[3/4] Generating error distribution analysis...")
    plot_error_distribution_heatmap(
        predictions, targets,
        save_path=output_dir / "error_distribution.png"
    )
    plt.close()

    # 4. Generate convergence analysis
    print("[4/4] Generating convergence analysis...")
    plot_convergence_analysis(
        train_losses, val_losses, convergence,
        save_path=output_dir / "convergence_analysis.png"
    )
    plt.close()

    # Calculate final metrics
    if predictions.ndim == 4:
        # (num_samples, num_nodes, features, extra_dim)
        pred = predictions[:, :, 0, 0]
        target = targets[:, :, 0] if targets.ndim == 3 else targets[:, :, 0, 0]
    elif predictions.ndim == 3:
        # (num_samples, num_nodes, features)
        pred = predictions[:, :, 0]
        target = targets[:, :, 0] if targets.ndim == 3 else targets[:, :]
    else:
        # (num_samples, num_nodes)
        pred = predictions
        target = targets

    error = pred - target
    abs_error = np.abs(error)

    # Overall metrics
    mae = np.mean(abs_error)
    rmse = np.sqrt(np.mean(error**2))

    mask_nonzero = target != 0
    if mask_nonzero.sum() > 0:
        mape = np.mean(np.abs(error[mask_nonzero] / target[mask_nonzero])) * 100
    else:
        mape = 0.0

    # R2 score
    ss_res = np.sum(error**2)
    ss_tot = np.sum((target - np.mean(target))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Compile report
    report = {
        'convergence': {
            'is_converged': bool(convergence.is_converged),
            'convergence_epoch': convergence.convergence_epoch,
            'final_loss': convergence.final_loss,
            'best_loss': convergence.best_loss,
            'loss_reduction_percent': convergence.loss_reduction_percent,
            'is_stable': bool(convergence.is_stable),
            'stability_score': convergence.stability_score,
            'trend': convergence.trend,
            'recommendations': convergence.recommendations
        },
        'metrics': {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2)
        },
        'node_analysis': {
            'num_nodes': pred.shape[1],
            'node_mae_mean': float(np.mean(np.mean(abs_error, axis=0))),
            'node_mae_std': float(np.std(np.mean(abs_error, axis=0))),
            'worst_node': int(np.argmax(np.mean(abs_error, axis=0))),
            'best_node': int(np.argmin(np.mean(abs_error, axis=0)))
        },
        'data_info': {
            'num_samples': pred.shape[0],
            'prediction_range': [float(pred.min()), float(pred.max())],
            'target_range': [float(target.min()), float(target.max())]
        },
        'output_files': {
            'prediction_heatmap': str(output_dir / "prediction_heatmap.png"),
            'error_distribution': str(output_dir / "error_distribution.png"),
            'convergence_analysis': str(output_dir / "convergence_analysis.png")
        }
    }

    # Save report as JSON
    report_path = output_dir / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[INFO] Validation report saved to {report_path}")

    # Print summary
    print("\n" + "="*60)
    print("  VALIDATION SUMMARY")
    print("="*60)
    print(f"\n  Convergence Status: {'CONVERGED' if convergence.is_converged else 'NOT CONVERGED'}")
    print(f"  Trend: {convergence.trend}")
    print(f"  Stability Score: {convergence.stability_score:.2f}/1.00")
    print(f"\n  Final Metrics:")
    print(f"    - MAE:  {mae:.6f}")
    print(f"    - RMSE: {rmse:.6f}")
    print(f"    - MAPE: {mape:.2f}%")
    print(f"    - R2:   {r2:.4f}")
    print(f"\n  Recommendations:")
    for rec in convergence.recommendations:
        print(f"    * {rec}")
    print("="*60 + "\n")

    return report


def quick_convergence_check(val_losses: List[float], verbose: bool = True) -> bool:
    """
    Quick check if training has converged (for use during training).

    Args:
        val_losses: Validation losses so far
        verbose: Whether to print status

    Returns:
        True if converged, False otherwise
    """
    result = analyze_convergence([], val_losses)

    if verbose:
        status = "CONVERGED" if result.is_converged else "IN PROGRESS"
        print(f"[Convergence Check] {status} | Trend: {result.trend} | Stability: {result.stability_score:.2f}")

    return result.is_converged
