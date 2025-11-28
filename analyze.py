"""
Analysis script for AGCRN traffic prediction

This script provides comprehensive analysis of trained models:
1. Heatmap visualization (prediction vs actual)
2. Error distribution analysis by node
3. Loss convergence analysis
4. Validation report generation

Usage:
    python analyze.py                    # Analyze latest trained model
    python analyze.py --model path/to/model.pt
    python analyze.py --history path/to/training_history.json
"""

import argparse
import json
from pathlib import Path
import torch
import numpy as np

from src.config import (
    NUM_NODES, INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, NUM_LAYERS,
    CHEB_K, EMBED_DIM, BATCH_SIZE, DEVICE, SAVED_MODELS_DIR, LOGS_DIR
)
from src.model_agcrn import AGCRN
from src.dataset import create_dataloaders
from src.validation_analysis import (
    analyze_convergence,
    plot_prediction_heatmap,
    plot_error_distribution_heatmap,
    plot_convergence_analysis,
    generate_validation_report,
    ConvergenceResult
)


def load_model(model_path: Path, device: str = DEVICE) -> AGCRN:
    """Load trained model from checkpoint"""
    model = AGCRN(
        num_nodes=NUM_NODES,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        cheb_k=CHEB_K,
        embed_dim=EMBED_DIM
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def get_predictions(model: AGCRN, data_loader, device: str = DEVICE):
    """Get predictions from model"""
    all_predictions = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for batch_data in data_loader:
            x, y, masks = batch_data
            x = x.to(device)
            y = y.to(device)

            y_target = y[:, -1, :, :]
            output = model(x)

            if output.shape[-1] != y_target.shape[-1]:
                pred = output.unsqueeze(-1)
                y_target = y_target[:, :, :1]
            else:
                pred = output

            all_predictions.append(pred.cpu().numpy())
            all_targets.append(y_target.cpu().numpy())

    return np.concatenate(all_predictions, axis=0), np.concatenate(all_targets, axis=0)


def analyze_from_history(history_path: Path, output_dir: Path = None):
    """Analyze convergence from training history only (no model needed)"""
    if output_dir is None:
        output_dir = LOGS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(history_path, 'r') as f:
        history = json.load(f)

    train_losses = history['train_losses']
    val_losses = history['val_losses']

    print("\n" + "="*60)
    print("  CONVERGENCE ANALYSIS (from history)")
    print("="*60)

    # Analyze convergence
    result = analyze_convergence(train_losses, val_losses)

    # Plot convergence analysis
    plot_convergence_analysis(
        train_losses, val_losses, result,
        save_path=output_dir / "convergence_analysis.png"
    )

    # Print summary
    print(f"\n  Status: {'CONVERGED' if result.is_converged else 'NOT CONVERGED'}")
    print(f"  Trend: {result.trend}")
    print(f"  Stability Score: {result.stability_score:.2f}/1.00")
    print(f"  Loss Reduction: {result.loss_reduction_percent:.2f}%")
    print(f"  Best Loss: {result.best_loss:.6f}")
    print(f"  Final Loss: {result.final_loss:.6f}")

    if result.convergence_epoch:
        print(f"  Convergence Epoch: {result.convergence_epoch}")

    print("\n  Recommendations:")
    for rec in result.recommendations:
        print(f"    * {rec}")

    print("="*60)

    return result


def main():
    parser = argparse.ArgumentParser(description='Analyze AGCRN model training')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint (default: saved_models/best_agcrn.pt)')
    parser.add_argument('--history', type=str, default=None,
                        help='Path to training history JSON (default: logs/training_history.json)')
    parser.add_argument('--data', type=str, default='loops_035',
                        help='Data name for prediction analysis')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for analysis results')
    parser.add_argument('--history-only', action='store_true',
                        help='Only analyze training history (no model/predictions)')
    parser.add_argument('--device', type=str, default=DEVICE,
                        help='Device to use')

    args = parser.parse_args()

    # Set paths
    model_path = Path(args.model) if args.model else SAVED_MODELS_DIR / "best_agcrn.pt"
    history_path = Path(args.history) if args.history else LOGS_DIR / "training_history.json"
    output_dir = Path(args.output) if args.output else LOGS_DIR

    print("\n" + "="*60)
    print("  AGCRN Model Analysis")
    print("="*60)
    print(f"  Model: {model_path}")
    print(f"  History: {history_path}")
    print(f"  Output: {output_dir}")
    print("="*60)

    # Check if history exists
    if not history_path.exists():
        print(f"\n[ERROR] Training history not found: {history_path}")
        print("Please train a model first or specify --history path")
        return

    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)

    train_losses = history['train_losses']
    val_losses = history['val_losses']

    # History-only analysis
    if args.history_only:
        analyze_from_history(history_path, output_dir)
        return

    # Full analysis with model
    if not model_path.exists():
        print(f"\n[WARNING] Model not found: {model_path}")
        print("Running history-only analysis...")
        analyze_from_history(history_path, output_dir)
        return

    # Load model and data
    print("\n[1/4] Loading model and data...")
    model = load_model(model_path, args.device)

    _, val_loader, _ = create_dataloaders(
        data_name=args.data,
        batch_size=BATCH_SIZE
    )

    # Get predictions
    print("[2/4] Getting predictions...")
    predictions, targets = get_predictions(model, val_loader, args.device)

    # Generate comprehensive report
    print("[3/4] Generating analysis...")
    report = generate_validation_report(
        predictions=predictions,
        targets=targets,
        train_losses=train_losses,
        val_losses=val_losses,
        output_dir=output_dir
    )

    print("\n[4/4] Analysis complete!")
    print(f"\nOutput files saved to: {output_dir}")
    print("  - prediction_heatmap.png")
    print("  - error_distribution.png")
    print("  - convergence_analysis.png")
    print("  - validation_report.json")


if __name__ == "__main__":
    main()
