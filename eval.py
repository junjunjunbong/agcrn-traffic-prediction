"""
Evaluation script for AGCRN traffic prediction model
"""
import argparse
from pathlib import Path
import torch

from src.config import (
    NUM_NODES, INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, NUM_LAYERS,
    CHEB_K, EMBED_DIM, BATCH_SIZE, DEVICE, SAVED_MODELS_DIR
)
from src.model_agcrn import AGCRN
from src.dataset import create_dataloaders
from src.eval import evaluate_model, plot_predictions, load_model


def main():
    parser = argparse.ArgumentParser(description='Evaluate AGCRN model')
    parser.add_argument('--data', type=str, default='loops_035',
                       help='Data name (e.g., loops_035)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default=DEVICE,
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate prediction plots')
    parser.add_argument('--num_nodes_to_plot', type=int, default=5,
                       help='Number of nodes to plot')
    parser.add_argument('--num_samples_to_plot', type=int, default=100,
                       help='Number of samples to plot')

    args = parser.parse_args()

    print("="*60)
    print("AGCRN Traffic Prediction - Evaluation")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Device: {args.device}")
    print("="*60)

    # Determine model path
    if args.model_path is None:
        model_path = SAVED_MODELS_DIR / "best_agcrn.pt"
    else:
        model_path = Path(args.model_path)

    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        print("Please train a model first using train.py")
        return

    # Load data
    print("\nLoading data...")
    _, _, test_loader = create_dataloaders(
        data_name=args.data,
        batch_size=args.batch_size
    )

    # Load model
    print("\nLoading model...")
    model_config = {
        'num_nodes': NUM_NODES,
        'input_dim': INPUT_DIM,
        'output_dim': OUTPUT_DIM,
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        'cheb_k': CHEB_K,
        'embed_dim': EMBED_DIM
    }

    model = load_model(model_path, model_config)
    model.to(args.device)

    print(f"Model loaded from {model_path}")

    # Evaluate
    print("\nEvaluating model...")
    metrics, predictions, targets = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=args.device
    )

    # Print metrics
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    print(f"MSE:  {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"Number of samples: {metrics['num_samples']}")
    print("="*60)

    # Plot predictions
    if args.plot:
        print("\nGenerating prediction plots...")
        plot_path = SAVED_MODELS_DIR / "predictions.png"
        plot_predictions(
            predictions=predictions,
            targets=targets,
            save_path=plot_path,
            num_nodes_to_plot=args.num_nodes_to_plot,
            num_samples_to_plot=args.num_samples_to_plot
        )
        print(f"Plot saved to {plot_path}")

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
