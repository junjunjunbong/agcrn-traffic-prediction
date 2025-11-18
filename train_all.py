"""
Train all datasets automatically
Usage: python train_all.py
"""
import subprocess
import json
import shutil
from pathlib import Path

# Configuration
DATASETS = ["loops_033", "loops_035", "loops_040"]
EPOCHS = 20
LOSS = "observed_only"
LR = 0.001

def train_dataset(dataset_name):
    """Train a single dataset"""
    print("\n" + "="*60)
    print(f"Training: {dataset_name}")
    print("="*60)

    # Create results directory
    results_dir = Path("results") / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "python", "train.py",
        "--data", dataset_name,
        "--loss", LOSS,
        "--lr", str(LR),
        "--epochs", str(EPOCHS)
    ]

    # Run training
    log_file = results_dir / "training.log"
    print(f"Command: {' '.join(cmd)}")
    print(f"Log: {log_file}")

    try:
        with open(log_file, "w") as f:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )
            f.write(result.stdout)
            print(result.stdout)

        print(f"[OK] {dataset_name} training completed!")

        # Copy results
        if Path("saved_models/best_agcrn.pt").exists():
            shutil.copy(
                "saved_models/best_agcrn.pt",
                results_dir / "best_model.pt"
            )
            print(f"[OK] Model saved to {results_dir}/best_model.pt")

        if Path("logs/training_history.json").exists():
            shutil.copy(
                "logs/training_history.json",
                results_dir / "history.json"
            )
            print(f"[OK] History saved to {results_dir}/history.json")

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error training {dataset_name}:")
        print(e.stdout)
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def compare_results():
    """Compare results from all datasets"""
    print("\n" + "="*60)
    print("Comparing Results")
    print("="*60)

    results = []
    for dataset in DATASETS:
        history_file = Path("results") / dataset / "history.json"
        if history_file.exists():
            with open(history_file) as f:
                history = json.load(f)
                results.append({
                    "dataset": dataset,
                    "best_val_loss": history.get("best_val_loss", float('inf')),
                    "final_train_loss": history.get("train_losses", [float('inf')])[-1],
                    "num_epochs": len(history.get("train_losses", []))
                })

    if not results:
        print("No results found!")
        return

    # Sort by best validation loss
    results.sort(key=lambda x: x["best_val_loss"])

    print("\nResults (sorted by best validation loss):")
    print("-" * 60)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['dataset']}")
        print(f"   Best Val Loss:  {r['best_val_loss']:.6f}")
        print(f"   Final Train Loss: {r['final_train_loss']:.6f}")
        print(f"   Epochs: {r['num_epochs']}")
        print()

    best = results[0]
    print(f"🏆 Best dataset: {best['dataset']} (val_loss: {best['best_val_loss']:.6f})")


def main():
    print("="*60)
    print("Training All Datasets")
    print("="*60)
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Epochs: {EPOCHS}")
    print(f"Loss: {LOSS}")
    print(f"Learning rate: {LR}")
    print("="*60)

    # Train all datasets
    success_count = 0
    for dataset in DATASETS:
        if train_dataset(dataset):
            success_count += 1

    print("\n" + "="*60)
    print(f"Training Summary: {success_count}/{len(DATASETS)} successful")
    print("="*60)

    # Compare results
    if success_count > 0:
        compare_results()

    print("\nResults saved in: results/")


if __name__ == "__main__":
    main()
