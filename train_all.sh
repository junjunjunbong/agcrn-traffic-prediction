#!/bin/bash
# Train all datasets automatically

echo "=========================================="
echo "Training all datasets: loops_033, loops_035, loops_040"
echo "=========================================="

# Dataset list
datasets=("loops_033" "loops_035" "loops_040")

# Training parameters
EPOCHS=20
LOSS="observed_only"
LR=0.001

for dataset in "${datasets[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training: $dataset"
    echo "=========================================="

    # Create output directory for this dataset
    mkdir -p results/$dataset

    # Train
    python train.py \
        --data $dataset \
        --loss $LOSS \
        --lr $LR \
        --epochs $EPOCHS \
        2>&1 | tee results/$dataset/training.log

    # Copy results
    if [ -f "saved_models/best_agcrn.pt" ]; then
        cp saved_models/best_agcrn.pt results/$dataset/best_model.pt
        echo "✓ Model saved to results/$dataset/best_model.pt"
    fi

    if [ -f "logs/training_history.json" ]; then
        cp logs/training_history.json results/$dataset/history.json
        echo "✓ History saved to results/$dataset/history.json"
    fi

    echo ""
    echo "✓ $dataset training completed!"
    echo ""
done

echo "=========================================="
echo "All datasets trained!"
echo "=========================================="
echo ""
echo "Results are in results/ directory:"
ls -lh results/*/

echo ""
echo "Compare best validation losses:"
for dataset in "${datasets[@]}"; do
    if [ -f "results/$dataset/history.json" ]; then
        best_loss=$(python -c "import json; data=json.load(open('results/$dataset/history.json')); print(f'{data[\"best_val_loss\"]:.6f}')")
        echo "  $dataset: $best_loss"
    fi
done
