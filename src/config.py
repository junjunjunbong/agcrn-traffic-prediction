"""
Configuration file for AGCRN traffic prediction project
"""
import os
from pathlib import Path

# Try to import torch, but don't fail if it's not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
META_DATA_DIR = DATA_DIR / "meta"

# Model paths
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Data processing parameters
NODE_MODE = "raw_id"  # Options: "raw_id" (480 nodes) or "det_pos" (160 nodes)
FEATURES = ["flow", "occupancy", "harmonicMeanSpeed"]  # Features to use
TIME_STEP_SIZE = 5.0  # seconds per time step

# Missing value handling
MISSING_SPEED_VALUE = -1.0  # Value indicating missing speed
FREE_FLOW_SPEED = 15.0  # m/s for free flow conditions
CONGESTED_SPEED = 2.5  # m/s for congested conditions

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Model hyperparameters
NUM_NODES = 480 if NODE_MODE == "raw_id" else 160
INPUT_DIM = len(FEATURES)  # Number of features
OUTPUT_DIM = 1  # Predict single feature (speed) or multi-feature
HIDDEN_DIM = 64
NUM_LAYERS = 2
CHEB_K = 2  # Chebyshev polynomial order
EMBED_DIM = 10  # Node embedding dimension

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
PATIENCE = 10  # Early stopping patience
SEQUENCE_LENGTH = 12  # Input sequence length (12 steps = 1 minute)
HORIZON = 3  # Prediction horizon (3 steps = 15 seconds ahead)

# Device
DEVICE = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

# Create directories if they don't exist
for dir_path in [PROCESSED_DATA_DIR, META_DATA_DIR, SAVED_MODELS_DIR, LOGS_DIR, CONFIGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

