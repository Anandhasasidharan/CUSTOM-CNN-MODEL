"""
Configuration file for Multi-Crop Disease Detection CNN.
All hyperparameters and paths are defined here.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = r"D:\college\DATASET\Augmented_Dataset"
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
IMG_SIZE = 224                   # Input image resolution
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Train / Val split ratio
TRAIN_RATIO = 0.80
VAL_RATIO = 0.20

# ─── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
NUM_EPOCHS = 40
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
STEP_LR_SIZE = 10               # Drop LR every N epochs
STEP_LR_GAMMA = 0.5             # Multiply LR by this factor
EARLY_STOP_PATIENCE = 12        # Stop if val loss doesn't improve for N epochs

# ─── Device ───────────────────────────────────────────────────────────────────
NUM_WORKERS = 0                  # 0 is safest on Windows (avoids spawn issues)
PIN_MEMORY = True
