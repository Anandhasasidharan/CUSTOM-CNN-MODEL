"""
Configuration for the 3-stage crop-disease pipeline.

Stages
------
scratch   Train the custom CropDiseaseNet from random init on a small
          PlantVillage subset.  This is the "custom CNN" deliverable.

pretrain  Fine-tune an ImageNet backbone on the full PlantVillage
          (54k images, 38 classes) to build leaf-domain features.

finetune  Transfer that backbone onto PlantDoc field images
          (2.6k images, 27 classes).  This is the headline result.

Selecting a stage
-----------------
    python train.py --stage pretrain
    CNN_STAGE=finetune python evaluate.py

Or set the CNN_STAGE environment variable.  Defaults to "scratch".

Overriding dataset location
---------------------------
    CNN_DATASET_DIR=D:/data python train.py --stage scratch

Every stage writes to its own checkpoints/<stage>/ and logs/<stage>/ so the
three runs never overwrite each other.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Root that holds PlantVillage-Dataset/, PlantDoc-Dataset/, PlantVillage_3Crops/
DATASET_BASE = os.environ.get("CNN_DATASET_DIR", r"D:\AI\DATASET")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Device ───────────────────────────────────────────────────────────────────
NUM_WORKERS = 0                  # 0 is safest on Windows (avoids spawn issues)
PIN_MEMORY = True

# ─── Split (ignored when the dataset ships its own train/test folders) ───────
TRAIN_RATIO = 0.80
VAL_RATIO = 0.20


# ─── Per-stage presets ────────────────────────────────────────────────────────
# architecture      torchvision backbone name, or "custom" for CropDiseaseNet
# pretrained        start from ImageNet weights
# init_from         checkpoint to warm-start from: a stage name ("pretrain")
#                   or an explicit path.  The classifier head is re-initialised.
# head_lr           LR for the classification head
# backbone_lr       LR for the feature extractor (keep << head_lr when finetuning)
# freeze_backbone_epochs
#                   train only the head for N epochs before unfreezing
# scheduler         "step" or "cosine"
STAGES = {
    "scratch": {
        "dataset_subdir": "PlantVillage_3Crops",
        "architecture": "custom",
        "pretrained": False,
        "init_from": None,
        "img_size": 224,
        "batch_size": 64,
        "num_epochs": 40,
        "head_lr": 5e-4,
        "backbone_lr": 5e-4,
        "weight_decay": 1e-4,
        "dropout": 0.4,
        "label_smoothing": 0.1,
        "freeze_backbone_epochs": 0,
        "scheduler": "step",
        "step_lr_size": 10,
        "step_lr_gamma": 0.5,
        "warmup_epochs": 2,
        "early_stop_patience": 12,
        "use_weighted_sampler": True,
    },
    "pretrain": {
        "dataset_subdir": os.path.join("PlantVillage-Dataset", "color"),
        "architecture": "resnet18",
        "pretrained": True,
        "init_from": None,
        "img_size": 224,
        "batch_size": 64,
        "num_epochs": 12,
        "head_lr": 3e-4,
        "backbone_lr": 3e-5,
        "weight_decay": 1e-4,
        "dropout": 0.2,
        "label_smoothing": 0.1,
        "freeze_backbone_epochs": 1,
        "scheduler": "cosine",
        "step_lr_size": 10,
        "step_lr_gamma": 0.5,
        "warmup_epochs": 1,
        "early_stop_patience": 5,
        "use_weighted_sampler": True,
    },
    "finetune": {
        "dataset_subdir": "PlantDoc-Dataset",
        "architecture": "resnet18",
        "pretrained": False,
        "init_from": "pretrain",
        "img_size": 224,
        "batch_size": 32,          # smaller batch keeps BatchNorm stats sane
        "num_epochs": 25,
        "head_lr": 3e-4,
        "backbone_lr": 1e-5,       # ~30x lower than the head
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "label_smoothing": 0.1,
        "freeze_backbone_epochs": 3,
        "scheduler": "cosine",
        "step_lr_size": 10,
        "step_lr_gamma": 0.5,
        "warmup_epochs": 1,
        "early_stop_patience": 6,
        "use_weighted_sampler": True,
    },
}

DEFAULT_STAGE = "scratch"


# ─── Resolve the active stage ─────────────────────────────────────────────────
def get_stage() -> str:
    stage = os.environ.get("CNN_STAGE", DEFAULT_STAGE).strip().lower()
    if stage not in STAGES:
        raise SystemExit(
            f"Unknown stage {stage!r}. Choose one of: {', '.join(STAGES)}"
        )
    return stage


def _resolve_init(stage_name: str, value):
    """Turn a stage name into that stage's best-checkpoint path."""
    if not value:
        return None
    if value in STAGES:
        return os.path.join(BASE_DIR, "checkpoints", value, "best_model.pth")
    return value


ACTIVE_STAGE = get_stage()
_ACTIVE = STAGES[ACTIVE_STAGE]

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET_ROOT = os.path.join(DATASET_BASE, _ACTIVE["dataset_subdir"])
IMG_SIZE = _ACTIVE["img_size"]

# ─── Model ────────────────────────────────────────────────────────────────────
ARCHITECTURE = _ACTIVE["architecture"]
PRETRAINED = _ACTIVE["pretrained"]
INIT_CHECKPOINT = _resolve_init(ACTIVE_STAGE, _ACTIVE["init_from"])
DROPOUT = _ACTIVE["dropout"]

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = _ACTIVE["batch_size"]
NUM_EPOCHS = _ACTIVE["num_epochs"]
HEAD_LR = _ACTIVE["head_lr"]
BACKBONE_LR = _ACTIVE["backbone_lr"]
LEARNING_RATE = HEAD_LR          # backwards-compatible alias
WEIGHT_DECAY = _ACTIVE["weight_decay"]
LABEL_SMOOTHING = _ACTIVE["label_smoothing"]
FREEZE_BACKBONE_EPOCHS = _ACTIVE["freeze_backbone_epochs"]
SCHEDULER = _ACTIVE["scheduler"]
STEP_LR_SIZE = _ACTIVE["step_lr_size"]
STEP_LR_GAMMA = _ACTIVE["step_lr_gamma"]
WARMUP_EPOCHS = _ACTIVE["warmup_epochs"]
EARLY_STOP_PATIENCE = _ACTIVE["early_stop_patience"]
USE_WEIGHTED_SAMPLER = _ACTIVE["use_weighted_sampler"]

# ─── Output dirs (per stage, so stages never clobber each other) ─────────────
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints", ACTIVE_STAGE)
LOG_DIR = os.path.join(BASE_DIR, "logs", ACTIVE_STAGE)

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
