"""
Stage-aware training script for the crop-disease pipeline.

Stages
------
    python train.py --stage scratch     # custom CNN, random init, small subset
    python train.py --stage pretrain    # ImageNet backbone -> full PlantVillage
    python train.py --stage finetune    # PlantVillage backbone -> PlantDoc

What is different from the original script
------------------------------------------
* Per-stage config (see config.py) with per-stage output folders, so the three
  runs no longer overwrite each other's checkpoints and logs.
* Two-phase fine-tuning: classifier head only for the first N epochs, then the
  whole backbone at a much lower LR. Unfreezing everything at once at a high LR
  destroys pretrained features when there are only ~2.6k images.
* Frozen BatchNorm layers are held in eval mode, so a frozen backbone cannot
  drift its running statistics onto the new dataset.
* Single LambdaLR replaces the previous warm-up class, which used to skip
  StepLR steps and desynchronise the LR schedule.
* Writes logs/<stage>/summary.json for the cross-stage comparison report.
"""

import argparse
import csv
import json
import math
import os
import time

# ── The stage must be selected BEFORE config is imported ─────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="Train one stage of the pipeline.")
    p.add_argument("--stage", default=None,
                   choices=["scratch", "pretrain", "finetune"],
                   help="Which stage to run (default: $CNN_STAGE or scratch)")
    p.add_argument("--epochs", type=int, default=None, help="Override NUM_EPOCHS")
    p.add_argument("--batch-size", type=int, default=None, help="Override BATCH_SIZE")
    p.add_argument("--img-size", type=int, default=None, help="Override IMG_SIZE")
    p.add_argument("--dataset", default=None, help="Override DATASET_ROOT")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-sampler", action="store_true",
                   help="Disable the class-balanced sampler")
    return p.parse_args()


ARGS = _parse_args()
if ARGS.stage:
    os.environ["CNN_STAGE"] = ARGS.stage

import config                                                    # noqa: E402

if ARGS.epochs:
    config.NUM_EPOCHS = ARGS.epochs
if ARGS.batch_size:
    config.BATCH_SIZE = ARGS.batch_size
if ARGS.img_size:
    config.IMG_SIZE = ARGS.img_size
if ARGS.dataset:
    config.DATASET_ROOT = ARGS.dataset
if ARGS.num_workers is not None:
    config.NUM_WORKERS = ARGS.num_workers
if ARGS.no_sampler:
    config.USE_WEIGHTED_SAMPLER = False

import torch                                                     # noqa: E402
import torch.nn as nn                                            # noqa: E402
from torch.amp import GradScaler, autocast                        # noqa: E402
from tqdm import tqdm                                             # noqa: E402

from dataset import get_dataloaders                               # noqa: E402
from model import (build_model, set_backbone_trainable,           # noqa: E402
                   set_backbone_bn_eval, head_parameters,
                   backbone_parameters)


# ─── Utility ─────────────────────────────────────────────────────────────────
def accuracy(output, target):
    _, preds = output.max(1)
    return preds.eq(target).sum().item() / target.size(0)


def banner(text, char="=", width=78):
    print(f"\n{char * width}\n  {text}\n{char * width}")


def make_scheduler(optimizer, total_epochs, warmup_epochs, kind,
                   step_size, gamma):
    """One scheduler for warm-up + decay, so the two never fight.

    Returns a LambdaLR whose multiplier is:
        linear ramp 0->1 over warmup_epochs, then
        cosine decay, or a step decay every step_size epochs.
    """
    def schedule(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        if kind == "cosine":
            span = max(1, total_epochs - warmup_epochs)
            progress = min(1.0, (epoch - warmup_epochs) / span)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        progress = (epoch - warmup_epochs) // max(1, step_size)
        return float(gamma ** progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


# ─── One epoch ───────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    freeze_backbone=False):
    model.train()
    if freeze_backbone:
        set_backbone_bn_eval(model)

    running_loss = running_acc = 0.0
    total = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(outputs, labels) * bs
        total += bs
        pbar.set_postfix(loss=f"{running_loss/total:.4f}",
                         acc=f"{running_acc/total:.4f}")

    return running_loss / total, running_acc / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = running_acc = 0.0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(outputs, labels) * bs
        total += bs

    return running_loss / total, running_acc / total


# ─── Main ────────────────────────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stage: {config.ACTIVE_STAGE}")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    banner("Configuration")
    print(f"  Dataset      : {config.DATASET_ROOT}")
    print(f"  Architecture : {config.ARCHITECTURE}"
          + (" (ImageNet weights)" if config.PRETRAINED else ""))
    if config.INIT_CHECKPOINT:
        print(f"  Warm start   : {config.INIT_CHECKPOINT}")
    print(f"  Image size   : {config.IMG_SIZE}")
    print(f"  Batch size   : {config.BATCH_SIZE}")
    print(f"  Epochs       : {config.NUM_EPOCHS}")
    print(f"  Head LR      : {config.HEAD_LR}")
    print(f"  Backbone LR  : {config.BACKBONE_LR}")
    print(f"  Freeze epochs: {config.FREEZE_BACKBONE_EPOCHS}")
    print(f"  Scheduler    : {config.SCHEDULER}  (warmup {config.WARMUP_EPOCHS})")
    print(f"  Sampler      : {'class-balanced' if config.USE_WEIGHTED_SAMPLER else 'plain'}")

    # ── Data ──
    train_loader, val_loader, meta = get_dataloaders()
    num_classes = meta["num_classes"]
    idx_to_class = meta["idx_to_class"]

    mapping_path = os.path.join(config.SAVE_DIR, "class_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump({str(k): v for k, v in idx_to_class.items()}, f, indent=2)
    print(f"Class mapping saved -> {mapping_path}")

    # ── Model ──
    model = build_model(
        num_classes=num_classes,
        architecture=config.ARCHITECTURE,
        pretrained=config.PRETRAINED,
        init_weights=not bool(config.INIT_CHECKPOINT),
        dropout=config.DROPOUT,
        init_from=config.INIT_CHECKPOINT,
    ).to(device)

    # ── Freeze phase ──
    frozen = config.FREEZE_BACKBONE_EPOCHS > 0
    if frozen:
        set_backbone_trainable(model, False)
        print(f"Backbone frozen for the first {config.FREEZE_BACKBONE_EPOCHS} "
              f"epoch(s); head only.")

    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters(model), "lr": config.BACKBONE_LR,
             "weight_decay": config.WEIGHT_DECAY, "name": "backbone"},
            {"params": head_parameters(model), "lr": config.HEAD_LR,
             "weight_decay": config.WEIGHT_DECAY, "name": "head"},
        ],
        lr=config.HEAD_LR,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = make_scheduler(optimizer, config.NUM_EPOCHS,
                               config.WARMUP_EPOCHS, config.SCHEDULER,
                               config.STEP_LR_SIZE, config.STEP_LR_GAMMA)

    scaler = GradScaler(device=device.type, enabled=(device.type == "cuda"))

    # ── Logging ──
    log_path = os.path.join(config.LOG_DIR, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "train_acc",
                         "val_loss", "val_acc", "lr"])

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    patience = 0
    ckpt_path = os.path.join(config.SAVE_DIR, "best_model.pth")

    banner(f"Training {config.ACTIVE_STAGE}  |  {num_classes} classes  |  "
           f"{config.NUM_EPOCHS} epochs")
    start = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        lr = optimizer.param_groups[1]["lr"]

        if frozen and epoch > config.FREEZE_BACKBONE_EPOCHS:
            frozen = False
            set_backbone_trainable(model, True)
            print(f"  >> Unfroze backbone at epoch {epoch} "
                  f"(backbone LR {optimizer.param_groups[0]['lr']:.2e})")

        print(f"Epoch {epoch}/{config.NUM_EPOCHS}  (head lr={lr:.2e}"
              f"{', frozen backbone' if frozen else ''})")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            freeze_backbone=frozen)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        log_writer.writerow([epoch, f"{train_loss:.5f}", f"{train_acc:.5f}",
                             f"{val_loss:.5f}", f"{val_acc:.5f}", f"{lr:.6f}"])
        log_file.flush()

        print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc*100:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_val_acc = val_acc
            patience = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "num_classes": num_classes,
                "idx_to_class": idx_to_class,
                "architecture": config.ARCHITECTURE,
                "stage": config.ACTIVE_STAGE,
                "img_size": config.IMG_SIZE,
            }, ckpt_path)
            print(f"  [best] saved (val_loss={val_loss:.4f})")
        else:
            patience += 1
            print(f"  No improvement ({patience}/{config.EARLY_STOP_PATIENCE})")

        if patience >= config.EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}.")
            break
        print()

    log_file.close()
    elapsed = time.time() - start

    # ── Final evaluation with the best checkpoint ──
    print("\nLoading best checkpoint for final evaluation...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    summary = {
        "stage": config.ACTIVE_STAGE,
        "architecture": config.ARCHITECTURE,
        "pretrained": config.PRETRAINED,
        "warm_start": config.INIT_CHECKPOINT,
        "dataset": config.DATASET_ROOT,
        "layout": meta["layout"],
        "uses_official_split": meta["uses_official_split"],
        "num_classes": num_classes,
        "num_train": meta["num_train"],
        "num_val": meta["num_val"],
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "final_val_loss": val_loss,
        "final_val_acc": val_acc,
        "elapsed_min": elapsed / 60.0,
        "img_size": config.IMG_SIZE,
        "batch_size": config.BATCH_SIZE,
        "head_lr": config.HEAD_LR,
        "backbone_lr": config.BACKBONE_LR,
        "freeze_backbone_epochs": config.FREEZE_BACKBONE_EPOCHS,
        "scheduler": config.SCHEDULER,
    }
    summary_path = os.path.join(config.LOG_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    banner(f"Done in {elapsed/60:.1f} min  |  best epoch {best_epoch}  |  "
           f"val acc {val_acc*100:.2f}%")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Log        : {log_path}")
    print(f"Summary    : {summary_path}")
    print("\nNext: python report.py    (build the visual dashboard)")

    return model


if __name__ == "__main__":
    train()
