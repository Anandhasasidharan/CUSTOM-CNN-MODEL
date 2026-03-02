"""
Training script for CropDiseaseNet.

Features
--------
- Mixed-precision training (AMP) for speed on GPU
- Learning-rate scheduling (StepLR)
- Early stopping based on validation loss
- Saves best model checkpoint
- Logs training curves to CSV + plots
"""

import os
import csv
import time
import json

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

import config
from dataset import get_dataloaders
from model import build_model


# ─── Utility ──────────────────────────────────────────────────────────────────
def accuracy(output, target):
    """Top-1 accuracy."""
    _, preds = output.max(1)
    correct = preds.eq(target).sum().item()
    return correct / target.size(0)


# ─── Training for one epoch ──────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy(outputs, labels) * batch_size
        total += batch_size

        pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{running_acc/total:.4f}")

    return running_loss / total, running_acc / total


# ─── Learning-rate warm-up helper ─────────────────────────────────────────────
class LinearWarmupScheduler:
    """Linearly ramp LR from 0 → base_lr over `warmup_epochs` epochs."""

    def __init__(self, optimizer, warmup_epochs: int = 2):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        """Call BEFORE the main scheduler during warmup epochs."""
        if epoch > self.warmup_epochs:
            return False  # warmup finished
        factor = epoch / self.warmup_epochs
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * factor
        return True  # still warming up


# ─── Validation / Test for one epoch ─────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy(outputs, labels) * batch_size
        total += batch_size

    return running_loss / total, running_acc / total


# ─── Main training loop ──────────────────────────────────────────────────────
def train():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Data
    train_loader, val_loader, meta = get_dataloaders()
    num_classes = meta["num_classes"]
    idx_to_class = meta["idx_to_class"]

    # Save class mapping
    mapping_path = os.path.join(config.SAVE_DIR, "class_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump({str(k): v for k, v in idx_to_class.items()}, f, indent=2)
    print(f"Class mapping saved → {mapping_path}")

    # Model
    model = build_model(num_classes=num_classes).to(device)

    # Loss (with label smoothing for regularisation)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Scheduler (with 2-epoch linear warm-up)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.STEP_LR_SIZE,
        gamma=config.STEP_LR_GAMMA,
    )
    warmup = LinearWarmupScheduler(optimizer, warmup_epochs=2)

    # AMP scaler
    scaler = GradScaler(device=device.type, enabled=(device.type == "cuda"))

    # Logging
    log_path = os.path.join(config.LOG_DIR, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    # Early stopping state
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"  Training CropDiseaseNet  |  {num_classes} classes  |  {config.NUM_EPOCHS} epochs")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{config.NUM_EPOCHS}  (lr={lr:.6f})")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Warm-up overrides scheduler for the first N epochs
        if not warmup.step(epoch):
            scheduler.step()

        # Log
        log_writer.writerow([epoch, f"{train_loss:.5f}", f"{train_acc:.5f}",
                             f"{val_loss:.5f}", f"{val_acc:.5f}", f"{lr:.6f}"])
        log_file.flush()

        print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc*100:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc*100:.2f}%")

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            ckpt_path = os.path.join(config.SAVE_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "num_classes": num_classes,
                "idx_to_class": idx_to_class,
            }, ckpt_path)
            print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config.EARLY_STOP_PATIENCE})")

        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping triggered at epoch {epoch}.")
            break

        print()

    log_file.close()
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  Training complete in {elapsed/60:.1f} min")
    print(f"  Best epoch: {best_epoch}  |  Best val loss: {best_val_loss:.4f}")
    print(f"{'='*60}")

    # ── Final evaluation on val set with best model ─────────────────────────
    print("\nLoading best model for final evaluation...")
    ckpt = torch.load(os.path.join(config.SAVE_DIR, "best_model.pth"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"  Val Loss: {val_loss:.4f}  |  Val Acc: {val_acc*100:.2f}%")

    return model


if __name__ == "__main__":
    train()
