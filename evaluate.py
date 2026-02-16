"""
Evaluation & Inference script for CropDiseaseNet.

• Per-class precision / recall / F1 report
• Confusion matrix heatmap
• Training-curve plots (loss & accuracy)
• Single-image prediction function
"""

import os
import csv

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import config
from dataset import get_dataloaders, get_val_transforms
from model import build_model
from PIL import Image


# ─── Load trained model ──────────────────────────────────────────────────────
def load_model(checkpoint_path: str = None, device=None):
    """Load the best saved model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.SAVE_DIR, "best_model.pth")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = ckpt["num_classes"]
    idx_to_class = ckpt["idx_to_class"]

    model = build_model(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  (val_acc={ckpt['val_acc']*100:.2f}%)")
    return model, idx_to_class, device


# ─── Full test-set evaluation ────────────────────────────────────────────────
@torch.no_grad()
def evaluate_test_set(model=None, idx_to_class=None, device=None):
    """Run full evaluation on the test split and produce reports."""
    if model is None:
        model, idx_to_class, device = load_model()

    _, _, test_loader, meta = get_dataloaders()
    if idx_to_class is None:
        idx_to_class = meta["idx_to_class"]

    all_preds = []
    all_labels = []

    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Class names in order
    num_classes = len(idx_to_class)
    class_names = [idx_to_class[i] if isinstance(i, int) else idx_to_class[str(i)]
                   for i in range(num_classes)]

    # ── Classification report ────────────────────────────────────────────
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names, digits=3, zero_division=0)
    print("\n" + "=" * 80)
    print("  CLASSIFICATION REPORT")
    print("=" * 80)
    print(report)

    report_path = os.path.join(config.LOG_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved → {report_path}")

    # ── Confusion matrix ─────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names)

    return all_preds, all_labels


# ─── Confusion matrix plot ────────────────────────────────────────────────────
def plot_confusion_matrix(cm, class_names):
    """Save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix", fontsize=16)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(config.LOG_DIR, "confusion_matrix.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {save_path}")


# ─── Training curves ─────────────────────────────────────────────────────────
def plot_training_curves():
    """Read training_log.csv and plot loss / accuracy curves."""
    log_path = os.path.join(config.LOG_DIR, "training_log.csv")
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    epochs, train_loss, val_loss, train_acc, val_acc = [], [], [], [], []
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_acc.append(float(row["val_acc"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_loss, label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in train_acc], label="Train Acc", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in val_acc], label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config.LOG_DIR, "training_curves.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved → {save_path}")


# ─── Single-image inference ──────────────────────────────────────────────────
def predict_image(image_path: str, model=None, idx_to_class=None, device=None, top_k: int = 5):
    """
    Predict the disease class for a single image.

    Returns:
        list of (class_name, probability) for top-k predictions.
    """
    if model is None:
        model, idx_to_class, device = load_model()

    transform = get_val_transforms()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()

    top_probs, top_idxs = probs.topk(top_k)
    results = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_idxs.cpu().numpy()):
        class_name = idx_to_class[idx] if isinstance(idx, (int, np.integer)) else idx_to_class[str(idx)]
        results.append((class_name, float(prob)))

    print(f"\nPrediction for: {os.path.basename(image_path)}")
    print("-" * 50)
    for rank, (name, prob) in enumerate(results, 1):
        bar = "█" * int(prob * 30)
        print(f"  {rank}. {name:<45s} {prob*100:5.2f}% {bar}")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Evaluating best model on test set...\n")
    evaluate_test_set()
    plot_training_curves()
