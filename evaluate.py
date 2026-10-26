"""
Evaluation & Inference script for CropDiseaseNet.

• Per-class precision / recall / F1 report
• Confusion matrix heatmap
• Training-curve plots (loss & accuracy)
• Single-image prediction function
"""

import os
import csv

# ── The stage must be selected BEFORE config is imported ─────────────────────
def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Evaluate one stage's checkpoint.")
    p.add_argument("--stage", default=None,
                   choices=["scratch", "pretrain", "finetune"],
                   help="Which stage to evaluate (default: $CNN_STAGE)")
    p.add_argument("--checkpoint", default=None, help="Explicit checkpoint path")
    p.add_argument("--dataset", default=None, help="Override DATASET_ROOT")
    p.add_argument("--img-size", type=int, default=None, help="Override IMG_SIZE")
    return p.parse_args()


ARGS = _parse_args()
if ARGS.stage:
    os.environ["CNN_STAGE"] = ARGS.stage

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import config

if ARGS.dataset:
    config.DATASET_ROOT = ARGS.dataset
if ARGS.img_size:
    config.IMG_SIZE = ARGS.img_size
from dataset import get_dataloaders, get_val_transforms
from model import build_model
from PIL import Image


# ─── Load trained model ──────────────────────────────────────────────────────
def load_model(checkpoint_path: str = None, device=None):
    """Load the best saved model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_path is None:
        checkpoint_path = ARGS.checkpoint or os.path.join(config.SAVE_DIR, "best_model.pth")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = ckpt["num_classes"]
    idx_to_class = ckpt["idx_to_class"]
    architecture = ckpt.get("architecture", "custom")

    # Rebuild the same architecture the checkpoint came from, without
    # re-downloading ImageNet weights and without re-initialising (the
    # state_dict load supplies every weight).
    model = build_model(num_classes=num_classes, architecture=architecture,
                        pretrained=False, init_weights=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    print(f"Loaded {ckpt.get('stage', '?')}/{architecture} from epoch "
          f"{ckpt['epoch']}  (val_acc={ckpt['val_acc']*100:.2f}%)")
    return model, idx_to_class, device



# ─── Per-crop classification report ──────────────────────────────────────────
def _save_crop_report(crop_name: str, all_labels, all_preds, idx_to_class):
    """
    Filter labels/preds to a single crop, then print & save a report
    in the format shown in the target image:

        Test Results:
        Accuracy: X.XXXX
        Precision (weighted): X.XXXX
        Recall (weighted): X.XXXX
        F1-score (weighted): X.XXXX

        Classification Report:
                              precision  recall  f1-score  support
        ...
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Find class indices that belong to this crop
    crop_class_indices = []
    disease_names = []
    for idx in sorted(idx_to_class.keys() if isinstance(list(idx_to_class.keys())[0], int)
                      else [int(k) for k in idx_to_class.keys()]):
        name = idx_to_class[idx] if isinstance(idx, int) else idx_to_class[str(idx)]
        # e.g. "Black_Gram - anthracnose"  →  crop prefix is before " - "
        crop_prefix = name.split(" - ")[0]
        if crop_prefix == crop_name:
            crop_class_indices.append(idx)
            disease_names.append(name.split(" - ", 1)[1])  # just disease name

    if not crop_class_indices:
        return

    # Filter predictions/labels to only samples belonging to this crop
    mask = np.isin(all_labels, crop_class_indices)
    crop_labels = all_labels[mask]
    crop_preds  = all_preds[mask]

    if len(crop_labels) == 0:
        return

    # Remap to 0-based within the crop
    idx_remap = {orig: new for new, orig in enumerate(crop_class_indices)}
    crop_labels_local = np.array([idx_remap[l] for l in crop_labels])
    crop_preds_local  = np.array([idx_remap[p] if p in idx_remap else -1
                                  for p in crop_preds])

    acc  = accuracy_score(crop_labels_local, crop_preds_local)
    prec = precision_score(crop_labels_local, crop_preds_local,
                           average="weighted", zero_division=0)
    rec  = recall_score(crop_labels_local, crop_preds_local,
                        average="weighted", zero_division=0)
    f1   = f1_score(crop_labels_local, crop_preds_local,
                    average="weighted", zero_division=0)

    valid_labels = list(range(len(crop_class_indices)))
    report = classification_report(crop_labels_local, crop_preds_local,
                                   labels=valid_labels,
                                   target_names=disease_names, digits=4, zero_division=0)

    header = (
        f"Test Results:\n"
        f"Accuracy: {acc:.4f}\n"
        f"Precision (weighted): {prec:.4f}\n"
        f"Recall (weighted): {rec:.4f}\n"
        f"F1-score (weighted): {f1:.4f}\n\n"
        f"Classification Report:\n"
    )

    full_report = header + report

    print(f"\n{'='*70}")
    print(f"  {crop_name.upper()} — CROP REPORT")
    print(f"{'='*70}")
    print(full_report)

    safe_name = crop_name.lower().replace(" ", "_")
    report_path = os.path.join(config.LOG_DIR, f"classification_report_{safe_name}.txt")
    with open(report_path, "w") as f:
        f.write(full_report)
    print(f"Report saved → {report_path}")


# ─── Full val-set evaluation ─────────────────────────────────────────────────
@torch.no_grad()
def evaluate_test_set(model=None, idx_to_class=None, device=None):
    """Run full evaluation on the val split and produce reports."""
    if model is None:
        model, idx_to_class, device = load_model()

    _, val_loader, meta = get_dataloaders()
    if idx_to_class is None:
        idx_to_class = meta["idx_to_class"]

    all_preds = []
    all_labels = []

    for images, labels in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Class names in order (resolve int or str keys)
    num_classes = len(idx_to_class)
    keys = list(idx_to_class.keys())
    if isinstance(keys[0], str):
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    # ── Overall classification report ────────────────────────────────────
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    overall_header = (
        f"Test Results:\n"
        f"Accuracy: {acc:.4f}\n"
        f"Precision (weighted): {prec:.4f}\n"
        f"Recall (weighted): {rec:.4f}\n"
        f"F1-score (weighted): {f1:.4f}\n\n"
        f"Classification Report:\n"
    )
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names, digits=4, zero_division=0)
    full_report = overall_header + report

    print("\n" + "=" * 80)
    print("  OVERALL CLASSIFICATION REPORT")
    print("=" * 80)
    print(full_report)

    report_path = os.path.join(config.LOG_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(full_report)
    print(f"Overall report saved → {report_path}")

    # ── Per-crop reports ──────────────────────────────────────────────────
    crop_names = sorted({name.split(" - ")[0] for name in class_names})
    for crop in crop_names:
        _save_crop_report(crop, all_labels, all_preds, idx_to_class)

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
