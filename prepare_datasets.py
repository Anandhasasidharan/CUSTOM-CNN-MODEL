"""
Dataset preparation & verification for the 3-stage pipeline.

Commands
--------
1. Build a small, balanced PlantVillage subset for the from-scratch stage:

    python prepare_datasets.py subset \
        --src  D:/AI/DATASET/PlantVillage-Dataset/color \
        --dst  D:/AI/DATASET/PlantVillage_3Crops \
        --crops Potato Corn Pepper_bell \
        --max-per-class 400

   PlantVillage folder names look like "Potato___Early_blight". This command
   re-writes them as PlantVillage_3Crops/Potato/Early_blight/*.jpg so the
   nested loader and the "Crop - Disease" label format both work.

2. Verify any dataset before training (counts, imbalance, split sanity):

    python prepare_datasets.py verify --root D:/AI/DATASET/PlantDoc-Dataset
    python prepare_datasets.py verify --root D:/AI/DATASET/PlantVillage_3Crops

   `verify` prints the quantities AND writes reports/dataset_<name>.png plus a
   CSV, so the same numbers can go straight into a review pack.

Notes
-----
* PlantVillage has no Black Gram and no Cotton. Its 13 crops are Apple,
  Blueberry, Cherry, Corn, Grape, Peach, Pepper_bell, Potato, Raspberry,
  Soybean, Squash, Strawberry and Tomato.
* PlantDoc ships its own train/test folders. Always prefer them over a random
  split: a random split leaks near-duplicate frames of the same leaf into the
  validation set and inflates accuracy by 10-15 points.
"""

import argparse
import csv
import os
import shutil
import sys

import dataset as ds
import config


PLANTVILLAGE_CROPS = [
    "Apple", "Blueberry", "Cherry", "Corn", "Grape", "Peach", "Pepper_bell",
    "Potato", "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato",
]


# ─── subset ──────────────────────────────────────────────────────────────────
def cmd_subset(args):
    src = args.src
    dst = args.dst

    if not os.path.isdir(src):
        sys.exit(f"Source not found: {src}\n"
                 f"Expected the PlantVillage 'color' folder, e.g.\n"
                 f"  PlantVillage-Dataset/color/Potato___Early_blight/*.JPG")

    wanted = [c.strip() for c in args.crops.split(",") if c.strip()]
    unknown = [c for c in wanted if c not in PLANTVILLAGE_CROPS]
    if unknown:
        sys.exit(f"Not PlantVillage crops: {', '.join(unknown)}\n"
                 f"Available: {', '.join(PLANTVILLAGE_CROPS)}")

    class_dirs = [d for d in sorted(os.listdir(src))
                  if os.path.isdir(os.path.join(src, d))]

    print(f"Source : {src}")
    print(f"Found  : {len(class_dirs)} class folders")
    print(f"Keeping: {', '.join(wanted)}")
    print(f"Cap    : {args.max_per_class or 'none'} images per class\n")

    total = 0
    per_class = {}
    for folder in class_dirs:
        if "___" not in folder:
            continue
        crop, disease = folder.split("___", 1)
        if crop not in wanted:
            continue

        images = ds._image_files(os.path.join(src, folder))
        if not images:
            continue
        if args.max_per_class:
            # Deterministic thinning: evenly spaced picks rather than the first N
            step = max(1, len(images) // args.max_per_class)
            images = images[::step][:args.max_per_class]

        out_dir = os.path.join(dst, crop, disease)
        os.makedirs(out_dir, exist_ok=True)

        for img in images:
            target = os.path.join(out_dir, os.path.basename(img))
            if args.dry_run:
                continue
            if os.path.exists(target) and not args.overwrite:
                continue
            shutil.copy2(img, target)

        per_class[f"{crop} / {disease}"] = len(images)
        total += len(images)
        print(f"  {crop:<12s} {disease:<35s} {len(images):>5d}")

    print(f"\nTotal images: {total}")
    print(f"Classes     : {len(per_class)}")

    if args.dry_run:
        print("\n[dry run] nothing copied.")
        return

    print(f"Written to  : {dst}")
    print(f"\nTrain it with:\n  python train.py --stage scratch --dataset \"{dst}\"")


# ─── verify ──────────────────────────────────────────────────────────────────
def _bar(value, max_value, width=32):
    filled = int(round(width * value / max_value)) if max_value else 0
    return "#" * filled + "." * (width - filled)


def cmd_verify(args):
    root = args.root or config.DATASET_ROOT
    stats = ds.dataset_stats(root, layout=args.layout)

    print("=" * 78)
    print(f"  DATASET: {os.path.basename(stats['root'].rstrip('/\\'))}")
    print("=" * 78)
    print(f"  Path             : {stats['root']}")
    print(f"  Layout           : {stats['layout']}"
          + ("  (official train/test)" if stats["uses_official_split"] else "  (random split)"))
    print(f"  Crops            : {stats['num_crops']}  ({', '.join(stats['crops']) or '-'})")
    print(f"  Classes          : {stats['num_classes']}")
    print(f"  Train / Val      : {stats['num_train']:,} / {stats['num_val']:,}")
    print(f"  Total images     : {stats['num_total']:,}")
    print(f"  Smallest class   : {stats['min_class']}")
    print(f"  Largest class    : {stats['max_class']}")
    print(f"  Imbalance ratio  : {stats['imbalance_ratio']:.2f}x")
    print(f"  Mean per class   : {stats['num_total'] / max(1, stats['num_classes']):.1f}")
    print()

    rows = sorted(stats["per_class"], key=lambda r: -r["total"])
    width = max(len(r["label"]) for r in rows)
    print(f"  {'CLASS'.ljust(width)}  TOTAL  TRAIN   VAL  ")
    print(f"  {'-' * width}  -----  -----  ----  {'-' * 32}")
    for r in rows:
        print(f"  {r['label'].ljust(width)}  {r['total']:>5d}  {r['train']:>5d}  "
              f"{r['val']:>4d}  {_bar(r['total'], stats['max_class'])}")
    print()

    # ── Interpretation, not just numbers ──
    print("  Assessment")
    print("  " + "-" * 74)
    avg = stats["num_total"] / max(1, stats["num_classes"])
    if avg < 60:
        print("  [!] Fewer than 60 images/class. A CNN trained from scratch will")
        print("      overfit. Use transfer learning for this dataset.")
    elif avg < 200:
        print("  [~] 60-200 images/class. From-scratch training needs strong")
        print("      augmentation; transfer learning will clearly win.")
    else:
        print("  [+] 200+ images/class. Reasonable for from-scratch training.")

    if stats["imbalance_ratio"] > 5:
        print(f"  [!] Imbalance {stats['imbalance_ratio']:.1f}x. Keep the")
        print("      class-balanced sampler enabled (default).")
    else:
        print(f"  [+] Imbalance {stats['imbalance_ratio']:.1f}x is manageable.")

    if not stats["uses_official_split"] and "augment" in stats["root"].lower():
        print("  [!] Path suggests offline augmentation with a random split.")
        print("      Split by original image before augmenting, or validation")
        print("      accuracy will be inflated by 10-15 points.")

    print()

    # ── Persist the quantities ──
    out_dir = os.path.join(config.BASE_DIR, "reports")
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(stats["root"].rstrip("/\\"))

    csv_path = os.path.join(out_dir, f"dataset_{name}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "crop", "train", "val", "total", "share_pct"])
        for r in rows:
            share = 100.0 * r["total"] / max(1, stats["num_total"])
            w.writerow([r["label"], r["crop"], r["train"], r["val"], r["total"],
                        f"{share:.2f}"])
    print(f"  CSV  -> {csv_path}")

    png = _plot_distribution(stats, name, out_dir)
    if png:
        print(f"  Plot -> {png}")

    return stats


def _plot_distribution(stats, name, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    rows = sorted(stats["per_class"], key=lambda r: r["total"])
    labels = [r["label"] for r in rows]
    train = [r["train"] for r in rows]
    val = [r["val"] for r in rows]

    height = max(3.0, 0.32 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(11, height))

    y = range(len(rows))
    ax.barh(list(y), train, color="#2f6f4e", label="Train")
    ax.barh(list(y), val, left=train, color="#8fbf9f", label="Val")

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Images")
    ax.set_title(f"{name} — class distribution "
                 f"({stats['num_total']:,} images, {stats['num_classes']} classes, "
                 f"{stats['imbalance_ratio']:.1f}x imbalance)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"dataset_{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Build and verify datasets for the 3-stage pipeline.")
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("subset", help="Build a small PlantVillage subset")
    ps.add_argument("--src", required=True,
                    help="PlantVillage 'color' folder (contains Crop___Disease dirs)")
    ps.add_argument("--dst", required=True, help="Destination root")
    ps.add_argument("--crops", default="Potato,Corn,Pepper_bell",
                    help=f"Comma-separated crops (default: Potato,Corn,Pepper_bell). "
                         f"Available: {', '.join(PLANTVILLAGE_CROPS)}")
    ps.add_argument("--max-per-class", type=int, default=None,
                    help="Cap images per class (e.g. 400 for a ~4k-image subset)")
    ps.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    ps.add_argument("--dry-run", action="store_true", help="Report without copying")
    ps.set_defaults(func=cmd_subset)

    pv = sub.add_parser("verify", help="Print and plot dataset quantities")
    pv.add_argument("--root", default=None, help="Dataset root (default: config)")
    pv.add_argument("--layout", default="auto",
                    choices=["auto", "nested", "flat", "presplit"])
    pv.set_defaults(func=cmd_verify)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
