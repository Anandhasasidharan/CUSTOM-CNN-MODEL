"""
Custom Dataset & DataLoaders for Multi-Crop Disease Detection.

Each subdirectory structure:  crop_folder / disease_folder / images
Labels are formatted as "CropName - DiseaseName" for the combined task.
"""

import os
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

import config


# ─── Build class mapping ─────────────────────────────────────────────────────
def build_class_mapping(dataset_root: str):
    """
    Walk the dataset and return:
      class_to_idx : {"Black_Gram - anthracnose": 0, ...}
      idx_to_class : {0: "Black_Gram - anthracnose", ...}
      samples      : [(image_path, class_idx), ...]
      crop_to_idx  : {"Black_Gram": 0, "Cotton": 1, "Soybean": 2}
      sample_crops : [crop_idx, ...]   (parallel to samples)
    """
    crop_names_map = {
        "Black_Gram_cleaned": "Black_Gram",
        "Cleaned_Cotten_Final": "Cotton",
        "soyabean_final": "Soybean",
    }

    class_names = []
    samples = []
    sample_crops = []

    crop_to_idx = {}
    crop_counter = 0

    for crop_folder in sorted(os.listdir(dataset_root)):
        crop_path = os.path.join(dataset_root, crop_folder)
        if not os.path.isdir(crop_path):
            continue

        crop_name = crop_names_map.get(crop_folder, crop_folder)
        if crop_name not in crop_to_idx:
            crop_to_idx[crop_name] = crop_counter
            crop_counter += 1
        crop_idx = crop_to_idx[crop_name]

        for disease_folder in sorted(os.listdir(crop_path)):
            disease_path = os.path.join(crop_path, disease_folder)
            if not os.path.isdir(disease_path):
                continue

            class_label = f"{crop_name} - {disease_folder}"
            if class_label not in class_names:
                class_names.append(class_label)
            class_idx = class_names.index(class_label)

            for fname in os.listdir(disease_path):
                ext = os.path.splitext(fname)[1].lower()
                if ext in config.VALID_EXTENSIONS:
                    fpath = os.path.join(disease_path, fname)
                    samples.append((fpath, class_idx))
                    sample_crops.append(crop_idx)

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    return class_to_idx, idx_to_class, samples, crop_to_idx, sample_crops


# ─── Transforms ───────────────────────────────────────────────────────────────
def get_train_transforms(img_size: int = config.IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25),
    ])


def get_val_transforms(img_size: int = config.IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ─── Dataset ─────────────────────────────────────────────────────────────────
class CropDiseaseDataset(Dataset):
    """PyTorch Dataset for crop disease images."""

    def __init__(self, samples, transform=None):
        """
        Args:
            samples: list of (image_path, class_idx)
            transform: torchvision transforms
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ─── Split & DataLoaders ─────────────────────────────────────────────────────
def split_dataset(samples, train_ratio=config.TRAIN_RATIO,
                  val_ratio=config.VAL_RATIO, seed=config.RANDOM_SEED):
    """Stratified split into train / val / test."""
    from sklearn.model_selection import train_test_split

    paths, labels = zip(*samples)
    paths, labels = list(paths), list(labels)

    # First split: train vs (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        paths, labels,
        test_size=(1 - train_ratio),
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs test  (50-50 of the remaining)
    relative_val = val_ratio / (val_ratio + (1 - train_ratio - val_ratio))
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - relative_val),
        stratify=y_temp,
        random_state=seed,
    )

    train_samples = list(zip(X_train, y_train))
    val_samples = list(zip(X_val, y_val))
    test_samples = list(zip(X_test, y_test))

    return train_samples, val_samples, test_samples


def make_weighted_sampler(samples):
    """Create a WeightedRandomSampler to handle class imbalance."""
    labels = [s[1] for s in samples]
    class_counts = Counter(labels)
    total = len(labels)
    class_weights = {cls: total / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_dataloaders(dataset_root: str = config.DATASET_ROOT,
                    batch_size: int = config.BATCH_SIZE):
    """
    Build and return train / val / test DataLoaders plus metadata.
    """
    class_to_idx, idx_to_class, samples, crop_to_idx, _ = build_class_mapping(dataset_root)
    num_classes = len(class_to_idx)

    train_samples, val_samples, test_samples = split_dataset(samples)

    print(f"Dataset Summary:")
    print(f"  Total images : {len(samples)}")
    print(f"  Num classes  : {num_classes}")
    print(f"  Train / Val / Test : {len(train_samples)} / {len(val_samples)} / {len(test_samples)}")
    print()

    train_ds = CropDiseaseDataset(train_samples, transform=get_train_transforms())
    val_ds = CropDiseaseDataset(val_samples, transform=get_val_transforms())
    test_ds = CropDiseaseDataset(test_samples, transform=get_val_transforms())

    sampler = make_weighted_sampler(train_samples)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
    )

    meta = {
        "num_classes": num_classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "crop_to_idx": crop_to_idx,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
    }

    return train_loader, val_loader, test_loader, meta
