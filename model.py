"""
Custom CNN Architecture for Multi-Crop Disease Detection.

Architecture: CropDiseaseNet
  - 5 Convolutional blocks (Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU → MaxPool)
  - Squeeze-and-Excitation (SE) attention in deeper blocks
  - Global Average Pooling
  - Classifier head with Dropout

Designed for ~8 000 images / 33 classes — large enough to learn,
small enough to avoid severe overfitting.
"""

import torch
import torch.nn as nn


# ─── Squeeze-and-Excitation Block ────────────────────────────────────────────
class SEBlock(nn.Module):
    """Channel attention via Squeeze-and-Excitation."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# ─── Convolutional Block ─────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Double-conv block: Conv→BN→ReLU → Conv→BN→ReLU → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, use_se: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        self.conv = nn.Sequential(*layers)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        x = self.pool(x)
        return x


# ─── Main Model ──────────────────────────────────────────────────────────────
class CropDiseaseNet(nn.Module):
    """
    Custom CNN for multi-crop, multi-disease classification.

    Input  : (B, 3, 224, 224)
    Output : (B, num_classes)

    Feature extractor spatial sizes (224 input):
        Block 1:  224 → 112
        Block 2:  112 →  56
        Block 3:   56 →  28
        Block 4:   28 →  14
        Block 5:   14 →   7
    Then Global Average Pool → 512-d vector → classifier.
    """

    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            ConvBlock(3, 32, use_se=False),       # Block 1
            ConvBlock(32, 64, use_se=False),      # Block 2
            ConvBlock(64, 128, use_se=True),      # Block 3  + SE
            ConvBlock(128, 256, use_se=True),     # Block 4  + SE
            ConvBlock(256, 512, use_se=True),     # Block 5  + SE
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        # Weight initialisation (Kaiming for Conv, Xavier for Linear)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ─── Helpers ──────────────────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(num_classes: int, dropout: float = 0.5):
    """Factory function to create the model."""
    model = CropDiseaseNet(num_classes=num_classes, dropout=dropout)
    print(f"CropDiseaseNet created — {count_parameters(model):,} trainable parameters")
    return model


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_model(num_classes=33)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Input shape : {dummy.shape}")
    print(f"Output shape: {out.shape}")
