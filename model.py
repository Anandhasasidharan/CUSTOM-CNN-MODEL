"""
Custom CNN Architecture for Multi-Crop Disease Detection.

Architecture: CropDiseaseNet
  - 3 Convolutional blocks (Conv → BN → ReLU → Conv → BN → Pool)
  - Global Average Pooling
  - Flatten → Linear(128 → 33)

Designed for ~8 000 images / 33 classes — large enough to learn,
small enough to avoid severe overfitting.
"""

import torch
import torch.nn as nn


# ─── Convolutional Block ─────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Double-conv block: Conv→BN→ReLU → Conv→BN→Pool."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


# ─── Main Model ──────────────────────────────────────────────────────────────
class CropDiseaseNet(nn.Module):
    """
    Custom CNN for multi-crop, multi-disease classification.

    Input  : (B, 3, 224, 224)
    Output : (B, num_classes)

    Feature extractor spatial sizes (224 input):
        Block 1 (32 filters):  224 → 112
        Block 2 (64 filters):  112 →  56
        Block 3 (128 filters):  56 →  28
        Block 4 (256 filters):  28 →  14
    Then Global Average Pool → 256-d vector → Dropout → Linear(256→num_classes).
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            ConvBlock(3, 32),        # Block 1  (32 filters)
            ConvBlock(32, 64),       # Block 2  (64 filters)
            ConvBlock(64, 128),      # Block 3  (128 filters)
            ConvBlock(128, 256),     # Block 4  (256 filters)
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier with dropout regularisation
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
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


def build_model(num_classes: int):
    """Factory function to create the model."""
    model = CropDiseaseNet(num_classes=num_classes)
    print(f"CropDiseaseNet created — {count_parameters(model):,} trainable parameters")
    return model


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_model(num_classes=33)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Input shape : {dummy.shape}")
    print(f"Output shape: {out.shape}")
