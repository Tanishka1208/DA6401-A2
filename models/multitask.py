import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout
from .segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    def __init__(self, num_breeds=37, seg_classes=3, in_channels=3):
        super().__init__()

        # ========================
        # SHARED BACKBONE
        # ========================
        self.encoder = VGG11Encoder(in_channels)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ========================
        # CLASSIFICATION HEAD
        # ========================
        self.classifier_head = nn.Sequential(
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(0.5),

            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(0.5),

            nn.Linear(4096, num_breeds),
        )

        # ========================
        # LOCALIZATION HEAD
        # ========================
        self.localization_head = nn.Sequential(
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(0.5),

            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(0.5),

            nn.Linear(4096, 4),
        )

        # ========================
        # SEGMENTATION (UNET)
        # ========================
        self.segmenter = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

    def forward(self, x):
        # ========================
        # SHARED ENCODER
        # ========================
        bottleneck = self.encoder(x)

        pooled = self.avgpool(bottleneck)
        flattened = torch.flatten(pooled, 1)

        # ========================
        # HEADS
        # ========================
        classification_logits = self.classifier_head(flattened)
        localization_bbox = self.localization_head(flattened)

        # segmentation uses full UNet
        segmentation_logits = self.segmenter(x)

        return {
            "classification": classification_logits,
            "localization": localization_bbox,
            "segmentation": segmentation_logits,
        }