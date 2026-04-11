import torch
import torch.nn as nn
import gdown
import os
from .vgg11 import VGG11Encoder
from .layers import CustomDropout
from .segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    def __init__(self, num_breeds=37, seg_classes=3, in_channels=3):
        super().__init__()
        classifier_path = "checkpoints/classifier.pth"
        localizer_path = "checkpoints/localizer.pth"
        unet_path = "checkpoints/unet.pth"

        # Create checkpoints folder if not exists
        os.makedirs("checkpoints", exist_ok=True)

        # Download weights
        if not os.path.exists(classifier_path):
            gdown.download(id="1CcLgl5ppsZJtiPeMv3fhe_JV8rXUM06Y", output=classifier_path, quiet=False)

        if not os.path.exists(localizer_path):
            gdown.download(id="1Zzla7VFndq9NFJVyagsz8W36doJCBaoi", output=localizer_path, quiet=False)

        if not os.path.exists(unet_path):
            gdown.download(id="18e7k0Wg9eeoVrQxprPh1q0T_ntM5QA9m", output=unet_path, quiet=False)


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