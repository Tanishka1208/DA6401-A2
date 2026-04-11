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

        os.makedirs("checkpoints", exist_ok=True)

        if not os.path.exists(classifier_path):
            gdown.download(id="1CcLgl5ppsZJtiPeMv3fhe_JV8rXUM06Y", output=classifier_path, quiet=False)

        if not os.path.exists(localizer_path):
            gdown.download(id="1Zzla7VFndq9NFJVyagsz8W36doJCBaoi", output=localizer_path, quiet=False)

        if not os.path.exists(unet_path):
            gdown.download(id="18e7k0Wg9eeoVrQxprPh1q0T_ntM5QA9m", output=unet_path, quiet=False)

        # ========================
        # MODELS
        # ========================
        self.encoder = VGG11Encoder(in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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

        self.segmenter = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        # ========================
        # 🔥 LOAD WEIGHTS (FIX)
        # ========================
        self.load_state_dict(torch.load(classifier_path, map_location="cpu"), strict=False)
        self.load_state_dict(torch.load(localizer_path, map_location="cpu"), strict=False)
        self.segmenter.load_state_dict(torch.load(unet_path, map_location="cpu"))

        self.eval()

    def forward(self, x):
        bottleneck = self.encoder(x)

        pooled = self.avgpool(bottleneck)
        flattened = torch.flatten(pooled, 1)

        classification_logits = self.classifier_head(flattened)

        localization_bbox = self.localization_head(flattened)

        # 👉 FIX bbox format
        x1, y1, x2, y2 = localization_bbox[:, 0], localization_bbox[:, 1], localization_bbox[:, 2], localization_bbox[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        localization_bbox = torch.stack([cx, cy, w, h], dim=1)

        segmentation_logits = self.segmenter(x)

        return {
            "classification": classification_logits,   # ✅ logits
            "localization": localization_bbox,         # ✅ (B,4)
            "segmentation": segmentation_logits,       # ✅ (B,C,H,W)
        }