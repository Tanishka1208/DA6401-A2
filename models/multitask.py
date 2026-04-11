import torch
import torch.nn as nn
import gdown
import os

from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    def __init__(self, num_breeds=37, seg_classes=3, in_channels=3):
        super().__init__()

        classifier_path = "checkpoints/classifier.pth"
        localizer_path = "checkpoints/localizer.pth"
        unet_path = "checkpoints/unet.pth"

        os.makedirs("checkpoints", exist_ok=True)

        # Download
        if not os.path.exists(classifier_path):
            gdown.download(id="1CcLgl5ppsZJtiPeMv3fhe_JV8rXUM06Y", output=classifier_path, quiet=False)

        if not os.path.exists(localizer_path):
            gdown.download(id="1Zzla7VFndq9NFJVyagsz8W36doJCBaoi", output=localizer_path, quiet=False)

        if not os.path.exists(unet_path):
            gdown.download(id="18e7k0Wg9eeoVrQxprPh1q0T_ntM5QA9m", output=unet_path, quiet=False)

        # ========================
        # LOAD FULL MODELS
        # ========================
        self.classifier = VGG11Classifier(num_classes=num_breeds)
        self.localizer = VGG11Localizer()
        self.segmenter = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        # 🔥 LOAD WEIGHTS (MOST IMPORTANT)
        self.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
        self.localizer.load_state_dict(torch.load(localizer_path, map_location="cpu"))
        self.segmenter.load_state_dict(torch.load(unet_path, map_location="cpu"))

        # eval mode
        self.classifier.eval()
        self.localizer.eval()
        self.segmenter.eval()

    def forward(self, x):
        # ========================
        # CLASSIFICATION
        # ========================
        logits = self.classifier(x)
        labels = torch.argmax(logits, dim=1)

        # ========================
        # LOCALIZATION
        # ========================
        boxes = self.localizer(x)

        # 👉 convert to (cx, cy, w, h) if needed
        if boxes.shape[1] == 4:
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            boxes = torch.stack([cx, cy, w, h], dim=1)

        # ========================
        # SEGMENTATION
        # ========================
        seg_logits = self.segmenter(x)
        masks = torch.argmax(seg_logits, dim=1)

        return {
            "classification": labels,
            "localization": boxes,
            "segmentation": masks,
        }