import torch
import torch.nn as nn
import gdown
import os
from .vgg11 import VGG11Encoder
from .layers import CustomDropout
from .segmentation import VGG11UNet
from .classification import VGG11Classifier
from .localization import VGG11Localizer


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
        

        
        self.classifier = VGG11Classifier(num_classes=num_breeds)
        self.localizer = VGG11Localizer()
        self.segmenter = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        # ========================
        # 🔥 LOAD WEIGHTS (FIX)
        # ========================
        # DEBUG: check classifier keys
        # state_dict = torch.load(classifier_path, map_location="cpu")
        # print("Classifier keys sample:", list(state_dict.keys())[:5])

        # # DEBUG: check localizer keys
        # state_dict_loc = torch.load(localizer_path, map_location="cpu")
        # print("Localizer keys sample:", list(state_dict_loc.keys())[:5])



        self.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
        self.localizer.load_state_dict(torch.load(localizer_path, map_location="cpu"))
        self.segmenter.load_state_dict(torch.load(unet_path, map_location="cpu"))

        self.eval()

    def forward(self, x):

        # classification_logits = self.classifier(x)
        # boxes = self.localizer(x)
        # localization_bbox = boxes
        
        classification_logits = self.classifier(x)
        localization_bbox = self.localizer(x)

        # 🔥 FIX SCALE
        if torch.max(localization_bbox) <= 1.5:
            localization_bbox = localization_bbox * 224

        localization_bbox = torch.clamp(localization_bbox, 0, 224)

        segmentation_logits = self.segmenter(x)

        return {
            "classification": classification_logits,   # ✅ logits
            "localization": localization_bbox,         # ✅ (B,4)
            "segmentation": segmentation_logits,       # ✅ (B,C,H,W)
        }