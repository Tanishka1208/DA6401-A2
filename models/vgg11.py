from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class VGG11Custom(nn.Module):
    def __init__(self, use_bn=True):
        super().__init__()

        def conv_block(in_c, out_c):
            if use_bn:
                return nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True)
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )

        self.features = nn.Sequential(
            conv_block(3, 64),    # conv1
            nn.MaxPool2d(2),

            conv_block(64, 128),  # conv2
            nn.MaxPool2d(2),

            conv_block(128, 256), # conv3 ← IMPORTANT (we hook here)
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, 512),  # adjust if needed
            nn.ReLU(),
            nn.Linear(512, 10)  # change classes if needed
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class VGG11Encoder(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(2)

        # Block 5
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool5 = nn.MaxPool2d(2)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        features = {}

        x = self.block1(x)
        if return_features:
            features["block1"] = x
        x = self.pool1(x)

        x = self.block2(x)
        if return_features:
            features["block2"] = x
        x = self.pool2(x)

        x = self.block3(x)
        if return_features:
            features["block3"] = x
        x = self.pool3(x)

        x = self.block4(x)
        if return_features:
            features["block4"] = x
        x = self.pool4(x)

        x = self.block5(x)
        bottleneck = self.pool5(x)

        if return_features:
            return bottleneck, features
        return bottleneck