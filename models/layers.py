import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In eval mode OR p=0 → no dropout
        if not self.training or self.p == 0:
            return x

        # Create mask
        mask = (torch.rand_like(x) > self.p).float()

        # Inverted dropout scaling
        return x * mask / (1 - self.p)
