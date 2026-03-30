import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 20.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits,
            target.float(),
            pos_weight=self.pos_weight
        )
