import torch
import torch.nn as nn


class GRU_Attention(nn.Module):
    def __init__(self, dimension, hidden_size, num_layers, num_classes, device,
                 bidirectional=True, dropout=0.0):
        super().__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            dimension, hidden_size, num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        D = 2 if bidirectional else 1

        # Attention Layer
        self.attention = nn.Linear(hidden_size * D, 1)

        self.fc = nn.Linear(hidden_size * D, num_classes)

    def forward(self, x):
        D = 2 if self.bidirectional else 1
        h0 = torch.zeros(D * self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.gru(x, h0)  # (B, L, H*D)

        # Attention score
        attn_score = self.attention(out)       # (B, L, 1)
        attn_weight = torch.softmax(attn_score, dim=1)

        # Weighted sum
        context = torch.sum(attn_weight * out, dim=1)  # (B, H*D)

        return self.fc(context), attn_weight
