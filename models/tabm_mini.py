import torch
import torch.nn as nn


class TabM_mini(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_layers: int = 2,
        dim: int = 512,
        k: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_layers = n_layers
        self.dim = dim
        self.k = k
        self.dropout = dropout

        init_r = torch.randint(0, 2, (k, in_features)).float() * 2.0 - 1.0
        self.R = nn.Parameter(init_r)

        # Backbone MLPs
        layers = []
        current_dim = in_features
        for _ in range(n_layers):
            layers.append(nn.Linear(current_dim, dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            current_dim = dim
        self.backbone = nn.Sequential(*layers)

        self.head_weight = nn.Parameter(torch.empty(k, dim, out_features))  # (k, dim, out_features)
        self.head_bias = nn.Parameter(torch.zeros(k, 1, out_features))  # (k, 1, out_features)

        nn.init.kaiming_uniform_(self.head_weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # (B, k, in_features)
        x_k = x.unsqueeze(1).expand(-1, self.k, -1)

        # (B, k, in_features)
        h = x_k * self.R

        h = h.reshape(B * self.k, self.in_features)
        h = self.backbone(h)
        h = h.reshape(B, self.k, self.dim)  # (B, k, dim)

        # (k, B, dim)
        h = h.permute(1, 0, 2)
        out = torch.bmm(h, self.head_weight) + self.head_bias  # (k, B, out_features)
        out = out.permute(1, 0, 2)  # (B, k, out_features)

        return out
