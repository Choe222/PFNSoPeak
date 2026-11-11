import torch
import torch.nn as nn


class BatchEnsembleBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int = 32,
        dropout: float = 0.1,
        is_first_layer: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.k = k

        # 1. Shared weights
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.kaiming_uniform_(self.W, a=5**0.5)

        # 2. Non-shared adapters R, S, B
        if is_first_layer:
            init_r = torch.randint(0, 2, (k, in_features)).float() * 2.0 - 1.0
        else:
            init_r = torch.ones(k, in_features)

        self.R = nn.Parameter(init_r)
        self.S = nn.Parameter(torch.ones(k, out_features))
        self.B = nn.Parameter(torch.zeros(k, out_features))

        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, k, in_features)
        batch_size, k, _ = x.shape

        # 1. Input adapter R
        h = x * self.R

        # 2. Shared weight matrix W
        h = h.reshape(batch_size * k, self.in_features)
        h = h @ self.W
        h = h.reshape(batch_size, k, self.out_features)

        # 3. Output adapter S and bias B
        h = h * self.S + self.B

        # 4. Activation and regularization
        h = self.relu(h)
        h = self.dropout_layer(h)

        return h


class TabM(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_layers: int = 2,
        dim: int = 128,
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

        # Backbone layers
        self.layers = nn.ModuleList()

        current_dim = in_features
        for i in range(n_layers):
            self.layers.append(
                BatchEnsembleBlock(
                    in_features=current_dim,
                    out_features=dim,
                    k=k,
                    dropout=dropout,
                    is_first_layer=(i == 0),
                )
            )
            current_dim = dim

        # Independent prediction heads
        # W_head: (k, dim, out_features)
        # B_head: (k, 1, out_features)
        self.head_weight = nn.Parameter(torch.empty(k, dim, out_features))
        self.head_bias = nn.Parameter(torch.zeros(k, 1, out_features))

        nn.init.kaiming_uniform_(self.head_weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, in_features)
        # Expand input across k submodels -> (B, k, in_features)
        h = x.unsqueeze(1).expand(-1, self.k, -1)

        # Backbone forward -> (B, k, dim)
        for layer in self.layers:
            h = layer(h)

        # Prediction heads -> convert to (k, B, dim)
        h = h.permute(1, 0, 2)
        out = torch.bmm(h, self.head_weight) + self.head_bias

        # Convert back to (B, k, out_features)
        out = out.permute(1, 0, 2)
        return out
