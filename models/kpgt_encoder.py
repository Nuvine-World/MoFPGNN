import torch.nn as nn


class PretrainedEmbeddingRegressor(nn.Module):
    """Regressor on pre-extracted KPGT embeddings.

    Builds a ReLU feedforward head from `mlp_hidden`; when that is None it falls
    back to a two-layer [hidden_dim, hidden_dim] head, which the older
    `regressor: {hidden_dim}` configs rely on.
    """

    def __init__(self, input_dim=2304, mlp_hidden=None, dropout=0.3,
                 hidden_dim=512):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [hidden_dim, hidden_dim]

        layers = []
        prev_dim = input_dim
        for h in mlp_hidden:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
