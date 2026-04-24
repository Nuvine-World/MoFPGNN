import torch.nn as nn


class PretrainedEmbeddingRegressor(nn.Module):
    """
    Regressor trained on pre-extracted KPGT embeddings.

    Args:
        input_dim: dimension of pre-extracted embeddings (KPGT default: 2304)
        hidden_dim: hidden layer size (default 512, per paper)
        dropout: dropout probability
    """

    def __init__(self, input_dim=2304, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)
