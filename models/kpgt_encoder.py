import torch.nn as nn


class PretrainedEmbeddingRegressor(nn.Module):
    """
    Regressor trained on pre-extracted KPGT embeddings.

    Builds a ReLU feedforward head from an explicit list of hidden-layer sizes,
    matching MorganOnlyMLP / PretrainedKPGTMorganHybrid so that Options A/B/C can
    share the same MLP head for a head-matched ablation.

    Args:
        input_dim: dimension of pre-extracted embeddings (KPGT default: 2304)
        mlp_hidden: list of hidden-layer sizes, e.g. [200, 300, 500, 500, 300, 200].
            If None, falls back to a 2-layer [hidden_dim, hidden_dim] head
            (backward-compatible with the old `regressor: {hidden_dim}` configs).
        dropout: dropout probability (0.0 = no dropout)
        hidden_dim: legacy 2-layer width, used only when mlp_hidden is None.
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
