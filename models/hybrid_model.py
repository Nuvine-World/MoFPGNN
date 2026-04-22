import torch
import torch.nn as nn


class PretrainedKPGTMorganHybrid(nn.Module):
    """
    Late-fusion hybrid using PRE-EXTRACTED KPGT embeddings + Morgan FPs.

    This is the core MoFPGNN model:
        ŷ = MLP([ kpgt_emb || morgan_fp ])

    where kpgt_emb is a 2304-dim vector from the official pretrained KPGT
    and morgan_fp is a 2048-dim count-based Morgan fingerprint.

    Total input: 2304 + 2048 = 4352 dimensions.

    Args:
        kpgt_dim: dimension of KPGT embeddings (default 2304)
        morgan_dim: dimension of Morgan fingerprint (default 2048)
        mlp_hidden: list of hidden layer sizes for the MLP head
        dropout: dropout probability
    """

    def __init__(self, kpgt_dim=2304, morgan_dim=2048,
                 mlp_hidden=None, dropout=0.3):
        super().__init__()
        self.kpgt_dim = kpgt_dim
        self.morgan_dim = morgan_dim

        if mlp_hidden is None:
            mlp_hidden = [512, 256]

        input_dim = kpgt_dim + morgan_dim
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
        self.mlp = nn.Sequential(*layers)

    def forward(self, kpgt_emb, morgan_fp):
        """
        Args:
            kpgt_emb: (batch_size, kpgt_dim)
            morgan_fp: (batch_size, morgan_dim)
        """
        fused = torch.cat([kpgt_emb, morgan_fp], dim=1)
        return self.mlp(fused)


class MorganOnlyMLP(nn.Module):
    """
    Fingerprint-only MLP baseline (for ablation studies).
    Uses only Morgan fingerprints without graph encoder.

    Args:
        morgan_dim: dimension of Morgan fingerprint (default 2048)
        mlp_hidden: list of hidden layer sizes
        dropout: dropout probability
    """

    def __init__(self, morgan_dim=2048, mlp_hidden=None, dropout=0.3):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [512, 256]

        layers = []
        prev_dim = morgan_dim
        for h in mlp_hidden:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, morgan_fp):
        return self.mlp(morgan_fp)
