from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import List

from chemff.chemprop.model import ChempropEncoder
from chemff.chemprop.featurizer import MolGraph, ATOM_FDIM, BOND_FDIM


class FeedforwardHead(nn.Module):
    """Feedforward regression head applied to the concatenated embedding."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = (1024, 512, 256),
        dropout: float = 0.1,
        output_dim: int = 1,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ConcatFFN(nn.Module):
    """
    Full ChemFF model:
        Chemprop D-MPNN --+
                          +-- concat -> FeedforwardHead -> prediction
        Morgan FP      ---+
    """

    def __init__(
        self,
        chemprop_hidden_size: int = 300,
        chemprop_depth: int = 3,
        chemprop_dropout: float = 0.0,
        morgan_n_bits: int = 2048,
        ffn_hidden_layers: List[int] = (1024, 512, 256),
        ffn_dropout: float = 0.1,
        output_dim: int = 1,
    ):
        super().__init__()

        self.chemprop_encoder = ChempropEncoder(
            hidden_size=chemprop_hidden_size,
            depth=chemprop_depth,
            dropout=chemprop_dropout,
            output_dim=chemprop_hidden_size,
        )

        self.morgan_proj = None
        concat_dim = chemprop_hidden_size + morgan_n_bits

        self.ffn_head = FeedforwardHead(
            input_dim=concat_dim,
            hidden_layers=list(ffn_hidden_layers),
            dropout=ffn_dropout,
            output_dim=output_dim,
        )

    def forward(
        self,
        graphs: List[MolGraph],
        morgan_fps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        graphs     : list[MolGraph], length = batch_size
        morgan_fps : torch.Tensor, shape (batch_size, morgan_n_bits)

        Returns
        -------
        torch.Tensor, shape (batch_size, output_dim)
        """
        device = morgan_fps.device

        # Batch graphs into GPU-friendly tensors
        f_atoms, f_bonds, b2a, b2revb, a_scope, mol_atom_idx = _batch_graphs(graphs)
        f_atoms = f_atoms.to(device)
        f_bonds = f_bonds.to(device)
        b2a = b2a.to(device)
        b2revb = b2revb.to(device)
        a_scope = a_scope.to(device)
        mol_atom_idx = mol_atom_idx.to(device)

        graph_emb = self.chemprop_encoder(
            f_atoms, f_bonds, b2a, b2revb, a_scope, mol_atom_idx
        )

        if self.morgan_proj is not None:
            morgan_emb = self.morgan_proj(morgan_fps)
        else:
            morgan_emb = morgan_fps

        combined = torch.cat([graph_emb, morgan_emb], dim=1)
        return self.ffn_head(combined)


# ---------------------------------------------------------------------------
# Graph batching helper (tensor-only, no list-of-lists)
# ---------------------------------------------------------------------------


def _batch_graphs(graphs: List[MolGraph]):
    """
    Collate a list of MolGraph objects into batched tensors.

    Returns all-tensor outputs (no Python lists) for full GPU compatibility.

    Returns
    -------
    f_atoms      : torch.Tensor  (total_atoms, ATOM_FDIM)
    f_bonds      : torch.Tensor  (total_bonds, ATOM_FDIM + BOND_FDIM)
    b2a          : torch.LongTensor  (total_bonds,)
    b2revb       : torch.LongTensor  (total_bonds,)
    a_scope      : torch.LongTensor  (batch_size, 2)  — [start, size] per molecule
    mol_atom_idx : torch.LongTensor  (total_atoms,)   — molecule index for each atom
    """
    atom_offset = 0
    bond_offset = 0
    all_f_atoms, all_f_bonds = [], []
    all_b2a, all_b2revb = [], []
    a_scope_list = []
    mol_atom_idx_list = []

    for mol_idx, g in enumerate(graphs):
        all_f_atoms.append(g.f_atoms)
        all_f_bonds.append(g.f_bonds)

        all_b2a.extend([a + atom_offset for a in g.b2a])
        all_b2revb.extend([b + bond_offset for b in g.b2revb])

        a_scope_list.append((atom_offset, g.n_atoms))
        mol_atom_idx_list.extend([mol_idx] * g.n_atoms)

        atom_offset += g.n_atoms
        bond_offset += g.n_bonds

    f_atoms_cat = np.vstack(all_f_atoms) if all_f_atoms else np.zeros((0, ATOM_FDIM))
    f_bonds_cat = (
        np.vstack(all_f_bonds) if all_f_bonds else np.zeros((0, ATOM_FDIM + BOND_FDIM))
    )

    return (
        torch.tensor(f_atoms_cat, dtype=torch.float32),
        torch.tensor(f_bonds_cat, dtype=torch.float32),
        torch.tensor(all_b2a, dtype=torch.long),
        torch.tensor(all_b2revb, dtype=torch.long),
        torch.tensor(a_scope_list, dtype=torch.long),
        torch.tensor(mol_atom_idx_list, dtype=torch.long),
    )
