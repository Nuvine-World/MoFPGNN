# chemff/chemprop/model.py
#
# GPU-Optimised D-MPNN Encoder (vectorised with scatter_add)
# ===========================================================
# All Python for-loops over atoms/bonds have been replaced with
# vectorised scatter_add operations for full GPU utilisation.
#
# Architecture:
#   Step 1: h0[vw] = ReLU( W_i * [x_v || e_vw] )
#   Step 2: T rounds of message passing (scatter_add, no Python loops)
#             m[vw] = sum_{u in N(v)\w} h[uw]
#                   = atom_incoming_sum[v] - h[rev(vw)]
#             h[vw] = ReLU( h0[vw] + W_m * m[vw] )
#   Step 3: Readout — scatter_add bond states into atoms, sum-pool per molecule
#   Step 4: FFN
# ===========================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .featurizer import ATOM_FDIM, BOND_FDIM, MolGraph


class DMPNNLayer(nn.Module):
    """
    Single directed message-passing layer (fully vectorised).

    m[vw]  = atom_incoming_sum[v] - h[rev(vw)]
    h[vw]' = ReLU( h0[vw] + W_m * m[vw] )
    """

    def __init__(self, hidden_size: int = 300, dropout: float = 0.0):
        super().__init__()
        self.W_m = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,           # (n_bonds, hidden)
        h0: torch.Tensor,          # (n_bonds, hidden)
        b2a: torch.Tensor,         # (n_bonds,) source atom
        b2revb: torch.Tensor,      # (n_bonds,) reverse bond index
        n_atoms: int,              # total atoms in batch
    ) -> torch.Tensor:
        """Vectorised message passing — no Python loops."""
        hidden = h.size(1)

        # Destination atom for each bond: dest(vw) = source(rev(vw)) = b2a[b2revb]
        b2dest = b2a[b2revb]  # (n_bonds,)

        # Per-atom sum of all incoming bond hidden states
        # atom_sum[a] = sum of h[b] for all bonds b that point TO atom a
        atom_sum = torch.zeros(n_atoms, hidden, device=h.device, dtype=h.dtype)
        atom_sum.scatter_add_(0, b2dest.unsqueeze(1).expand_as(h), h)

        # Message for bond vw = (sum of incoming to v) - h[reverse(vw)]
        # Source atom of bond vw is b2a[vw] = v
        m = atom_sum[b2a] - h[b2revb]  # (n_bonds, hidden)

        h_new = F.relu(h0 + self.W_m(m))
        h_new = self.dropout(h_new)
        return h_new


class ChempropEncoder(nn.Module):
    """
    D-MPNN encoder — fully vectorised for GPU.

    All scatter operations run on the same device as the input tensors,
    enabling full A100 utilisation with AMP.
    """

    def __init__(
        self,
        hidden_size: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        atom_fdim: int = ATOM_FDIM,
        bond_fdim: int = BOND_FDIM,
        output_dim: int = 300,
    ):
        super().__init__()
        self.W_i = nn.Linear(atom_fdim + bond_fdim, hidden_size, bias=False)
        self.message_layers = nn.ModuleList([
            DMPNNLayer(hidden_size, dropout) for _ in range(depth)
        ])
        self.W_a = nn.Linear(atom_fdim + hidden_size, hidden_size, bias=False)
        self.readout_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim),
        )
        self.output_dim = output_dim

    def forward(
        self,
        f_atoms: torch.Tensor,       # (total_atoms, atom_fdim)
        f_bonds: torch.Tensor,       # (total_bonds, atom_fdim + bond_fdim)
        b2a: torch.Tensor,           # (total_bonds,)
        b2revb: torch.Tensor,        # (total_bonds,)
        a_scope: torch.Tensor,       # (batch_size, 2) — [start, size] per molecule
        mol_atom_idx: torch.Tensor,  # (total_atoms,) — molecule index for each atom
    ) -> torch.Tensor:
        """
        Fully vectorised D-MPNN forward pass.

        Returns
        -------
        mol_vecs : torch.Tensor, shape (batch_size, output_dim)
        """
        n_atoms = f_atoms.size(0)
        n_bonds = f_bonds.size(0)

        if n_bonds == 0:
            # Edge case: no bonds — just pool atom features through FFN
            batch_size = a_scope.size(0)
            return torch.zeros(batch_size, self.output_dim, device=f_atoms.device, dtype=f_atoms.dtype)

        # ---- Step 1: Initialise bond hidden states ----
        h0 = F.relu(self.W_i(f_bonds))   # (n_bonds, hidden)
        h = h0.clone()

        # ---- Step 2: T rounds of vectorised message passing ----
        for layer in self.message_layers:
            h = layer(h, h0, b2a, b2revb, n_atoms)

        # ---- Step 3: Readout — vectorised scatter_add ----
        hidden = h.size(1)

        # Destination atom for each bond
        b2dest = b2a[b2revb]  # (n_bonds,)

        # Aggregate final bond states into atoms
        atom_msgs = torch.zeros(n_atoms, hidden, device=h.device, dtype=h.dtype)
        atom_msgs.scatter_add_(0, b2dest.unsqueeze(1).expand_as(h), h)

        atom_h = F.relu(self.W_a(torch.cat([f_atoms, atom_msgs], dim=1)))
        atom_h = self.readout_dropout(atom_h)

        # ---- Global sum pooling (vectorised with scatter_add) ----
        batch_size = a_scope.size(0)
        mol_vecs = torch.zeros(batch_size, hidden, device=atom_h.device, dtype=atom_h.dtype)
        mol_vecs.scatter_add_(0, mol_atom_idx.unsqueeze(1).expand_as(atom_h), atom_h)

        # ---- Step 4: FFN ----
        return self.ffn(mol_vecs)
