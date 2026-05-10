from __future__ import annotations

import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from chemff.chemprop.featurizer import MolecularGraphFeaturizer, MolGraph
from chemff.morgan.featurizer import MorganFeaturizer


class MoleculeDataset(Dataset):
    """
    Dataset that provides paired (MolGraph, morgan_fp, label) tuples.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings for each molecule.
    labels : list[float] or None
        Regression targets (transfection efficiency).  None for inference.
    morgan_featurizer : MorganFeaturizer
        Pre-initialised Morgan featurizer (or load from saved pickle).
    graph_featurizer : MolecularGraphFeaturizer
        Pre-initialised graph featurizer.
    morgan_fp_dict : dict[str, np.ndarray] or None
        Pre-computed {smiles: fp} dict.  If None, computed on the fly.

    Usage
    -----
    from chemff.dataset import MoleculeDataset
    from chemff.morgan.featurizer import MorganFeaturizer
    from chemff.chemprop.featurizer import MolecularGraphFeaturizer

    dataset = MoleculeDataset(
        smiles_list=smiles,
        labels=targets,
        morgan_featurizer=MorganFeaturizer(),
        graph_featurizer=MolecularGraphFeaturizer(),
    )
    graph, morgan_fp, label = dataset[0]
    """

    def __init__(
        self,
        smiles_list: List[str],
        labels: Optional[List[float]],
        morgan_featurizer: MorganFeaturizer,
        graph_featurizer: MolecularGraphFeaturizer,
        morgan_fp_dict: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.smiles = smiles_list
        self.labels = labels
        self.graph_featurizer = graph_featurizer
        self.morgan_featurizer = morgan_featurizer
        self.morgan_fp_dict = morgan_fp_dict  # optional cache

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns
        -------
        mol_graph : MolGraph
            Molecular graph for the chemprop encoder.
        morgan_fp : torch.Tensor, shape (n_bits,)
            Morgan fingerprint vector.
        label : torch.Tensor, shape (1,)  or  None
        """
        smiles = self.smiles[idx]

        # ---- Molecular graph (chemprop branch) ----
        mol_graph: MolGraph = self.graph_featurizer(smiles)

        # ---- Morgan fingerprint (morgan branch) ----
        if self.morgan_fp_dict is not None:
            fp = self.morgan_fp_dict[smiles]
        else:
            fp = self.morgan_featurizer(smiles)
        morgan_fp = torch.tensor(fp, dtype=torch.float32)

        # ---- Label ----
        label = None
        if self.labels is not None:
            label = torch.tensor([self.labels[idx]], dtype=torch.float32)

        return mol_graph, morgan_fp, label


def load_dataset_from_config(cfg: dict) -> Tuple[pd.DataFrame, List[str], List[float]]:
    """
    Load the CSV and return (dataframe, smiles_list, labels).

    Parameters
    ----------
    cfg : dict  (parsed from config.yaml)

    Returns
    -------
    df : pd.DataFrame
    smiles_list : list[str]
    labels : list[float]
    """
    df = pd.read_csv(cfg["data_path"])
    smiles_list = df["SMILES"].tolist()
    labels = df["Target"].tolist()
    return df, smiles_list, labels


def collate_fn(batch: List[Tuple]) -> Tuple:
    """
    Custom collate for DataLoader — separates graphs from tensors.

    Because MolGraph objects cannot be stacked by default, this function
    returns them as a plain list and stacks only the fixed-size tensors.

    Parameters
    ----------
    batch : list of (MolGraph, morgan_fp, label) tuples

    Returns
    -------
    graphs     : list[MolGraph]         (length = batch_size)
    morgan_fps : torch.Tensor           (batch_size, n_bits)
    labels     : torch.Tensor or None   (batch_size, 1)
    """
    graphs, morgan_fps, labels = zip(*batch)

    morgan_fps = torch.stack(morgan_fps, dim=0)

    if labels[0] is not None:
        labels = torch.stack(labels, dim=0)
    else:
        labels = None

    # graphs stays as a Python list — BatchMolGraph batching is handled
    # inside ChempropEncoder.forward() via a_scope / b_scope
    return list(graphs), morgan_fps, labels
